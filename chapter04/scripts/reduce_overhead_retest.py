"""Chapter 4: re-test of torch.compile(mode="reduce-overhead") on a hand-compiled forward with a static cache.

Uses the case study's decode setup (case_study/experiments.py: Gemma-2B-IT in FP16, 512-token prompt,
20 new tokens, greedy). Decode time per token = (time for 20 tokens - time for 1 token) / 19, each
time the median of 10 generate() calls after 2 warm-up calls.

Every variant runs in its own child interpreter, so a native crash (SIGSEGV) in one variant is recorded
as an error with its trace and does not stop the others.

    python reduce_overhead_retest.py --out result.json [--variants eager_dynamic,reduce_overhead,...]
    python reduce_overhead_retest.py --child <variant> <out.json>      # used internally

Variants:
    eager_dynamic          eager, dynamic cache (Table 4.1 row 1)
    eager_static           eager, static cache, transformers' automatic compilation off
    default                model.forward = torch.compile(model.forward, mode="default"), static cache
    reduce_overhead        model.forward = torch.compile(model.forward, mode="reduce-overhead"), static cache
    autocompile            static cache, transformers' automatic compilation on (pattern 2)
    ro_mark_step           reduce_overhead, with torch.compiler.cudagraph_mark_step_begin() before each forward
    ro_decode_only         reduce_overhead graph used only for one-token decode steps; prefill runs eager
    ro_no_partition        reduce_overhead with torch._inductor.config.graph_partition = False
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_STUDY = os.environ.get("HPML_CASE_STUDY", os.path.join(HERE, "..", "..", "case_study"))
ALL_VARIANTS = ["eager_dynamic", "eager_static", "default", "reduce_overhead", "autocompile",
                "ro_mark_step", "ro_decode_only", "ro_no_partition"]


def cpu_model():
    """CPU model from /proc/cpuinfo. Modal's sandbox reports "model name: unknown"; then record vendor,
    family, model number, clock and the AVX-512 flags, which identify the CPU generation."""
    info = {}
    try:
        for ln in open("/proc/cpuinfo"):
            if not ln.strip():
                break
            k, _, v = ln.partition(":")
            info[k.strip()] = v.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
    name = info.get("model name", "unknown")
    if name and name != "unknown":
        return name
    flags = info.get("flags", "").split()
    avx = ",".join(f for f in ("avx512f", "avx512_vnni", "avx512_bf16", "amx_tile") if f in flags) or "no avx512"
    return (f"model name hidden by the sandbox; {info.get('vendor_id', '?')} family {info.get('cpu family', '?')} "
            f"model {info.get('model', '?')}, {info.get('cpu MHz', '?')} MHz, {avx}")


def env_info():
    import platform
    out = {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "python": platform.python_version()}
    q = subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,driver_version,power.max_limit,pci.bus_id "
                             "--format=csv,noheader").splitlines()
    if q:
        name, mem, drv, pmax, _bus = [x.strip() for x in q[0].split(",")]
        ff = "SXM" if "SXM" in name or "HBM3" in name else ("PCIe" if "PCIe" in name else "unknown from name")
        out.update(gpu_name=name, gpu_memory=mem, gpu_form_factor=ff, gpu_power_max_limit=pmax, driver=drv)
    head = subprocess.getoutput("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'")
    out["driver_cuda_version"] = head.replace("CUDA Version:", "").strip()
    out["host_cpu"] = cpu_model()
    out["cpu_count_visible"] = os.cpu_count()
    try:
        import torch
        import transformers
        import triton
        out.update(torch=torch.__version__, torch_cuda=torch.version.cuda, triton=triton.__version__,
                   transformers=transformers.__version__)
        if torch.cuda.is_available():
            out["torch_device_name"] = torch.cuda.get_device_name(0)
            out["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as e:  # noqa: BLE001
        out["import_error"] = repr(e)
    return out


def child(variant, out_path):
    sys.path.insert(0, CASE_STUDY)
    import experiments as ex  # case_study/experiments.py
    torch, model, out, flush, measure = ex._decode_setup(lambda partial: json.dump(partial, open(out_path, "w")))
    off = ex._autocompile_kw() or {}
    out["variant"] = variant
    out["autocompile_switch"] = off
    with torch.inference_mode():
        if variant == "eager_dynamic":
            measure("result", model)
        elif variant == "eager_static":
            measure("result", model, cache_implementation="static", **off)
        elif variant == "autocompile":
            torch._dynamo.reset()
            measure("result", model, cache_implementation="static")
        elif variant in ("default", "reduce_overhead"):
            mode = variant.replace("_", "-")
            model.forward = torch.compile(model.forward, mode=mode)
            measure("result", model, cache_implementation="static", **off)
        elif variant == "ro_no_partition":
            import torch._inductor.config as icfg
            out["graph_partition_before"] = getattr(icfg, "graph_partition", None)
            icfg.graph_partition = False
            model.forward = torch.compile(model.forward, mode="reduce-overhead")
            measure("result", model, cache_implementation="static", **off)
        elif variant == "ro_mark_step":
            compiled = torch.compile(model.forward, mode="reduce-overhead")

            def fwd(*a, **k):
                torch.compiler.cudagraph_mark_step_begin()
                return compiled(*a, **k)
            model.forward = fwd
            measure("result", model, cache_implementation="static", **off)
        elif variant == "ro_decode_only":
            eager_fwd = model.forward
            compiled = torch.compile(model.forward, mode="reduce-overhead")

            def fwd(*a, **k):
                ids = k.get("input_ids", a[0] if a else None)
                if ids is not None and ids.shape[-1] == 1:
                    return compiled(*a, **k)
                return eager_fwd(*a, **k)
            model.forward = fwd
            measure("result", model, cache_implementation="static", **off)
        else:
            raise SystemExit(f"unknown variant {variant}")
    flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2)
    ap.add_argument("--out")
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    if a.child:
        return child(*a.child)

    res = {"command": " ".join([os.path.basename(sys.executable)] + sys.argv), "env": env_info(), "variants": {}}
    trace_dir = a.trace_dir or os.path.dirname(os.path.abspath(a.out))
    os.makedirs(trace_dir, exist_ok=True)
    for v in a.variants.split(","):
        tmp = f"/tmp/ro_child_{v}_{os.getpid()}.json"
        if os.path.exists(tmp):
            os.remove(tmp)
        env = dict(os.environ, PYTHONFAULTHANDLER="1", TORCH_SHOW_CPP_STACKTRACES="1")
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", v, tmp], env=env,
                               capture_output=True, text=True, timeout=a.timeout)
            rc, err = p.returncode, p.stderr
        except subprocess.TimeoutExpired as e:
            rc, err = "timeout", (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        entry = {"returncode": rc, "wall_s": round(time.time() - t0, 1)}
        if os.path.exists(tmp):
            entry["data"] = json.load(open(tmp))
        if rc != 0:
            # keep the end of stderr: Python fault handler stack plus any C++ frames
            tok = os.environ.get("HF_TOKEN")
            if tok:
                err = err.replace(tok, "<HF_TOKEN>")
            lines = err.splitlines()
            entry["signal"] = (-rc if isinstance(rc, int) and rc < 0 else None)
            entry["stderr_tail"] = "\n".join(lines[-60:])
            tpath = os.path.join(trace_dir, f"trace_{v}.txt")
            open(tpath, "w").write(err[-200000:])
            entry["trace_file"] = os.path.basename(tpath)
        res["variants"][v] = entry
        json.dump(res, open(a.out, "w"), indent=2)
        print(v, rc, entry.get("data", {}).get("result", {}).get("decode_ms_per_token"), flush=True)


if __name__ == "__main__":
    main()
