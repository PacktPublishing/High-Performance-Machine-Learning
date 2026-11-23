"""Chapter 11 case study: all H100 rows on Modal in one command (November 2026 plan).

  cd chapter11 && modal run h100/modal_h100_all.py

Runs five jobs in parallel, each on one H100:
  gemma, llama      PyTorch eager / torch.compile (static cache) / vLLM 0.33.0 BF16 / vLLM FP8 weights,
                    median of 3 repeats (colab/gpu_reference_v3.py, the same script as the Colab A100 rows)
  det_default,      batch-invariance experiment, concurrency 1/16/64, 100 probes (colab/determinism_colab.py)
  det_invariant
  onnx              Code 11.4 check: torch.export -> ONNX -> ONNX Runtime CUDA EP, prefill time and CPU-placed ops
Writes results/modal_h100_<date>.json. Needs a Modal secret named "huggingface" holding HF_TOKEN.
"""
import json, os, time
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, "..")
app = modal.App("hpml-ch11-h100")
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("huggingface")]
FILES = {"common.py": os.path.join(CH, "common.py"),
         "gpu_reference_v3.py": os.path.join(CH, "colab", "gpu_reference_v3.py"),
         "determinism_colab.py": os.path.join(CH, "colab", "determinism_colab.py")}


def _add(img):
    for name, path in FILES.items():
        img = img.add_local_file(path, f"/root/ch11/{name}")
    return img


# CUDA devel base: vLLM's FlashInfer kernels compile at first use and need nvcc (debian_slim has none;
# the first run on 2026-11-15 failed with "Could not find nvcc").
vllm_img = _add(modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.11")
                .pip_install("vllm==0.33.0", "transformers", "datasets", "httpx", "numpy")
                .env({"CUDA_HOME": "/usr/local/cuda"}))
onnx_img = _add(modal.Image.debian_slim(python_version="3.11")
                .pip_install("torch==2.14.1", "transformers", "onnx", "onnxscript", "onnxruntime-gpu", "datasets",
                             "numpy"))


def _sh(cmd, env):
    import subprocess
    p = subprocess.run(cmd, shell=True, cwd="/root/ch11", env={**os.environ, **env}, capture_output=True, text=True)
    return p.stdout[-4000:] + p.stderr[-2000:]


def _collect(pattern):
    import glob
    return {os.path.basename(f): json.load(open(f)) for f in glob.glob(f"/root/ch11/{pattern}")}


@app.function(image=vllm_img, gpu="H100", secrets=SECRETS, volumes={"/root/.cache/huggingface": hf}, timeout=7200)
def model_rows(model: str, tag: str, torch_rows: bool = True):
    env = {"CH11_MODEL": model, "CH11_TAG": tag, "CH11_REPEATS": "3"}
    log = _sh("python gpu_reference_v3.py torch", env) if torch_rows else ""
    log += _sh("python gpu_reference_v3.py vllm", env)
    log += _sh("CH11_QUANT=fp8 python gpu_reference_v3.py vllm", env)
    return {"files": _collect("case_study_gpu_*.json"), "log": log[-3000:]}


@app.function(image=vllm_img, gpu="H100", secrets=SECRETS, volumes={"/root/.cache/huggingface": hf}, timeout=7200)
def determinism(mode: str):
    log = _sh(f"python determinism_colab.py {mode}", {})
    return {"files": _collect("case_study_determinism_*.json"), "log": log[-3000:]}


@app.function(image=onnx_img, gpu="H100", secrets=SECRETS, volumes={"/root/.cache/huggingface": hf}, timeout=3600)
def onnx_check():
    import sys, traceback
    sys.path.insert(0, "/root/ch11")
    try:
        import numpy as np, torch, onnxruntime as ort
        from common import MAX_PROMPT, load_prompts, chat, environment
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name_ = "google/gemma-2b-it"
        tok = AutoTokenizer.from_pretrained(name_)
        model = AutoModelForCausalLM.from_pretrained(name_, dtype=torch.float16).cuda().eval()
        model.config.use_cache = False
        ids = tok(chat(tok, load_prompts()[0]), return_tensors="pt", padding="max_length", max_length=MAX_PROMPT,
                  truncation=True).input_ids
        x = ids.cuda()
        with torch.no_grad():
            for _ in range(3): model(x)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(10): ref = model(x).logits
            torch.cuda.synchronize()
        torch_ms = (time.perf_counter() - t) / 10 * 1e3
        batch, seq = torch.export.Dim("batch", max=16), torch.export.Dim("seq", max=4096)
        t = time.perf_counter()
        prog = torch.onnx.export(model, (x,), dynamo=True, dynamic_shapes=({0: batch, 1: seq},))
        prog.optimize(); prog.save("/tmp/gemma2b.onnx", external_data=True)
        export_s = time.perf_counter() - t
        so = ort.SessionOptions(); so.enable_profiling = True
        sess = ort.InferenceSession("/tmp/gemma2b.onnx", so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        for _ in range(3): out = sess.run(None, {inp: ids.numpy()})
        t = time.perf_counter()
        for _ in range(10): out = sess.run(None, {inp: ids.numpy()})
        ort_ms = (time.perf_counter() - t) / 10 * 1e3
        ev = json.load(open(sess.end_profiling()))
        cpu = sorted({e["args"].get("op_name") for e in ev if e.get("cat") == "Node"
                      and e.get("args", {}).get("provider") == "CPUExecutionProvider"})
        a = ref[0].float().cpu().numpy(); b = out[0][0].astype(np.float32)
        return {"torch_fp16_prefill_ms": round(torch_ms, 2), "ort_fp16_prefill_ms": round(ort_ms, 2),
                "export_s": round(export_s, 1), "cpu_ops": cpu, "providers": sess.get_providers(),
                "top1_agreement_torch_vs_ort": float((a.argmax(-1) == b.argmax(-1)).mean()),
                "max_abs_logit_diff": float(np.abs(a - b).max()), "env": environment({"gpu": "H100"})}
    except Exception as e:
        return {"error": repr(e)[:500], "trace": traceback.format_exc()[-2000:]}


@app.local_entrypoint()
def main(only: str = "all"):
    """--only vllm reruns just the vLLM rows and the determinism experiment."""
    t0 = time.time()
    full = only != "vllm"
    calls = {"gemma": model_rows.spawn("google/gemma-2b-it", "gemma2b", full),
             "llama": model_rows.spawn("meta-llama/Llama-3.1-8B-Instruct", "llama8b", full),
             "det_default": determinism.spawn("default"), "det_invariant": determinism.spawn("invariant")}
    if full:
        calls["onnx"] = onnx_check.spawn()
    out = {"source": "Modal H100 (modal_h100_all.py)", "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for k, c in calls.items():
        try:
            out[k] = c.get()
        except Exception as e:
            out[k] = {"error": repr(e)[:500]}
        print(f"done: {k} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    path = os.path.join(CH, "results", f"modal_h100_{time.strftime('%Y-%m-%d_%H%M')}_{only}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1, default=str)
    print("saved", path)
