"""Chapter 1: all-MiniLM-L6-v2 embedding and FAISS index build times on CPU and GPU.

Corpus: the case study's 2,000 WikiText-2 passages (wikitext-2-raw-v1 train paragraphs of 200 to 800
characters, first 2,000, as in chapter02_lab.ipynb). Embeddings are normalized, FP32 (the
sentence-transformers default), max_seq_length 256 (the model's limit). Index: faiss.IndexFlatIP.

    python minilm_timings.py --device cpu --runs 3 --out cpu.json [--threads 4]
    python minilm_timings.py --device cuda --runs 3 --out t4.json
Run 0 downloads and caches the model and dataset and is not counted. Runs 1..N each start a fresh process.
"""
import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCHES = (32, 128)


def corpus():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    return [t.strip() for t in ds["text"] if 200 < len(t) < 800][:2000]


def one_run(device, threads):
    import torch
    if threads:
        torch.set_num_threads(threads)
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    passages = corpus()
    t0 = time.perf_counter()
    m = SentenceTransformer(MODEL, device=device)
    load_s = time.perf_counter() - t0
    m.max_seq_length = 256
    lens = [len(x) for x in m.tokenizer(passages[:2000], truncation=True, max_length=256)["input_ids"]]
    out = {"load_s": load_s, "n_passages": len(passages), "torch_threads": torch.get_num_threads(),
           "tokens_per_passage_mean": statistics.mean(lens), "tokens_per_passage_max": max(lens),
           "truncated_at_256": sum(1 for x in lens if x >= 256), "batches": {}}
    t0 = time.perf_counter()
    m.encode(passages[:8], batch_size=8, convert_to_numpy=True); warm = time.perf_counter() - t0
    out["warmup_8_passages_s"] = warm
    for bs in BATCHES:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        emb = m.encode(passages, batch_size=bs, convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=False)
        if device == "cuda":
            torch.cuda.synchronize()
        embed_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb.astype(np.float32))
        add_s = time.perf_counter() - t0
        out["batches"][str(bs)] = {"embed_s": embed_s, "faiss_add_s": add_s, "index_total_s": embed_s + add_s,
                                   "passages_per_s": len(passages) / embed_s, "ntotal": index.ntotal,
                                   "dim": int(emb.shape[1])}
    out["end_to_end_bs32_s"] = load_s + warm + out["batches"]["32"]["index_total_s"]
    return out


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


def env_info(device, cmd):
    import faiss
    import sentence_transformers
    import torch
    import transformers
    e = {"date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "command": cmd,
         "device": device, "python": platform.python_version(), "torch": torch.__version__,
         "torch_cuda": torch.version.cuda, "transformers": transformers.__version__,
         "sentence_transformers": sentence_transformers.__version__, "faiss": faiss.__version__,
         "os": platform.platform(), "cpu_count_visible": os.cpu_count()}
    try:
        import triton
        e["triton"] = triton.__version__
    except Exception:  # noqa: BLE001
        e["triton"] = "not installed"
    if sys.platform == "darwin":
        e["host_cpu"] = subprocess.getoutput("sysctl -n machdep.cpu.brand_string")
        e["host_cores"] = subprocess.getoutput("sysctl -n hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu").split()
        e["host_memory_gb"] = int(subprocess.getoutput("sysctl -n hw.memsize")) / 2 ** 30
    else:
        e["host_cpu"] = cpu_model()
        for f in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
            if os.path.exists(f):
                e["cgroup_cpu_limit"] = open(f).read().strip()
    if device == "cuda":
        q = subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
        e["gpu_name"], e["gpu_memory"], e["driver"] = [x.strip() for x in q.split(",")]
        e["gpu_form_factor"] = "PCIe" if "T4" in e["gpu_name"] else "see name"
        e["driver_cuda_version"] = subprocess.getoutput("nvidia-smi | grep -oE 'CUDA (UMD )?Version: [0-9.]+'") \
            .split(":")[-1].strip()   # newer drivers print "CUDA UMD Version" 
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args()
    if a.child:
        print("RESULT " + json.dumps(one_run(a.device, a.threads)), flush=True)
        return
    cmd = " ".join(["python"] + sys.argv)
    runs = []
    for i in range(a.runs + 1):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", "--device", a.device,
                            "--threads", str(a.threads)], capture_output=True, text=True)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            raise SystemExit(p.stderr[-3000:])
        r = json.loads(line[7:]); r["run"] = i; r["counted"] = i > 0
        runs.append(r)
        print(i, {b: round(v["embed_s"], 2) for b, v in r["batches"].items()}, flush=True)
    counted = [r for r in runs if r["counted"]]
    med = {"load_s": statistics.median(r["load_s"] for r in counted),
           "end_to_end_bs32_s": statistics.median(r["end_to_end_bs32_s"] for r in counted)}
    for bs in map(str, BATCHES):
        for k in ("embed_s", "faiss_add_s", "index_total_s", "passages_per_s"):
            vals = [r["batches"][bs][k] for r in counted]
            med[f"bs{bs}_{k}"] = statistics.median(vals)
            med[f"bs{bs}_{k}_range"] = [min(vals), max(vals)]
    res = {"env": env_info(a.device, cmd), "model": MODEL, "median_of_counted_runs": med, "runs": runs}
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps(med, indent=1))


if __name__ == "__main__":
    main()
