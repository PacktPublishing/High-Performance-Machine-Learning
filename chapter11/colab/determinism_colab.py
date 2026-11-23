"""Chapter 11 batch-invariance experiment, Colab version (one GPU, no Modal).

  python determinism_colab.py default      # VLLM_BATCH_INVARIANT=0
  python determinism_colab.py invariant    # VLLM_BATCH_INVARIANT=1

For each concurrency C in CH11_CONC (default 1,16,64): keep C-1 background requests running, send the same
probe prompt CH11_PROBES times (default 100) at temperature 0 with 128 forced output tokens, and count distinct
completions, the share that match the most common completion, and the first token position where any differs.
Output tokens/s over the same window measures the throughput cost of batch invariance.
Results: case_study_determinism_<gpu>_<mode>.json in the current directory.
"""
import asyncio, collections, json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_prompts, chat, environment

MODEL = os.environ.get("CH11_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
CONC = [int(c) for c in os.environ.get("CH11_CONC", "1,16,64").split(",")]
PROBES = int(os.environ.get("CH11_PROBES", "100"))
NEW = 128
mode = sys.argv[1] if len(sys.argv) > 1 else "default"
inv = mode == "invariant"


def gpu_name():
    try:
        n = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                           text=True).stdout.strip().split("\n")[0]
        return "a100" if "A100" in n else "h100" if "H100" in n else "g4" if "PRO 6000" in n else "l4" if "L4" in n else "t4" if "T4" in n else "gpu"
    except Exception:
        return "gpu"


def main():
    import httpx
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [chat(tok, p) for p in load_prompts()]
    probe, others = prompts[0], prompts[1:]
    out_path = f"case_study_determinism_{gpu_name()}_{mode}.json"
    log = open(f"vllm_server_{mode}.log", "w")
    env = dict(os.environ, VLLM_BATCH_INVARIANT="1" if inv else "0")
    t0 = time.time()
    srv = subprocess.Popen(["vllm", "serve", MODEL, "--max-model-len", "4096", "--port", "8000",
                            "--no-enable-prefix-caching", "--max-num-seqs", "64"], env=env, stdout=log, stderr=log)
    ok = False
    for _ in range(450):
        if srv.poll() is not None:
            break
        try:
            if httpx.get("http://localhost:8000/health", timeout=2).status_code == 200:
                ok = True
                break
        except Exception:
            pass
        time.sleep(2)
    if not ok:
        srv.kill(); log.close()
        tail = open(f"vllm_server_{mode}.log").read()[-3000:]
        json.dump({"mode": mode, "error": "server did not start", "log_tail": tail,
                   "env": environment({"batch_invariant": inv})}, open(out_path, "w"), indent=1)
        print(f"{mode}: server failed to start; see {out_path}")
        print(tail[-1500:])
        return
    startup = time.time() - t0
    url = "http://localhost:8000/v1/completions"

    async def one(client, p):
        r = await client.post(url, json={"model": MODEL, "prompt": p, "max_tokens": NEW, "min_tokens": NEW,
                                         "temperature": 0, "logprobs": 0}, timeout=900)
        c = r.json()["choices"][0]
        return c["text"], c["logprobs"]["tokens"]

    async def at(conc):
        stop, bg_tokens = asyncio.Event(), [0]
        limits = httpx.Limits(max_connections=conc + 4)
        async with httpx.AsyncClient(limits=limits) as client:
            await one(client, probe)  # warm-up
            async def background(i):
                while not stop.is_set():
                    await one(client, others[i % len(others)])
                    bg_tokens[0] += NEW
            bg = [asyncio.create_task(background(i)) for i in range(conc - 1)]
            await asyncio.sleep(3 if conc > 1 else 0)  # let the batch fill
            bg_tokens[0] = 0
            t = time.perf_counter(); outs = []
            for _ in range(PROBES):
                outs.append(await one(client, probe))
            dt = time.perf_counter() - t
            stop.set(); await asyncio.gather(*bg)
        texts = [o[0] for o in outs]; toks = [o[1] for o in outs]
        counts = collections.Counter(texts)
        base = toks[0]
        divs = [next((i for i, (a, b) in enumerate(zip(base, x)) if a != b), None) for x in toks[1:]]
        divs = [d for d in divs if d is not None]
        r = {"distinct": len(counts), "share_most_common": round(counts.most_common(1)[0][1] / PROBES, 3),
             "first_div": min(divs) if divs else None, "probe_latency_s": round(dt / PROBES, 3),
             "tput_tok_s": round((bg_tokens[0] + PROBES * NEW) / dt, 1), "probes": PROBES}
        print(f"{mode} C={conc}: {r}", flush=True)
        return r

    res = {}
    for c in CONC:
        res[str(c)] = asyncio.run(at(c))
        json.dump({"mode": mode, "model": MODEL, "startup_s": round(startup, 1), "results": res},
                  open(out_path, "w"), indent=1)
    srv.terminate(); srv.wait(timeout=60); log.close()
    json.dump({"mode": mode, "model": MODEL, "startup_s": round(startup, 1), "results": res,
               "env": environment({"batch_invariant": inv})}, open(out_path, "w"), indent=1)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
