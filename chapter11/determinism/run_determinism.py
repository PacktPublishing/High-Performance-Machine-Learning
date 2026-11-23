"""Chapter 11 batch-invariance experiment (Section 11.5, Table 11.7): Llama-3.1-8B-Instruct, vLLM 0.33.0, H100.

  modal run determinism/run_determinism.py
For each concurrency C in (1, 8, 32, 64) and each mode (default, VLLM_BATCH_INVARIANT=1): start a vLLM server,
keep C-1 background requests running continuously, send the same probe prompt 200 times at temperature 0 with
128 output tokens, and count distinct completions and the first token position where they diverge. Background
throughput (output tokens/s) is measured over the same window.
"""
import json, os, sys
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("hpml-ch11-determinism")
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.11").pip_install("vllm==0.33.0", "httpx", "datasets")
       .add_local_file(os.path.join(HERE, "..", "common.py"), "/root/common.py"))
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@app.function(image=img, gpu="H100", secrets=[modal.Secret.from_name("huggingface")],
              volumes={"/root/.cache/huggingface": hf}, timeout=7200)
def run_mode(batch_invariant: bool):
    import asyncio, subprocess, time, httpx
    sys.path.insert(0, "/root")
    from common import load_prompts, environment
    env = dict(os.environ, VLLM_BATCH_INVARIANT="1" if batch_invariant else "0")
    srv = subprocess.Popen(["vllm", "serve", MODEL, "--max-model-len", "4096", "--port", "8000",
                            "--no-enable-prefix-caching"], env=env)
    url = "http://localhost:8000/v1/completions"
    for _ in range(600):
        try:
            if httpx.get("http://localhost:8000/health", timeout=2).status_code == 200:
                break
        except Exception:
            time.sleep(2)
    prompts = load_prompts()
    probe, others = prompts[0], prompts[1:]

    async def one(client, p, n=128):
        r = await client.post(url, json={"model": MODEL, "prompt": p, "max_tokens": n, "temperature": 0,
                                         "logprobs": 0, "min_tokens": n}, timeout=600)
        c = r.json()["choices"][0]
        return c["text"], c["logprobs"]["tokens"] if c.get("logprobs") else None

    async def at(conc):
        stop, done_tokens = asyncio.Event(), [0]
        async with httpx.AsyncClient() as client:
            async def background(i):
                while not stop.is_set():
                    await one(client, others[i % len(others)]); done_tokens[0] += 128
            bg = [asyncio.create_task(background(i)) for i in range(conc - 1)]
            t = time.perf_counter(); outs = []
            for _ in range(200):
                outs.append(await one(client, probe))
            dt = time.perf_counter() - t
            stop.set(); await asyncio.gather(*bg)
        texts = [o[0] for o in outs]; toks = [o[1] for o in outs]
        base = toks[0]
        firsts = [next((i for i, (a, b) in enumerate(zip(base, t)) if a != b), None) for t in toks[1:]]
        firsts = [f for f in firsts if f is not None]
        return {"distinct": len(set(texts)), "first_div": min(firsts) if firsts else None,
                "tput": (done_tokens[0] + 200 * 128) / dt}

    res = {c: asyncio.run(at(c)) for c in (1, 8, 32, 64)}
    srv.terminate()
    return {"results": res, "env": environment({"batch_invariant": batch_invariant})}


@app.local_entrypoint()
def main():
    d, b = run_mode.remote(False), run_mode.remote(True)
    out = {}
    for c in ("1", "8", "32", "64"):
        rd, rb = d["results"][c] if c in d["results"] else d["results"][int(c)], \
                 b["results"][c] if c in b["results"] else b["results"][int(c)]
        out[c] = {"distinct_default": rd["distinct"], "first_div_default": rd["first_div"],
                  "distinct_bi": rb["distinct"], "tput_default": round(rd["tput"]), "tput_bi": round(rb["tput"])}
    out["env"] = {"default": d["env"], "batch_invariant": b["env"]}
    path = os.path.join(HERE, "..", "..", "..", "results", "determinism.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1)[:1500])
