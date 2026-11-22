"""Serving benchmark for LoRA adapters with vLLM 0.33 (Chapter 10).

Starts `vllm serve` for one configuration (base, merged, or unmerged with N adapters), measures single-request TPOT,
then runs an open-loop Poisson load sweep with the Chapter 8 objectives (TTFT 500 ms, TPOT 50 ms) and reports the
highest rate at which at least 90% of requests meet both. Unmerged runs register N copies of the same adapter under
different names so that each request in a batch uses a distinct adapter (worst case for the kernels).
"""
import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import time

import httpx

from common import DATA, MODELS, RESULTS, RUNS, save_json, for_model

def _free_port():
    import socket
    with socket.socket() as s_:
        s_.bind(("", 0)); return s_.getsockname()[1]


# Free port and per-job adapter folder: several benchmark jobs can share a node.
PORT = int(os.environ.get("SERVE_PORT") or _free_port())
URL = f"http://127.0.0.1:{PORT}"
ADAPTER_DIR = f"/tmp/{os.environ.get('USER', 'user')}/serve_bench_{os.getpid()}/adapters"
TTFT_S, TPOT_S = 0.5, 0.05


def start_server(model, run, mode, n_adapters, rank):
    if mode == "merged":
        args = [f"{RUNS}/{run}-merged"]
    else:
        args = [MODELS[model]]
    if mode == "unmerged":
        mods = []
        for i in range(n_adapters):
            dst = f"{ADAPTER_DIR}/a{i}"
            shutil.copytree(f"{RUNS}/{run}", dst, dirs_exist_ok=True)
            mods.append(json.dumps({"name": f"a{i}", "path": dst, "base_model_name": MODELS[model]}))
        args += ["--enable-lora", "--max-loras", str(n_adapters), "--max-lora-rank", str(rank),
                 "--max-cpu-loras", str(max(n_adapters, 16)), "--lora-modules", *mods]
    p = subprocess.Popen(["vllm", "serve", *args, "--port", str(PORT), "--max-model-len", "4096",
                          "--enable-prefix-caching"])
    for _ in range(600):
        try:
            if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
                return p
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("server did not start")


async def one(client, name, messages, max_tokens=128):
    t0 = time.perf_counter(); first = None; n = 0
    body = {"model": name, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
            "ignore_eos": True}
    async with client.stream("POST", f"{URL}/v1/chat/completions", json=body, timeout=120) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                if first is None:
                    first = time.perf_counter()
                n += 1
    end = time.perf_counter()
    return first - t0, (end - first) / max(n - 1, 1)


async def sweep(names, prompts, rates, dur=60):
    out = {}
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=2000)) as c:
        for rate in rates:
            rng = random.Random(0); tasks = []; t = 0.0; start = time.perf_counter()
            while t < dur:
                t += rng.expovariate(rate)
                await asyncio.sleep(max(0, start + t - time.perf_counter()))
                tasks.append(asyncio.create_task(one(c, rng.choice(names), rng.choice(prompts)["messages"])))
            res = await asyncio.gather(*tasks, return_exceptions=True)
            ok = [r for r in res if not isinstance(r, Exception)]
            good = sum(1 for a, b in ok if a <= TTFT_S and b <= TPOT_S) / max(len(res), 1)
            ttft = sorted(a for a, _ in ok)
            out[rate] = {"goodput_share": good, "ttft_p90_ms": 1000 * ttft[int(0.9 * len(ttft))] if ttft else None}
            print(rate, out[rate])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.1-8b")
    ap.add_argument("--run", default=None)
    ap.add_argument("--mode", choices=["base", "merged", "unmerged"], default="base")
    ap.add_argument("--adapters", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--rates", default="4,8,16,24,32,48,64")
    ap.add_argument("--gpu", default="H100")
    a = ap.parse_args()
    prompts = [{**r, "messages": for_model(a.model, r["messages"])}       # Gemma-2B: no system role
               for r in [json.loads(l) for l in open(f"{DATA}/eval.jsonl")][:500]]
    p = start_server(a.model, a.run, a.mode, a.adapters, a.rank)
    try:
        names = ([f"a{i}" for i in range(a.adapters)] if a.mode == "unmerged" else
                 [f"{RUNS}/{a.run}-merged" if a.mode == "merged" else MODELS[a.model]])
        single = [asyncio.run(one(httpx.AsyncClient(), names[0], prompts[i]["messages"])) for i in range(20)]
        tpot = sorted(b for _, b in single)[10] * 1000
        sw = asyncio.run(sweep(names, prompts, [float(x) for x in a.rates.split(",")]))
        cap = max([r for r, v in sw.items() if v["goodput_share"] >= 0.9], default=0)
        res = {"model": a.model, "run": a.run, "mode": a.mode, "adapters": a.adapters, "gpu": a.gpu,
               "tpot_ms": tpot, "capacity_rps": cap, "ttft_p90_ms": sw.get(cap, {}).get("ttft_p90_ms"),
               "sweep": sw, "vllm": subprocess.run(["vllm", "--version"], capture_output=True, text=True).stdout}
        save_json(res, f"{RESULTS}/serve/{a.model}-{a.mode}-{a.adapters}-{a.gpu}.json")
    finally:
        p.terminate()


if __name__ == "__main__":
    main()
