"""Open-loop multi-turn load generator (extends the Chapter 8 load generator to agent tasks).

Tasks arrive as a Poisson process at each rate; every task runs the agent loop to completion. Results per task
and per call are written to results/<exp>_<gpu>_<rate>_<rep>.jsonl, and summarize() computes one row of the results table.

    VLM_URL=... python loadgen_agent.py --exp E2 --gpu H100
    VLM_URLS=url1,url2,url3,url4 python loadgen_agent.py --exp E8 --policy prefix
"""
import argparse
import json
import os
import random
import threading
import time

import numpy as np
import yaml

from agent import Agent, Tracer, single_pass
from retrieval_client import RemoteRetrieval, load_pages
from serve_vlm import resolve


def run_rate(cfg, rate, duration, warmup, seed, questions, make_agent, out_path):
    rng = random.Random(seed)
    results, threads, t0 = [], [], time.time()
    lock = threading.Lock()

    def one(q, t_arrive):
        ag = make_agent()
        try:
            r = single_pass(ag, q["query"]) if cfg.get("mode") == "single_pass" else ag.run(q["query"])
            r.update(query_id=q["query_id"], arrive=t_arrive, ok=True)
        except Exception as e:                          # a failed task counts as missing its objective
            r = {"query_id": q["query_id"], "arrive": t_arrive, "ok": False, "error": str(e)}
        with lock:
            results.append(r)

    t = 0.0
    while t < duration:
        t += rng.expovariate(rate)
        time.sleep(max(0.0, t0 + t - time.time()))
        th = threading.Thread(target=one, args=(rng.choice(questions), t)); th.start(); threads.append(th)
    for th in threads:
        th.join()
    kept = [r for r in results if r["arrive"] >= warmup]
    with open(out_path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in kept)
    return kept


def summarize(rows, objective_s):
    ok = [r for r in rows if r.get("ok")]
    tt = np.array([r["task_time_s"] for r in ok]) if ok else np.array([np.inf])
    hit = sum(r["cached_tokens"] for r in ok) / max(sum(r["prompt_tokens"] for r in ok), 1)
    return {"task_p50_s": float(np.percentile(tt, 50)), "task_p95_s": float(np.percentile(tt, 95)),
            "steps": float(np.mean([r["calls"] for r in ok])) if ok else None, "hit_rate": 100 * hit,
            "met_objective": float(np.mean([r.get("ok") and r["task_time_s"] <= objective_s for r in rows]))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2"); ap.add_argument("--gpu", default="H100")
    ap.add_argument("--policy", default=None); ap.add_argument("--rates", default=None)
    ap.add_argument("--repeats", type=int, default=None); ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--warmup", type=float, default=None)   # defaults: the load block of experiments.yaml
    ap.add_argument("--stop-after-fail", action="store_true",
                    help="rates ascending: stop after the first rate at which a repeat met the objective for < 90%% of tasks")
    a = ap.parse_args()
    full = yaml.safe_load(open("configs/experiments.yaml"))
    cfg, load = resolve(a.exp, "configs/experiments.yaml"), full["load"]
    qs = [json.loads(l) for l in open("data/queries.jsonl")]
    retrieval, pages = RemoteRetrieval(), load_pages("data")
    urls = os.environ.get("VLM_URLS", os.environ.get("VLM_URL", "")).split(",")
    os.makedirs("results", exist_ok=True)
    rates = [float(x) for x in a.rates.split(",")] if a.rates else load["rates_tasks_per_s"]
    for rate in rates:
        failed = False
        for rep in range(a.repeats or load["repeats"]):
            tag = f"{a.exp}_{a.gpu}_{a.policy or 'single'}_{rate}_{rep}"
            tracer = Tracer(f"results/{tag}_trace.jsonl")
            if len(urls) == 1:
                make = lambda: Agent(cfg, retrieval, pages, tracer, urls[0] + "/v1")
            else:
                from routed_client import routed_agent_factory
                make = routed_agent_factory(cfg, retrieval, pages, tracer, urls, a.policy)
            rows = run_rate(cfg, rate, a.duration or load["duration_s"],
                            load["warmup_s"] if a.warmup is None else a.warmup, load["seed"] + rep, qs, make,
                            f"results/{tag}.jsonl")
            sm = summarize(rows, full["objectives"]["task_p95_single_s"])
            print(tag, json.dumps(sm), flush=True)
            failed = failed or sm["met_objective"] < 0.9
        if a.stop_after_fail and failed:
            print("stop after fail at rate", rate, flush=True); break
