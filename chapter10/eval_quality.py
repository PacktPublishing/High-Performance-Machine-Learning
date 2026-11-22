"""Quality, abstention, forgetting and KL evaluation with vLLM (Chapter 10).

  --task hotpot    exact match / F1 on eval.jsonl, per-item results saved for paired differences
  --task abstain   abstention precision/recall/F1 on abstain.jsonl
  --task forget    lm-evaluation-harness GSM8K (CoT), IFEval, MMLU through vLLM, per-sample logs
  --task kl        per-token KL from the base model on Chapter 9's 200 RAG prompts (prompt_logprobs)
  --task make_probe  generate probe_qa.jsonl (QA pairs) and probe_train.jsonl (article text) from the articles
  --task probe     knowledge probe: closed-book vs retrieved-article accuracy
Adapters are served unmerged through vLLM's LoRA support unless --merged is given.
"""
import argparse
import json
import math
import os
import re
import string
import subprocess
from collections import Counter

from common import DATA, MODELS, NOT_IN_CONTEXT, RESULTS, RUNS, for_model, generation_config_dir, save_json


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred, gold):
    return float(norm(pred) == norm(gold))


def f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    common = Counter(p) & Counter(g); n = sum(common.values())
    if n == 0:
        return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)


def load(name):
    return [json.loads(l) for l in open(f"{DATA}/{name}")]


def adapter_rank(run, default=16):
    p = f"{RUNS}/{run}/adapter_config.json"
    return max(default, json.load(open(p))["r"]) if (run and os.path.exists(p)) else default


def engine(model, run, merged, max_lora_rank=16):
    from vllm import LLM
    gc = generation_config_dir(model)                 # Gemma-2B: stop at <end_of_turn>
    kw = {"generation_config": gc} if gc else {}
    max_lora_rank = adapter_rank(run, max_lora_rank)   # adapters above r=16 need a larger max_lora_rank
    if run and merged:
        return LLM(model=f"{RUNS}/{run}-merged", max_model_len=4096, seed=0, **kw), None
    if run and not os.path.exists(f"{RUNS}/{run}/adapter_config.json"):   # full fine-tuning: the run is a model
        return LLM(model=f"{RUNS}/{run}", max_model_len=4096, seed=0, **kw), None
    llm = LLM(model=MODELS[model], max_model_len=4096, seed=0, enable_lora=bool(run),
              max_lora_rank=max_lora_rank, max_loras=1, **kw)
    lora = None
    if run:
        from vllm.lora.request import LoRARequest
        lora = LoRARequest(run, 1, f"{RUNS}/{run}")
    return llm, lora


def generate(llm, lora, rows, max_tokens=48):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outs = llm.chat([r["messages"] for r in rows], sp, lora_request=lora)
    return [o.outputs[0].text.strip() for o in outs]


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return [100 * m, 100 * sd / math.sqrt(n)]


def make_probe(n_qa=200):
    """Four QA pairs per article from Llama-3.1-8B-Instruct, answers checked to occur in the article text."""
    from vllm import LLM, SamplingParams
    from common import rag_messages
    arts = load("probe_articles.jsonl")
    llm = LLM(model=MODELS["llama-3.1-8b"], max_model_len=8192, seed=0)
    ask = [[{"role": "user", "content": "Write four factual questions that can be answered only from this article, "
             "each with a short answer copied from the article. Output JSON lines {\"question\": ..., \"answer\": ...}."
             f"\n\nTitle: {x['title']}\n\n{x['text']}"}] for x in arts]
    outs = llm.chat(ask, SamplingParams(temperature=0.0, max_tokens=600))
    qa = []
    for x, o in zip(arts, outs):
        for line in o.outputs[0].text.splitlines():
            try:
                d = json.loads(line.strip().rstrip(","))
            except Exception:
                continue
            if d.get("answer") and norm(d["answer"]) in norm(x["text"]):
                qa.append({"question": d["question"], "answer": d["answer"], "title": x["title"],
                           "messages": rag_messages(d["question"], [x["text"][:3000]])})
    with open(f"{DATA}/probe_qa.jsonl", "w") as f:
        for q in qa[:n_qa]:
            f.write(json.dumps(q) + "\n")
    with open(f"{DATA}/probe_train.jsonl", "w") as f:
        for x in arts:
            f.write(json.dumps({"messages": [{"role": "user", "content": f"Write the article about {x['title']}."},
                                             {"role": "assistant", "content": x["text"]}]}) + "\n")
    print("probe QA", len(qa[:n_qa]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", default=None, help="adapter or model run name; omit for the base model")
    ap.add_argument("--merged", action="store_true")
    ap.add_argument("--task", choices=["hotpot", "abstain", "forget", "kl", "make_probe", "probe"], required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--rag-prompts", default=f"{DATA}/ch09_rag_prompts.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="first N items only (0 = all); passed to lm_eval for forget")
    ap.add_argument("--forget-tasks", default="gsm8k_cot:8,ifeval:0,mmlu:5",
                    help="lm_eval tasks as task:fewshot[:limit], comma-separated")
    a = ap.parse_args()
    tag = a.run or f"{a.model}-base"
    out = f"{RESULTS}/eval/{tag}{'-merged' if a.merged else ''}-{a.task}.json"

    if a.task == "forget":
        full = bool(a.run) and not a.merged and not os.path.exists(f"{RUNS}/{a.run}/adapter_config.json")
        target = (f"{RUNS}/{a.run}-merged" if (a.run and a.merged) else f"{RUNS}/{a.run}" if full else MODELS[a.model])
        args = f"pretrained={target},max_model_len=4096,gpu_memory_utilization=0.85"
        gc = generation_config_dir(a.model)
        if gc:
            args += f",generation_config={gc}"
        if a.run and not a.merged and not full:
            args += f",enable_lora=True,max_lora_rank={adapter_rank(a.run, a.rank)},lora_local_path={RUNS}/{a.run}"
        for spec in a.forget_tasks.split(","):
            task, shots, *lim = spec.split(":")
            n = int(lim[0]) if lim else a.limit
            extra = (f"--num_fewshot {shots}" if int(shots) else "") + (f" --limit {n}" if n else "")
            cmd = (f"lm_eval --model vllm --model_args {args} --tasks {task} {extra} --batch_size auto "
                   f"--apply_chat_template --log_samples --output_path {RESULTS}/lmeval/{tag}/{task}")
            subprocess.run(cmd, shell=True, check=True)
        return

    if a.task == "make_probe":
        make_probe(); return
    llm, lora = engine(a.model, a.run, a.merged, a.rank)
    def mload(name):                                 # Gemma-2B: system text folded into the user turn
        return [{**r, "messages": for_model(a.model, r["messages"])} for r in load(name)]
    if a.task == "hotpot":
        rows = mload("eval.jsonl")[:a.limit or None]; preds = generate(llm, lora, rows)
        res = {"em": [em(p, r["answer"]) for p, r in zip(preds, rows)],
               "f1": [f1(p, r["answer"]) for p, r in zip(preds, rows)], "preds": preds}
        res["em_mean"] = 100 * sum(res["em"]) / len(rows)
    elif a.task == "abstain":
        rows = mload("abstain.jsonl")[:a.limit or None]; preds = generate(llm, lora, rows)
        said = [norm(NOT_IN_CONTEXT) in norm(p) for p in preds]
        un = [not r["answerable"] for r in rows]
        tp = sum(s and u for s, u in zip(said, un)); fp = sum(s and not u for s, u in zip(said, un))
        fn = sum((not s) and u for s, u in zip(said, un))
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res = {"precision": pr, "recall": rc, "f1": 100 * 2 * pr * rc / max(pr + rc, 1e-9), "said": said}
    elif a.task == "kl":
        from vllm import SamplingParams
        rows = [{**r, "messages": for_model(a.model, r["messages"])}
                for r in [json.loads(l) for l in open(a.rag_prompts)][:200][:a.limit or None]]
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=20)
        outs = llm.chat([r["messages"] for r in rows], sp, lora_request=lora)
        res = {"prompt_logprobs": [[{str(k): v.logprob for k, v in (pl or {}).items()} for pl in o.prompt_logprobs]
                                   for o in outs]}
    else:  # probe
        qa = mload("probe_qa.jsonl")[:a.limit or None]
        closed = [{"messages": [{"role": "user", "content": q["question"] + " Answer with a short phrase."}]}
                  for q in qa]
        rag = [{"messages": q["messages"]} for q in qa]
        pc, pr_ = generate(llm, lora, closed), generate(llm, lora, rag)
        res = {"closed": [f1(p, q["answer"]) >= 0.5 for p, q in zip(pc, qa)],
               "rag": [f1(p, q["answer"]) >= 0.5 for p, q in zip(pr_, qa)]}
    res["n_items"] = len(rows if a.task != "probe" else qa); res["limit"] = a.limit
    save_json(res, out)
    print("saved", out)


if __name__ == "__main__":
    main()
