"""Build the Chapter 10 datasets.

train.jsonl / eval.jsonl : HotpotQA (distractor) as RAG prompts, RAFT-style: 20% of training prompts lose their
                           supporting paragraphs and get the not-in-context target.
abstain.jsonl            : SQuAD 2.0 unanswerable and answerable questions (abstention F1).
probe_articles.jsonl     : Wikipedia articles first created after 2024-01-01 (knowledge probe; QA pairs are generated
                           later by eval_quality.py --make-probe).
Only scripts are distributed; the data is downloaded from the original sources.
"""
import argparse
import json
import os
import random

from common import DATA, NOT_IN_CONTEXT, rag_messages


def hotpot(n_train=10_000, n_eval=2_000, seed=0, no_gold=0.2, max_par_chars=900):
    from datasets import load_dataset
    rng = random.Random(seed)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")

    def row(ex, drop_gold):
        titles = ex["context"]["title"]; sents = ex["context"]["sentences"]
        gold = set(ex["supporting_facts"]["title"])
        pars = [(t, " ".join(s)[:max_par_chars]) for t, s in zip(titles, sents)]
        g = [p for t, p in pars if t in gold]; d = [p for t, p in pars if t not in gold]
        passages = (d[:4] if drop_gold else g + d[:3])
        rng.shuffle(passages)
        ans = NOT_IN_CONTEXT if drop_gold else ex["answer"]
        return {"id": ex["id"], "messages": rag_messages(ex["question"], passages), "answer": ans,
                "gold_answer": ex["answer"], "answerable": not drop_gold}
    train = ds["train"].shuffle(seed=seed).select(range(n_train))
    ev = ds["validation"].shuffle(seed=seed).select(range(n_eval))
    tr = [row(ex, rng.random() < no_gold) for ex in train]
    evr = [row(ex, False) for ex in ev]
    return tr, evr


def squad_abstain(n=1000, seed=0):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation").shuffle(seed=seed)
    out, k_ans, k_un = [], 0, 0
    for ex in ds:
        un = len(ex["answers"]["text"]) == 0
        if (un and k_un >= n // 2) or (not un and k_ans >= n // 2):
            continue
        out.append({"id": ex["id"], "messages": rag_messages(ex["question"], [ex["context"]]),
                    "answer": NOT_IN_CONTEXT if un else ex["answers"]["text"][0], "answerable": not un})
        k_un += un; k_ans += (not un)
        if len(out) >= n:
            break
    return out


def probe_articles(n=60, after="2024-01-01T00:00:00Z"):
    """Articles from yearly categories whose first revision is after the Llama 3.1 knowledge cutoff."""
    import requests
    api = "https://en.wikipedia.org/w/api.php"
    hdr = {"User-Agent": "hpml-book-ch10/1.0 (research script)"}

    def get(params, tries=6):
        """Wikipedia API call; retry with backoff when the reply is not JSON (rate limiting, HTTP 429/503)."""
        import time
        for i in range(tries):
            r = requests.get(api, params=params, headers=hdr, timeout=30)
            try:
                return r.json()
            except ValueError:
                print("wikipedia API: status", r.status_code, "retry", i + 1, flush=True)
                time.sleep(2 ** i)
        r.raise_for_status(); return r.json()
    cats = ["Category:2025 in science", "Category:2025 in technology", "Category:2025 in spaceflight",
            "Category:2026 in science", "Category:2026 in technology", "Category:2025 software"]
    titles = []
    for c in cats:
        r = get(dict(action="query", list="categorymembers", cmtitle=c, cmlimit=200, cmtype="page", format="json"))
        titles += [m["title"] for m in r.get("query", {}).get("categorymembers", [])]
    out = []
    for t in titles:
        r = get(dict(action="query", prop="revisions", titles=t, rvlimit=1, rvdir="newer", rvprop="timestamp",
                     format="json"))
        page = next(iter(r["query"]["pages"].values()))
        ts = page.get("revisions", [{}])[0].get("timestamp", "")
        if ts <= after:
            continue
        r = get(dict(action="query", prop="extracts", explaintext=1, titles=t, format="json"))
        text = next(iter(r["query"]["pages"].values())).get("extract", "")
        if len(text) > 1500:
            out.append({"title": t, "created": ts, "text": text[:6000]})
        if len(out) >= n:
            break
    return out


def dump(rows, name):
    os.makedirs(DATA, exist_ok=True)
    with open(f"{DATA}/{name}", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-probe", action="store_true")
    a = ap.parse_args()
    tr, ev = hotpot()
    dump(tr, "train.jsonl"); dump(ev, "eval.jsonl"); dump(squad_abstain(), "abstain.jsonl")
    if not a.skip_probe:
        dump(probe_articles(), "probe_articles.jsonl")
