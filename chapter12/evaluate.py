"""Accuracy: exact or numeric match first, then an LLM judge with both answer orders.

    JUDGE_URL=... python evaluate.py results/E2_H100_single_0.5_0.jsonl
The judge (Qwen/Qwen3-32B-FP8 at a pinned revision) is served with serve_vlm.py-style vLLM on an H100.
Report judge agreement on the human-labelled subset (data/human_labels.jsonl) next to every accuracy figure.
"""
import json
import os
import sys

from openai import OpenAI

from metrics import exact_or_numeric

JUDGE_PROMPT = ("Question: {q}\nReference answer: {gold}\nAnswer {a_name}: {a}\nAnswer {b_name}: {b}\n"
                "Is Answer {target} correct with respect to the reference answer? Reply with only YES or NO.")


def judge(client, model, q, gold, pred):
    votes = []
    for first, second, target in ((pred, gold, "1"), (gold, pred, "2")):     # both positions
        msg = JUDGE_PROMPT.format(q=q, gold=gold, a_name="1", a=first, b_name="2", b=second, target=target)
        # Qwen3 judges start in thinking mode ("<think>..."), which a 3-token reply can never get past: switch it off.
        # Other chat templates ignore the extra variable.
        r = client.chat.completions.create(model=model, temperature=0.0, max_tokens=3,
                                           messages=[{"role": "user", "content": msg}],
                                           extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        votes.append(r.choices[0].message.content.strip().upper().startswith("YES"))
    return all(votes)                                                        # correct only if both orders agree


def score(rows, queries, client=None, model=None, verdicts=None):
    """verdicts: optional list; filled with the per-row decision (for <file>.judged.json)."""
    q = {x["query_id"]: x for x in queries}
    out, judged = [], 0
    for r in rows:
        if not r.get("ok"):
            out.append(False); continue
        gold = q[r["query_id"]]["answer"]
        v = exact_or_numeric(r["answer"], gold)
        if v is None and client is not None:
            v = judge(client, model, q[r["query_id"]]["query"], gold, r["answer"]); judged += 1
        out.append(bool(v))
    if verdicts is not None:
        verdicts.extend(out)
    return 100 * sum(out) / max(len(out), 1), judged


if __name__ == "__main__":
    queries = [json.loads(l) for l in open("data/queries.jsonl")]
    client = OpenAI(base_url=os.environ["JUDGE_URL"] + "/v1", api_key="none") if os.environ.get("JUDGE_URL") else None
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path)]
        v = []
        acc, n = score(rows, queries, client, os.environ.get("JUDGE_MODEL", "judge"), verdicts=v)
        print(f"{path}: accuracy {acc:.1f}%  ({n} judged)")
        if client is not None:                         # report.py reads these verdicts
            json.dump({"accuracy": acc, "judged": n, "verdicts": v, "judge_model": os.environ.get("JUDGE_MODEL", "judge")},
                      open(path[:-len(".jsonl")] + ".judged.json", "w"))
