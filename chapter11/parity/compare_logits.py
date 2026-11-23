"""Model-level parity check (Code 11.5): compare saved logits from a candidate platform with the FP32 reference.

Each .npy file holds float32 logits of shape [prompts, positions, vocab] for the same prompts and the same
reference token sequence (teacher-forced on the reference's greedy tokens, so positions line up).
Usage: python compare_logits.py ref_cpu_fp32.npy cand.npy [--greedy ref_tokens.npy cand_tokens.npy] --key NAME
"""
import argparse, json, os, sys
import numpy as np


def log_softmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def compare(ref, cand):
    lp, lq = log_softmax(ref.astype(np.float64)), log_softmax(cand.astype(np.float64))
    kl = (np.exp(lp) * (lp - lq)).sum(-1)
    return {"kl_mean": float(kl.mean()), "kl_p99": float(np.quantile(kl, 0.99)),
            "top1_agreement": float((ref.argmax(-1) == cand.argmax(-1)).mean()),
            "max_rel_logit_err": float(np.abs(cand - ref).max() / np.abs(ref).max())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("cand"); ap.add_argument("--key", required=True)
    ap.add_argument("--greedy", nargs=2, help="token id arrays [prompts, 128] from free-running greedy generation")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "results", "parity.json"))
    a = ap.parse_args()
    r = compare(np.load(a.ref), np.load(a.cand))
    if a.greedy:
        g0, g1 = np.load(a.greedy[0]), np.load(a.greedy[1])
        # identical prefix length per prompt, averaged
        same = (g0 == g1)
        prefix = [int(np.argmin(s)) if not s.all() else s.size for s in same]
        r["greedy_identical_tokens"] = float(np.mean(prefix))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from common import merge_json
    merge_json(a.out, a.key, r)
    print(json.dumps(r, indent=1))
