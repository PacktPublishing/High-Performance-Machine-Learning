"""The regression gate (Code 12.5). Exit status 1 fails the pipeline.

    python regress.py results/candidate.json results/baseline.json
Each file: {"pins": {...}, "changed": {...}, "runs": [{"task_p50_s":..,"task_p95_s":..,"cost_per_task":..,"accuracy":..}, x3]}
"""
import json
import statistics
import sys

LIMITS = {"task_p50_s": 1.10, "task_p95_s": 1.15,      # at most 10% / 15% slower
          "cost_per_task": 1.10}
ACC_DROP = 2.0                                          # accuracy points


def summarize(runs):                     # runs: results of 3 repetitions
    return {k: statistics.median(r[k] for r in runs)
            for k in (*LIMITS, "accuracy")}


def gate(candidate, baseline):
    assert candidate["pins"] == baseline["pins"] | candidate["changed"]
    c, b = summarize(candidate["runs"]), summarize(baseline["runs"])
    fails = [f"{k}: {c[k]:.3g} vs {b[k]:.3g}"
             for k, lim in LIMITS.items() if c[k] > lim * b[k]]
    if c["accuracy"] < b["accuracy"] - ACC_DROP:
        fails.append(f"accuracy: {c['accuracy']:.1f} vs {b['accuracy']:.1f}")
    return fails


if __name__ == "__main__":
    cand, base = (json.load(open(p)) for p in sys.argv[1:3])
    fails = gate(cand, base)
    print("\n".join(fails) or "PASS")
    sys.exit(1 if fails else 0)
