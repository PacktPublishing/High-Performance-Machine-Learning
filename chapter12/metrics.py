"""Retrieval and answer metrics."""
import math
import re


def ndcg_at_k(runs, qrels, k=10):
    tot = 0.0
    for q, ranked in runs.items():
        rel = qrels.get(q, {})
        dcg = sum(rel.get(p, 0) / math.log2(i + 2) for i, p in enumerate(ranked[:k]))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        tot += dcg / idcg if idcg else 0.0
    return 100 * tot / max(len(runs), 1)


def recall_at_k(runs, qrels, k=5):
    tot = 0.0
    for q, ranked in runs.items():
        rel = {p for p, g in qrels.get(q, {}).items() if g > 0}
        tot += len(rel & set(ranked[:k])) / len(rel) if rel else 0.0
    return 100 * tot / max(len(runs), 1)


def normalize(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9.%\- ]", " ", s)
    return " ".join(w for w in s.split() if w not in {"a", "an", "the"})


def numbers(s):
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", s or "")]


def exact_or_numeric(pred, gold, rel_tol=0.01):
    """True / False when decidable, None when the judge must decide."""
    if normalize(pred) == normalize(gold) or (normalize(gold) and normalize(gold) in normalize(pred)):
        return True
    g = numbers(gold)
    if g:
        p = numbers(pred)
        return bool(p) and all(any(abs(x - y) <= rel_tol * max(abs(y), 1e-9) for x in p) for y in g)
    return None
