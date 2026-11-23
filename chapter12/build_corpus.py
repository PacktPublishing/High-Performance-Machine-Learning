"""Download the ViDoRe V3 English corpora used in Chapter 12, render page images, and write a local corpus.

Corpora: Computer Science, Finance-EN, HR (dataset cards: CC BY 4.0; per-document licenses in
documents_metadata["license"]). The data is downloaded at run time and is never committed to the repository.

    python build_corpus.py --out data/        # writes pages/*.png, corpus.jsonl, queries.jsonl, license_report.json

The exact split and column names of the ViDoRe V3 datasets must be checked on first use
(the script prints them); adjust COLUMNS below if they differ.
"""
import argparse
import json
import os

from datasets import get_dataset_config_names, load_dataset

DATASETS = {"cs": "vidore/vidore_v3_computer_science", "finance_en": "vidore/vidore_v3_finance_en",
            "hr": "vidore/vidore_v3_hr"}
# Dataset revisions resolved on 2026-11-20 (Step 1 inspection); --revision overrides all three.
REVISIONS = {"cs": "d5cc75883d92e294f0c0fc2662551c9708a06ebc", "finance_en": "7f432c176d82e27546501ad8064a713ac3071809",
             "hr": "0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3"}
COLUMNS = {"page_id": "corpus_id", "image": "image", "text": "markdown",
           "query_id": "query_id", "query": "query", "answer": "answer", "query_type": "query_types"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--revision", default=None, help="pin the dataset revision for reproducibility")
    ap.add_argument("--max-side", type=int, default=2048, help="longest side of the stored page image in pixels")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/pages", exist_ok=True)
    corpus, queries, licenses, qrels = [], [], {}, {}
    for short, name in DATASETS.items():
        rev = a.revision or REVISIONS[short]
        configs = get_dataset_config_names(name, revision=rev)
        print(name, "configs:", configs)
        docs = load_dataset(name, "corpus", split="test", revision=rev)
        qs = load_dataset(name, "queries", split="test", revision=rev)
        print(" corpus columns:", docs.column_names, " query columns:", qs.column_names)
        for row in docs:
            pid = f"{short}:{row[COLUMNS['page_id']]}"
            img = row[COLUMNS["image"]]
            img.thumbnail((a.max_side, a.max_side))
            path = f"{a.out}/pages/{pid.replace(':', '_')}.png"
            img.save(path)
            corpus.append({"page_id": pid, "path": path, "text": row.get(COLUMNS["text"]) or "", "corpus": short})
        for row in qs:
            if row.get("language", "english") not in ("english", "en"):
                continue                                  # original English queries only, no translations
            queries.append({"query_id": f"{short}:{row[COLUMNS['query_id']]}", "query": row[COLUMNS["query"]],
                            "answer": row.get(COLUMNS["answer"]), "type": row.get(COLUMNS["query_type"]),
                            "corpus": short})
        try:
            meta = load_dataset(name, "documents_metadata", split="test", revision=rev)
            licenses[short] = sorted(set(meta["license"]))
        except Exception as e:                            # report, do not guess
            licenses[short] = f"could not read documents_metadata: {e}"
        for row in load_dataset(name, "qrels", split="test", revision=rev):     # graded relevance for R0
            qrels.setdefault(f"{short}:{row['query_id']}", {})[f"{short}:{row['corpus_id']}"] = int(row["score"])
    json.dump(qrels, open(f"{a.out}/qrels.json", "w"))
    with open(f"{a.out}/corpus.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in corpus)
    with open(f"{a.out}/queries.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in queries)
    json.dump(licenses, open(f"{a.out}/license_report.json", "w"), indent=1)
    print(len(corpus), "pages,", len(queries), "queries; licenses:", licenses)


if __name__ == "__main__":
    main()
