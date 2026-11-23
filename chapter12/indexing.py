"""Build the two indexes of the case study and serve them as a retrieval service on Modal.

  text index : BGE-M3 dense embeddings of each page's extracted text, FAISS inner product (as in Chapter 2)
  page index : tomoro-colqwen3-embed-4b multi-vector embeddings of page images, scored by MaxSim on the GPU

    modal run indexing.py::build --data data/           # writes the indexes to the Modal volume hpml-ch12-index
    modal deploy indexing.py                             # serves search_text / search_pages
    modal run indexing.py::eval_retrieval                # R0: nDCG@10 and recall@5 for text, page, hybrid

The ColQwen3 loading code follows the model card's usage section; check it against the current card on first use.
"""
import json
import os

import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.14.*", "transformers>=5.16", "FlagEmbedding", "faiss-cpu", "pillow", "numpy",
                      "accelerate", "autoawq")
         .add_local_python_source("metrics"))
vol = modal.Volume.from_name("hpml-ch12-index", create_if_missing=True)
hf_cache = modal.Volume.from_name("hpml-hf-cache", create_if_missing=True)
app = modal.App("hpml-ch12-retrieval")
PAGE_MODEL = os.environ.get("PAGE_MODEL", "TomoroAI/tomoro-colqwen3-embed-4b")


def load_page_model(name=PAGE_MODEL):
    import torch
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True, max_num_visual_tokens=1280)   # model card
    return model, proc


@app.function(image=image, gpu="H100", timeout=4 * 3600, volumes={"/index": vol, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name("huggingface")])
def build(corpus_rows: list, images: dict):
    import faiss
    import numpy as np
    import torch
    from FlagEmbedding import BGEM3FlagModel
    from PIL import Image
    import io
    bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    texts = [r["text"] or " " for r in corpus_rows]
    dense = bge.encode(texts, batch_size=64, max_length=1024)["dense_vecs"].astype("float32")
    idx = faiss.IndexFlatIP(dense.shape[1]); idx.add(dense)
    faiss.write_index(idx, "/index/text.faiss")
    model, proc = load_page_model()
    vecs, offsets = [], [0]
    for i in range(0, len(corpus_rows), 16):
        batch = [Image.open(io.BytesIO(images[r["page_id"]])).convert("RGB") for r in corpus_rows[i:i + 16]]
        with torch.no_grad():
            inp = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in proc.process_images(images=batch).items()}
            emb = model(**inp).embeddings                        # [B, patches, 320]
        for e, m in zip(emb, inp["attention_mask"]):
            e = e[m.bool()].half().cpu()
            vecs.append(e); offsets.append(offsets[-1] + e.shape[0])
    torch.save({"vecs": torch.cat(vecs), "offsets": torch.tensor(offsets)}, "/index/pages.pt")
    json.dump([r["page_id"] for r in corpus_rows], open("/index/page_ids.json", "w"))
    vol.commit()
    return {"pages": len(corpus_rows), "patch_vectors": offsets[-1]}


@app.cls(image=image, gpu=os.environ.get("RETRIEVER_GPU", "L4"), volumes={"/index": vol, "/root/.cache/huggingface": hf_cache},
         secrets=[modal.Secret.from_name("huggingface")], scaledown_window=600)
@modal.concurrent(max_inputs=64)
class Retrieval:
    @modal.enter()
    def load(self):
        import faiss
        import torch
        from FlagEmbedding import BGEM3FlagModel
        self.bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        self.text = faiss.read_index("/index/text.faiss")
        self.ids = json.load(open("/index/page_ids.json"))
        p = torch.load("/index/pages.pt")
        self.pvecs, self.off = p["vecs"].cuda(), p["offsets"].tolist()
        self.page_owner = torch.repeat_interleave(torch.arange(len(self.ids)),
                                                  torch.tensor([b - a for a, b in zip(self.off, self.off[1:])])).cuda()
        self.model, self.proc = load_page_model()
        if os.environ.get("COMPILE") == "1":
            self.model = torch.compile(self.model, dynamic=True)  # E7; dynamic: query lengths vary

    @modal.method()
    def search_text(self, q: str, k: int = 5):
        v = self.bge.encode([q], max_length=256)["dense_vecs"].astype("float32")
        s, i = self.text.search(v, k)
        return [{"page_id": self.ids[j], "score": float(x)} for j, x in zip(i[0], s[0])]

    @modal.method()
    def search_pages(self, q: str, k: int = 5):
        import torch
        with torch.no_grad():
            qi = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in self.proc.process_texts(texts=[q]).items()}
            qv = self.model(**qi).embeddings[0].half()                                 # [q_tokens, 320]
            sim = qv @ self.pvecs.T                                                   # [q_tokens, all_patches]
            # MaxSim: for each query token, the best patch of each page; then sum over query tokens
            best = torch.full((qv.shape[0], len(self.ids)), -1e4, device="cuda", dtype=sim.dtype)
            best = best.scatter_reduce(1, self.page_owner.expand(qv.shape[0], -1), sim, reduce="amax")
            score = best.float().sum(0)
            top = torch.topk(score, k)
        return [{"page_id": self.ids[j], "score": float(s)} for s, j in zip(top.values.tolist(), top.indices.tolist())]

    @modal.method()
    def search_hybrid(self, q: str, k: int = 5, rrf_k: int = 60):
        """Reciprocal rank fusion of the two retrievers."""
        fused = {}
        for res in (self.search_text.local(q, 50), self.search_pages.local(q, 50)):
            for r, hit in enumerate(res):
                fused[hit["page_id"]] = fused.get(hit["page_id"], 0.0) + 1.0 / (rrf_k + r + 1)
        return [{"page_id": p, "score": s} for p, s in sorted(fused.items(), key=lambda x: -x[1])[:k]]


@app.local_entrypoint()
def eval_retrieval(data: str = "data", qrels: str = "data/qrels.json"):
    """R0: nDCG@10 and recall@5 per retriever (qrels from the ViDoRe V3 'qrels' config)."""
    from metrics import ndcg_at_k, recall_at_k
    rel = json.load(open(qrels))
    qs = [json.loads(l) for l in open(f"{data}/queries.jsonl")]
    r = Retrieval()
    out = {}
    for name in ("search_text", "search_pages", "search_hybrid"):
        runs = {q["query_id"]: [h["page_id"] for h in getattr(r, name).remote(q["query"], 10)] for q in qs}
        out[name] = {"ndcg10": ndcg_at_k(runs, rel, 10), "recall5": recall_at_k(runs, rel, 5)}
    print(json.dumps(out, indent=1))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/R0_retrieval.json", "w"), indent=1)
