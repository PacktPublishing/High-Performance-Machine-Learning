"""Thin client for the retrieval service (Modal, or HTTP when RETRIEVAL_URL is set), and the local page-image map."""
import json
import os


class RemoteRetrieval:
    def __init__(self, app="hpml-ch12-retrieval"):
        self.url = os.environ.get("RETRIEVAL_URL")          # HTTP service, e.g. inside a cluster job
        if self.url:
            import requests
            self.http = requests.Session()
        else:
            import modal
            self.r = modal.Cls.from_name(app, "Retrieval")()

    def _call(self, name, q, k):
        if self.url:
            resp = self.http.post(f"{self.url}/{name}", json={"q": q, "k": k}, timeout=120)
            resp.raise_for_status()
            return resp.json()
        return getattr(self.r, name).remote(q, k)

    def search_text(self, q, k=5):
        return self._call("search_text", q, k)

    def search_pages(self, q, k=5):
        return self._call("search_pages", q, k)

    def search_hybrid(self, q, k=5):
        return self._call("search_hybrid", q, k)


def load_pages(data="data"):
    return {r["page_id"]: r["path"] for r in map(json.loads, open(f"{data}/corpus.jsonl"))}
