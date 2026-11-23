"""Wrap the OpenAI client so that each model call of a task goes through the Router (experiment E8)."""
from openai import OpenAI
from transformers import AutoTokenizer

from agent import Agent
from router import Router

_router = None


class RoutedCompletions:
    def __init__(self, task_id):
        self.task_id = task_id

    def create(self, **kw):
        text = "".join(m["content"] if isinstance(m.get("content"), str) else "" for m in kw["messages"])
        rep = _router.pick(self.task_id, text)
        try:
            return OpenAI(base_url=rep.url + "/v1", api_key="none", timeout=300).chat.completions.create(**kw)
        finally:
            _router.done(rep)


class RoutedClient:
    def __init__(self, task_id):
        self.chat = type("C", (), {"completions": RoutedCompletions(task_id)})()


def routed_agent_factory(cfg, retrieval, pages, tracer, urls, policy):
    global _router
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    _router = Router(urls, policy or "prefix", tokenizer=tok)
    import uuid

    def make():
        return Agent(cfg, retrieval, pages, tracer, base_url=None, client=RoutedClient(str(uuid.uuid4())))
    return make
