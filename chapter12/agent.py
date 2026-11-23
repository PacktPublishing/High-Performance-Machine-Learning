"""Agent controller for the Chapter 12 case study.

One task = one question. The controller calls the VLM (OpenAI-compatible vLLM endpoint) in a loop of at most
`max_calls` steps; the model chooses tools. Every model call, tool call and task is written as a span with
OpenTelemetry GenAI names (invoke_agent, chat, execute_tool) to a JSON-lines trace file.

    VLM_URL=https://...modal.run/v1 python agent.py --exp E2 --question "..."     # one task, prints the trace
"""
import argparse
import ast
import base64
import io
import json
import operator
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

SYSTEM_PROMPT = open(__file__.replace("agent.py", "system_prompt.txt")).read()
TOOLS = json.load(open(__file__.replace("agent.py", "tools.json")))    # fixed order


class Tracer:
    def __init__(self, path):
        self.f = open(path, "a")

    def span(self, name, trace_id, parent, start, end, **attrs):
        sid = uuid.uuid4().hex[:16]
        self.f.write(json.dumps({"name": name, "trace_id": trace_id, "span_id": sid, "parent": parent,
                                 "start": start, "end": end, "attributes": attrs}) + "\n")
        self.f.flush()
        return sid


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg}


def calculate(expr: str):
    """Safe arithmetic: numbers and + - * / ** only."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)


class Agent:
    def __init__(self, cfg, retrieval, pages, tracer, base_url, client=None):
        self.cfg, self.retrieval, self.pages, self.tracer = cfg, retrieval, pages, tracer
        self.client = client or OpenAI(base_url=base_url, api_key="none", timeout=300)
        self.model = cfg["served_name"]

    def system(self):
        if self.cfg["agent"]["stable_prompt"]:
            return SYSTEM_PROMPT
        # E1: the naive layout that breaks prefix reuse
        return f"Current time: {time.time():.3f}\nRequest: {uuid.uuid4()}\n" + SYSTEM_PROMPT

    def tools(self):
        if self.cfg["agent"]["stable_prompt"]:
            return TOOLS
        return random.sample(TOOLS, len(TOOLS))

    def page_image(self, page_id, crop=None):
        from PIL import Image
        img = Image.open(self.pages[page_id]).convert("RGB")
        if crop:
            w, h = img.size
            x0, y0, x1, y1 = [float(v) for v in crop]  # fractions of width and height
            if max(x0, y0, x1, y1) > 1:                # the model sometimes passes pixels: convert, then clamp
                x0, x1, y0, y1 = x0 / w, x1 / w, y0 / h, y1 / h
            x0, x1 = sorted(min(max(v, 0.0), 1.0) for v in (x0, x1))
            y0, y1 = sorted(min(max(v, 0.0), 1.0) for v in (y0, y1))
            if x1 - x0 > 0.01 and y1 - y0 > 0.01:     # ignore an empty box
                img = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        mp = self.cfg["agent"].get("image_max_pixels")
        if mp and img.size[0] * img.size[1] > mp:
            s = (mp / (img.size[0] * img.size[1])) ** 0.5
            img = img.resize((int(img.size[0] * s), int(img.size[1] * s)))
        buf = io.BytesIO(); img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def run_tool(self, name, args):
        if name == "search_text":
            return self.retrieval.search_text(args["query"], args.get("k", 5)), None
        if name == "search_pages":
            return self.retrieval.search_pages(args["query"], args.get("k", 5)), None
        if name == "view_page":
            return {"page_id": args["page_id"]}, self.page_image(args["page_id"], args.get("crop"))
        if name == "calculate":
            try:
                return {"result": calculate(args["expression"])}, None
            except Exception as e:
                return {"error": str(e)}, None
        raise ValueError(name)

    def run(self, question, task_id=None):
        tid = task_id or str(uuid.uuid4())
        t_task = time.time()
        history = [{"role": "user", "content": question}]
        stats = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "images": 0}
        answer, cited = None, []
        for step in range(self.cfg["agent"]["max_calls"]):
            t0 = time.time()
            # require_tool_first: true = the first call must use some tool; a tool name = the first call is that tool
            # (the model writes the arguments). Same tools and prompt, so prefix reuse is unchanged.
            first = self.cfg["agent"].get("require_tool_first") if step == 0 else None
            kw = ({"tool_choice": "required"} if first is True else
                  {"tool_choice": {"type": "function", "function": {"name": first}}} if first else {})
            resp = self.client.chat.completions.create(
                model=self.model, max_tokens=512, temperature=0.0, tools=self.tools(),
                messages=[{"role": "system", "content": self.system()}] + history, stream=False, **kw)
            t1 = time.time()
            u = resp.usage
            cached = (u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0) or 0
            stats["calls"] += 1; stats["prompt_tokens"] += u.prompt_tokens
            stats["cached_tokens"] += cached; stats["output_tokens"] += u.completion_tokens
            self.tracer.span("chat", tid, None, t0, t1, **{"gen_ai.request.model": self.model,
                             "gen_ai.usage.input_tokens": u.prompt_tokens, "gen_ai.usage.output_tokens": u.completion_tokens,
                             "hpml.cached_tokens": cached, "hpml.step": step})
            msg = resp.choices[0].message
            history.append(msg.model_dump(exclude_none=True))
            calls = msg.tool_calls or []
            if not calls:
                answer = msg.content
                break
            done = [c for c in calls if c.function.name == "answer"]
            if done:
                a = json.loads(done[0].function.arguments)
                answer, cited = a.get("text"), a.get("pages", [])
                break
            run = lambda c: (c, *self.run_tool(c.function.name, json.loads(c.function.arguments)), time.time())
            ts = time.time()
            if self.cfg["agent"]["parallel_tools"] and len(calls) > 1:
                with ThreadPoolExecutor(len(calls)) as ex:
                    results = list(ex.map(run, calls))       # map keeps the call order: deterministic history
            else:
                results = [run(c) for c in calls]
            for c, result, img, te in results:
                self.tracer.span("execute_tool", tid, None, ts, te, **{"gen_ai.tool.name": c.function.name})
                history.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result)})
                if img:
                    stats["images"] += 1
                    history.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": img}}]})
        t_end = time.time()
        self.tracer.span("invoke_agent", tid, None, t_task, t_end, **{"hpml.task_time_s": t_end - t_task, **stats})
        return {"task_id": tid, "answer": answer, "cited": cited, "task_time_s": t_end - t_task, **stats}


def single_pass(agent, question, k=3):
    """E0: hybrid retrieval of the top-k pages, then one model call with the page images."""
    t0 = time.time()
    hits = agent.retrieval.search_hybrid(question, k)
    content = [{"type": "text", "text": question}] + [
        {"type": "image_url", "image_url": {"url": agent.page_image(h["page_id"])}} for h in hits]
    resp = agent.client.chat.completions.create(model=agent.model, max_tokens=512, temperature=0.0,
                                                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                                                          {"role": "user", "content": content}])
    u = resp.usage
    return {"answer": resp.choices[0].message.content, "cited": [h["page_id"] for h in hits],
            "task_time_s": time.time() - t0, "calls": 1, "prompt_tokens": u.prompt_tokens,
            "cached_tokens": (u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else 0) or 0,
            "output_tokens": u.completion_tokens, "images": len(hits)}


if __name__ == "__main__":
    import os
    from serve_vlm import resolve
    from retrieval_client import RemoteRetrieval, load_pages
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="E2"); ap.add_argument("--question", required=True)
    a = ap.parse_args()
    cfg = resolve(a.exp, "configs/experiments.yaml")
    ag = Agent(cfg, RemoteRetrieval(), load_pages("data"), Tracer("results/trace_debug.jsonl"), os.environ["VLM_URL"])
    print(json.dumps(single_pass(ag, a.question) if cfg.get("mode") == "single_pass" else ag.run(a.question), indent=1))
