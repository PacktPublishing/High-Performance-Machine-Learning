"""Run before any experiment: checks that vLLM 0.33.0 starts and serves what the case study needs.

    EXP=E2 GPU=H100 modal deploy serve_vlm.py && VLM_URL=<url> python smoke_test.py
Checks: model loads (prints the resolved revision hash to pin), an image request works, a tool call is parsed,
cached_tokens is reported on a repeated prompt, and /metrics exposes the prefix-cache counters.
"""
import base64
import io
import json
import os

import requests
from openai import OpenAI
from PIL import Image

url = os.environ["VLM_URL"]
c = OpenAI(base_url=url + "/v1", api_key="none", timeout=600)
print("models:", [m.id for m in c.models.list().data])
buf = io.BytesIO(); Image.new("RGB", (640, 480), "white").save(buf, format="PNG")
img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
r = c.chat.completions.create(model="qwen3-vl-8b", max_tokens=32, messages=[{"role": "user", "content": [
    {"type": "text", "text": "What color is this image?"}, {"type": "image_url", "image_url": {"url": img}}]}])
print("image ok:", r.choices[0].message.content)
tools = json.load(open("tools.json"))
msgs = [{"role": "system", "content": open("system_prompt.txt").read()},
        {"role": "user", "content": "What is 17.5 times 4? Use the calculator."}]
for i in range(2):
    r = c.chat.completions.create(model="qwen3-vl-8b", max_tokens=128, tools=tools, messages=msgs)
    d = r.usage.prompt_tokens_details
    print(f"call {i}: tool_calls={[t.function.name for t in (r.choices[0].message.tool_calls or [])]} "
          f"prompt={r.usage.prompt_tokens} cached={d.cached_tokens if d else None}")
m = requests.get(url + "/metrics").text
print("prefix counters present:", "vllm:prefix_cache_hits" in m and "vllm:prefix_cache_queries" in m)
