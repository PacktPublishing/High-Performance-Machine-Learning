"""Single-request and batch-16 timing of Gemma-2B-IT with vLLM on ROCm; same method as h100/modal_reference.py."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import MODEL, NEW_TOKENS, load_prompts, environment

ap = argparse.ArgumentParser(); ap.add_argument("--backend", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
from vllm import LLM, SamplingParams
llm = LLM(MODEL, dtype="bfloat16", max_model_len=2048, enable_prefix_caching=False,
          attention_backend=a.backend)   # if this argument is rejected, set VLLM_ATTENTION_BACKEND instead
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
           for p in load_prompts()]
one, full = SamplingParams(temperature=0, max_tokens=1), SamplingParams(temperature=0, max_tokens=NEW_TOKENS,
                                                                         min_tokens=NEW_TOKENS)
llm.generate(prompts[:2], full)
res = {}
for n in (1, 16):
    t = time.perf_counter(); llm.generate(prompts[:n], one); t1 = time.perf_counter() - t
    t = time.perf_counter(); llm.generate(prompts[:n], full); tn = time.perf_counter() - t
    res[n] = (t1, (tn - t1) / (NEW_TOKENS - 1))
json.dump({"ttft_ms": round(res[1][0] * 1e3, 2), "tpot_b1_ms": round(res[1][1] * 1e3, 3),
           "tpot_b16_ms": round(res[16][1] * 1e3, 3), "fallback_ops": 0, "backend": a.backend,
           "env": environment()}, open(a.out, "w"), indent=1)
