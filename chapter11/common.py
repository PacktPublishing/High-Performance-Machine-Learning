"""Shared helpers for the Chapter 11 case study.

Prompts: the 32 RAG prompts of up to 512 tokens from the Chapter 8 workload (JSONL with a "prompt" field).
If that file is not available, build_prompts() makes 32 RAG-style prompts from SQuAD validation contexts so the
scripts can run on their own. The Hugging Face token is read from the environment (HF_TOKEN), which Modal and
Colab provide from their secret stores; it is never written into code or results.
"""
import json, os, platform, subprocess, sys, time

MODEL = os.environ.get("CH11_MODEL", "google/gemma-2b-it")
N_PROMPTS, MAX_PROMPT, NEW_TOKENS, PARITY_POSITIONS = 32, 512, 128, 32


def load_prompts(path=None):
    if path and os.path.exists(path):
        rows = [json.loads(l) for l in open(path)]
        return [r["prompt"] for r in rows][:N_PROMPTS]
    return build_prompts()


def build_prompts():
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation").select(range(400))
    out, seen = [], set()
    for r in ds:
        if r["context"] in seen:
            continue
        seen.add(r["context"])
        out.append("Answer the question using only the context.\n\nContext:\n" + r["context"] +
                   "\n\nQuestion: " + r["question"] + "\nAnswer:")
        if len(out) == N_PROMPTS:
            break
    return out


def chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def environment(extra=None):
    """Step 1 of the debugging method: record versions that can change results or timings."""
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for mod in ("torch", "transformers", "vllm", "jax", "jaxlib", "onnxruntime", "onnx", "keras", "keras_hub"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            pass
    for cmd in (["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], ["rocm-smi", "--showproductname"]):
        try:
            env[cmd[0]] = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()[:300]
        except Exception:
            pass
    env.update({k: v for k, v in os.environ.items() if k.startswith(("TORCH_TPU_", "XLA_", "VLLM_", "JAX_", "CUBLAS_"))})
    env.update(extra or {})
    return env


def merge_json(path, key, value):
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[key] = value
    json.dump(data, open(path, "w"), indent=1)
