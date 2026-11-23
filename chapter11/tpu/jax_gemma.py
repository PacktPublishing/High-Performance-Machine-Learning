"""Chapter 11 case study, JAX row: Gemma-2B-IT with KerasHub on the JAX backend (Colab TPU).

  python jax_gemma.py speed                     # BF16: compile, prefill (512 padded, x5), TPOT via generate (x3)
  python jax_gemma.py logits <bf16|fp32> <default|highest>   # log-probs at the last 32 prompt positions, 8 prompts
  python jax_gemma.py compare                   # KL / top-1 of each logits file against fp32 + highest
Prompts are tokenized with the Hugging Face tokenizer (no TensorFlow needed), the same ids as the PyTorch rows.
"""
import json, os, sys, time
os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MAX_PROMPT, NEW_TOKENS, load_prompts, chat, environment

MODEL = "google/gemma-2b-it"
OUT = "jax_results.json"


def stats(xs):
    xs = sorted(xs)
    return {"median": round(xs[len(xs) // 2], 3), "min": round(xs[0], 3), "max": round(xs[-1], 3), "n": len(xs)}


def merge(key, val):
    d = json.load(open(OUT)) if os.path.exists(OUT) else {}
    d[key] = val
    json.dump(d, open(OUT, "w"), indent=1)
    print(key, json.dumps({k: (v.get("median") if isinstance(v, dict) and "median" in v else v)
                           for k, v in val.items() if k not in ("env", "trace")}), flush=True)


def load(dtype):
    import keras_hub
    return keras_hub.models.GemmaCausalLM.from_preset("hf://" + MODEL, preprocessor=None, dtype=dtype)


def prompt_ids():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    return [tok(chat(tok, p), add_special_tokens=False).input_ids[:MAX_PROMPT] for p in load_prompts()]


def padded(ids_list, length):
    x = np.zeros((len(ids_list), length), np.int32)
    m = np.zeros((len(ids_list), length), bool)
    for i, ids in enumerate(ids_list):
        x[i, :len(ids)] = ids
        m[i, :len(ids)] = True
    return x, m


def forward_fn(lm):
    import jax
    tv = [v.value for v in lm.trainable_variables]
    ntv = [v.value for v in lm.non_trainable_variables]

    @jax.jit
    def f(tv, ntv, x, m):
        logits, _ = lm.stateless_call(tv, ntv, {"token_ids": x, "padding_mask": m})
        return logits
    return lambda x, m: f(tv, ntv, x, m)


def speed():
    key = "v6e_jax_gemma2b_bf16"
    try:
        import jax
        lm = load("bfloat16")
        ids = prompt_ids()[0]
        n = len(ids)
        fwd = forward_fn(lm)
        x, m = padded([ids], MAX_PROMPT)
        t = time.perf_counter(); fwd(x, m).block_until_ready(); cold = time.perf_counter() - t
        pre = []
        for _ in range(5):
            t = time.perf_counter(); fwd(x, m).block_until_ready(); pre.append((time.perf_counter() - t) * 1e3)

        def gen(new):
            gx, gm = padded([ids], n + new)
            out = lm.generate({"token_ids": gx, "padding_mask": gm}, stop_token_ids=None)
            return np.asarray(out["token_ids"] if isinstance(out, dict) else out)
        t = time.perf_counter(); gen(NEW_TOKENS); gen_cold = time.perf_counter() - t
        gen(1)
        tpot = []
        for _ in range(3):
            t = time.perf_counter(); gen(1); t1 = time.perf_counter() - t
            t = time.perf_counter(); out = gen(NEW_TOKENS); tn = time.perf_counter() - t
            tpot.append((tn - t1) / (NEW_TOKENS - 1) * 1e3)
        row = {"compile_s": round(cold - pre[0] / 1e3, 1), "generate_compile_s": round(gen_cold, 1),
               "prefill_ms": stats(pre), "tpot_b1_ms": stats(tpot), "prompt_tokens": n,
               "env": environment({"jax": jax.__version__, "devices": str(jax.devices())})}
    except Exception as e:
        import traceback
        row = {"error": repr(e)[:500], "trace": traceback.format_exc()[-2000:], "env": environment()}
    merge(key, row)


def logits(dtype, prec):
    key = f"v6e_jax_gemma2b_{dtype}_{prec}_logits"
    try:
        import jax
        lm = load({"bf16": "bfloat16", "fp32": "float32"}[dtype])
        ids = prompt_ids()[:8]
        x, m = padded(ids, MAX_PROMPT)
        with jax.default_matmul_precision(prec):
            fwd = forward_fn(lm)
            lg = np.asarray(fwd(x, m).astype("float32"))
        rows = []
        for i, seq in enumerate(ids):
            z = lg[i, len(seq) - 32: len(seq)].astype(np.float64)
            z = z - z.max(-1, keepdims=True)
            rows.append((z - np.log(np.exp(z).sum(-1, keepdims=True))).astype(np.float32))
        np.save(f"lp_{dtype}_{prec}.npy", np.stack(rows))
        row = {"saved": f"lp_{dtype}_{prec}.npy", "shape": list(np.stack(rows).shape)}
    except Exception as e:
        import traceback
        row = {"error": repr(e)[:500], "trace": traceback.format_exc()[-2000:]}
    merge(key, row)


def compare():
    ref_path = "lp_fp32_highest.npy"
    if not os.path.exists(ref_path):
        merge("v6e_jax_compare", {"error": "no fp32 highest reference"}); return
    ref = np.load(ref_path).astype(np.float64)
    res = {}
    for name in ("fp32_default", "bf16_default"):
        p = f"lp_{name}.npy"
        if not os.path.exists(p):
            continue
        q = np.load(p).astype(np.float64)
        kl = (np.exp(ref) * (ref - q)).sum(-1)
        res[name] = {"kl_mean": float(kl.mean()), "kl_p99": float(np.percentile(kl, 99)),
                     "top1_agreement": float((ref.argmax(-1) == q.argmax(-1)).mean()),
                     "identical": bool(np.array_equal(ref, q))}
    merge("v6e_jax_compare_vs_fp32_highest", res)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"speed": lambda: speed(), "logits": lambda: logits(a[1], a[2]), "compare": lambda: compare()}[a[0]]()
