"""Chapter 11 case study, NVIDIA GPU rows on Colab (A100, G4, L4, T4). Version 3 adds top-1 agreement
computed in-process (no 0.7 GB logits file), vLLM quantized rows (CH11_QUANT=fp8, or CH11_VLLM_MODEL for a
pre-quantized NVFP4 checkpoint) and vLLM agreement through prompt_logprobs. Version 2: CH11_REPEATS runs per
measurement (median, min, max), and prefill_ms = one forward pass over the padded 512-token prompt, the same
quantity the TPU script reports as ttft_ms.

  python gpu_reference.py torch   # PyTorch eager and torch.compile (static cache), plus parity logits
  python gpu_reference.py vllm    # vLLM (install vllm==0.33.0 first; run in its own process)
Model from CH11_MODEL (default google/gemma-2b-it), tag from CH11_TAG. Uses the same 32 RAG prompts and the
same FP32 CPU reference (ref/) as the TPU rows. Writes case_study_gpu_<gpu>_<tag>.json and out/<gpu>_<tag>_<mode>.npy.
"""
import json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODEL, MAX_PROMPT, NEW_TOKENS, PARITY_POSITIONS, load_prompts, chat, environment

TAG = os.environ.get("CH11_TAG", "gemma2b")
REPEATS = int(os.environ.get("CH11_REPEATS", "3"))


def stats(xs):
    xs = sorted(xs)
    return {"median": round(xs[len(xs) // 2], 3), "min": round(xs[0], 3), "max": round(xs[-1], 3), "n": len(xs)}


def gpu_name():
    n = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
    n = n.strip().split("\n")[0]
    for k in ("A100", "H100", "L4", "T4", "RTX PRO 6000", "B200", "L40S"):
        if k in n:
            return k.replace("RTX PRO 6000", "rtxpro6000").lower()
    return n.replace(" ", "_").lower()


def merge(path, key, value):
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[key] = value
    json.dump(data, open(path, "w"), indent=1)


def torch_rows():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    gpu = gpu_name()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16   # T4 has no BF16
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    prompts = [chat(tok, p) for p in load_prompts()]
    ref = None
    if TAG == "gemma2b" and os.path.exists("ref/ref_ids.json"):
        ref = json.load(open("ref/ref_ids.json")); ref["tokens"] = np.load("ref/ref_tokens.npy").tolist()

    def timed(fn):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
        return time.perf_counter() - t

    out_path = f"case_study_gpu_{gpu}_{TAG}_r{REPEATS}.json"
    for mode in ("eager", "compile"):
        key = f"{gpu}_{TAG}_{mode}"
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).cuda().eval()
            kw = dict(do_sample=False, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS, pad_token_id=tok.pad_token_id)
            if mode == "compile":
                model.generation_config.cache_implementation = "static"
                model.forward = torch.compile(model.forward)

            def batch(n):
                return tok(prompts[:n], return_tensors="pt", padding="max_length", truncation=True,
                           max_length=MAX_PROMPT).to("cuda")
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                b1 = batch(1)
                cold = timed(lambda: model.generate(**b1, **kw))
                warm = timed(lambda: model.generate(**b1, **kw))
                fwd = timed(lambda: model(input_ids=b1["input_ids"], attention_mask=b1["attention_mask"]))   # warm
                prefill = [timed(lambda: model(input_ids=b1["input_ids"], attention_mask=b1["attention_mask"])) * 1e3
                           for _ in range(REPEATS)]
                res = {}
                for n in (1, 16):
                    b = batch(n)
                    model.generate(**b, **kw)                       # warm this batch shape
                    ttft, tpot = [], []
                    for _ in range(REPEATS):
                        t1 = timed(lambda: model.generate(**b, do_sample=False, max_new_tokens=1, pad_token_id=tok.pad_token_id))
                        tn = timed(lambda: model.generate(**b, **kw))
                        ttft.append(t1 * 1e3); tpot.append((tn - t1) / (NEW_TOKENS - 1) * 1e3)
                    res[n] = (stats(ttft), stats(tpot))
            row = {"dtype": str(dtype).split(".")[-1], "compile_s": round(cold - warm, 2) if mode == "compile" else None,
                   "prefill_ms": stats(prefill), "ttft_generate_ms": res[1][0], "tpot_b1_ms": res[1][1],
                   "tpot_b16_ms": res[16][1],
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   "env": environment({"mode": mode, "gpu": gpu})}
            if ref:
                hit = tot = 0
                for ids, toks in zip(ref["prompt_ids"], ref["tokens"]):
                    full = torch.tensor([ids + toks[:PARITY_POSITIONS]], device="cuda")
                    with torch.no_grad():
                        lg = model(full).logits[0, len(ids) - 1: len(ids) - 1 + PARITY_POSITIONS]
                    pred = lg.argmax(-1).cpu().tolist()
                    hit += sum(int(a == b) for a, b in zip(pred, toks[:PARITY_POSITIONS])); tot += PARITY_POSITIONS
                row["top1_vs_fp32"] = round(hit / tot, 4)
            del model; torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            row = {"error": repr(e)[:500], "trace": traceback.format_exc()[-1500:], "env": environment({"mode": mode})}
        merge(out_path, key, row)
        print(key, json.dumps({k: (v["median"] if isinstance(v, dict) and "median" in v else v) for k, v in row.items() if k not in ("env", "trace")}))


def vllm_row():
    from vllm import LLM, SamplingParams
    gpu = gpu_name()
    quant = os.environ.get("CH11_QUANT", "")
    vmodel = os.environ.get("CH11_VLLM_MODEL", MODEL)
    label = os.environ.get("CH11_VLABEL", quant or ("bf16" if vmodel == MODEL else "prequant"))
    key = f"{gpu}_{TAG}_vllm_{label}"
    try:
        kw = dict(max_model_len=2048, enable_prefix_caching=False, gpu_memory_utilization=0.85)
        if quant:
            kw["quantization"] = quant
        llm = LLM(vmodel, **kw)
        tok = llm.get_tokenizer()
        prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                   for p in load_prompts()]
        one = SamplingParams(temperature=0, max_tokens=1)
        full = SamplingParams(temperature=0, max_tokens=NEW_TOKENS, min_tokens=NEW_TOKENS)
        llm.generate(prompts[:16], full, use_tqdm=False)          # warm-up and CUDA-graph capture
        res = {}
        for n in (1, 16):
            ttft, tpot = [], []
            for _ in range(REPEATS):
                t = time.perf_counter(); llm.generate(prompts[:n], one, use_tqdm=False); t1 = time.perf_counter() - t
                t = time.perf_counter(); llm.generate(prompts[:n], full, use_tqdm=False); tn = time.perf_counter() - t
                ttft.append(t1 * 1e3); tpot.append((tn - t1) / (NEW_TOKENS - 1) * 1e3)
            res[n] = (stats(ttft), stats(tpot))
        row = {"model": vmodel, "quant": label, "ttft_ms": res[1][0], "tpot_b1_ms": res[1][1],
               "tpot_b16_ms": res[16][1], "repeats": REPEATS}
        # Agreement: teacher-forced top-1 against a reference continuation (FP32 CPU for Gemma; vLLM BF16 otherwise)
        greedy = SamplingParams(temperature=0, max_tokens=PARITY_POSITIONS)
        if TAG == "gemma2b" and os.path.exists("ref/ref_ids.json"):
            r = json.load(open("ref/ref_ids.json")); pairs = list(zip(r["prompt_ids"], np.load("ref/ref_tokens.npy").tolist()))
            row["agreement_ref"] = "fp32_cpu"
        elif label != "bf16" and os.path.exists(f"ref_{TAG}_bf16.json"):
            pairs = json.load(open(f"ref_{TAG}_bf16.json")); row["agreement_ref"] = "vllm_bf16"
        else:
            pairs = None
        if label == "bf16":
            outs = llm.generate(prompts, greedy, use_tqdm=False)
            json.dump([[list(o.prompt_token_ids), list(o.outputs[0].token_ids)] for o in outs], open(f"ref_{TAG}_bf16.json", "w"))
        if pairs:
            from vllm.inputs import TokensPrompt
            pl = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
            reqs = [TokensPrompt(prompt_token_ids=list(ids) + list(toks[:PARITY_POSITIONS])) for ids, toks in pairs]
            hit = tot = 0
            for (ids, toks), o in zip(pairs, llm.generate(reqs, pl, use_tqdm=False)):
                for k in range(min(PARITY_POSITIONS, len(toks))):
                    j = len(ids) + k
                    lp = o.prompt_logprobs[j]
                    hit += int(lp[toks[k]].rank == 1); tot += 1
            row["top1_agreement"] = round(hit / tot, 4)
        row["env"] = environment({"gpu": gpu, "quant": label})
    except Exception as e:
        import traceback
        row = {"error": repr(e)[:500], "trace": traceback.format_exc()[-1500:], "env": environment()}
    merge(f"case_study_gpu_{gpu}_{TAG}_r{REPEATS}.json", key, row)
    print(key, json.dumps({k: (v["median"] if isinstance(v, dict) and "median" in v else v) for k, v in row.items() if k not in ("env", "trace")}))


if __name__ == "__main__":
    {"torch": torch_rows, "vllm": vllm_row}[sys.argv[1] if len(sys.argv) > 1 else "torch"]()
