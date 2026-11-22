"""Modal application for the Chapter 10 case study.

Two images, because vLLM 0.33 and TRL 1.17 pin different Transformers versions:
  train_image  - PyTorch 2.14, Transformers >= 5.22, PEFT 0.21.2, TRL 1.17.1, bitsandbytes 0.50.2
  serve_image  - vLLM 0.33.0 (CUDA 13 wheels), lm-evaluation-harness
The Hugging Face token comes only from the Modal Secret "huggingface" (HF_TOKEN); it never appears in code or logs.

Examples:
  modal run chapter10/modal_app.py::smoke
  modal run chapter10/modal_app.py::prep
  modal run chapter10/modal_app.py::train --model llama-3.1-8b --method qlora --gpu H100
  modal run chapter10/modal_app.py::evaluate --model llama-3.1-8b --run llama-3.1-8b-qlora-r16-s0-H100 --task hotpot
  modal run chapter10/modal_app.py::serve_bench --model llama-3.1-8b --run ... --mode unmerged --adapters 4
"""
import os
import subprocess

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("hpml-ch10")
vol = modal.Volume.from_name("hpml-ch10", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
secret = modal.Secret.from_name("huggingface")
MOUNTS = {"/vol": vol, "/root/.cache/huggingface": hf_cache}

train_image = (modal.Image.debian_slim(python_version="3.11")
               .pip_install("torch==2.14.0", "transformers>=5.22,<6", "peft==0.21.2", "trl==1.17.1",
                            "bitsandbytes==0.50.2", "accelerate>=1.15", "datasets>=4", "safetensors",
                            "requests", "huggingface_hub")
               .env({"CH10_VOL": "/vol", "TOKENIZERS_PARALLELISM": "false"})
               .add_local_dir(HERE, "/root/chapter10"))
serve_image = (modal.Image.debian_slim(python_version="3.11")
               .pip_install("vllm==0.33.0", "lm-eval[api,ifeval]==0.4.13", "httpx", "datasets>=4", "requests")
               .env({"CH10_VOL": "/vol", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "False"})
               .add_local_dir(HERE, "/root/chapter10"))


def sh(cmd):
    print("+", cmd)
    subprocess.run(cmd, shell=True, check=True, cwd="/root/chapter10")
    vol.commit()


GPUS = {"H100": "H100", "A100": "A100-80GB", "A100-80GB": "A100-80GB", "T4": "T4", "H100:2": "H100:2"}


@app.function(image=serve_image, gpu="H100", volumes=MOUNTS, secrets=[secret], timeout=3600)
def smoke_gpu(gpu_name: str = "H100"):
    """vLLM 0.33 smoke test: each model loads, LoRA works on Llama and on Qwen3-VL's language model, FP8 KV cache."""
    import json
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    report = {"gpu": gpu_name}
    sp = SamplingParams(max_tokens=8, temperature=0)
    for key, mid, kw in (("llama-3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", {}),
                         ("gemma-2b", "google/gemma-2b-it", {}),
                         ("llama-3.2-1b", "meta-llama/Llama-3.2-1B-Instruct", {}),
                         ("qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct", {"limit_mm_per_prompt": {"image": 0}}),
                         ("gemma-4-e2b", "google/gemma-4-E2B-it", {})):
        if gpu_name == "T4" and key in ("llama-3.1-8b", "qwen3-vl-8b"):
            continue
        try:
            llm = LLM(model=mid, max_model_len=2048, enable_lora=True, max_lora_rank=16,
                      gpu_memory_utilization=0.85, **kw)
            out = llm.chat([[{"role": "user", "content": "Say OK."}]], sp)[0].outputs[0].text
            report[key] = {"load": True, "text": out}
            ad = f"/vol/smoke_adapters/{key}"
            if os.path.exists(ad):
                o = llm.chat([[{"role": "user", "content": "Say OK."}]], sp,
                             lora_request=LoRARequest("smoke", 1, ad))[0].outputs[0].text
                report[key]["lora"] = o
            del llm
        except Exception as e:  # report and continue
            report[key] = {"load": False, "error": repr(e)[:400]}
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    if gpu_name != "T4":
        try:
            llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=2048, kv_cache_dtype="fp8")
            report["fp8_kv"] = llm.chat([[{"role": "user", "content": "Say OK."}]], sp)[0].outputs[0].text
        except Exception as e:
            report["fp8_kv"] = "error: " + repr(e)[:300]
    os.makedirs("/vol/results", exist_ok=True)
    json.dump(report, open(f"/vol/results/smoke_{gpu_name}.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(report, indent=1))


@app.function(image=train_image, gpu="H100", volumes=MOUNTS, secrets=[secret], timeout=3600)
def smoke_adapters():
    """Tiny random rank-16 adapters (one training step) used by the smoke test to check vLLM LoRA loading."""
    sh("python smoke_adapters.py")


@app.function(image=train_image, volumes=MOUNTS, secrets=[secret], timeout=3 * 3600)
def prep_remote():
    sh("python prep_data.py")


@app.function(image=train_image, gpu="H100", volumes=MOUNTS, secrets=[secret], timeout=12 * 3600)
def train_remote(args: str, n_gpus: int = 1):
    if n_gpus > 1:
        sh(f"accelerate launch --num_processes {n_gpus} --use_fsdp --fsdp_sharding_strategy FULL_SHARD "
           f"--fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --mixed_precision bf16 train_peft.py {args} "
           f"--n-gpus {n_gpus}")
    else:
        sh(f"python train_peft.py {args}")


@app.function(image=train_image, gpu="H100", volumes=MOUNTS, secrets=[secret], timeout=3 * 3600)
def merge_remote(args: str):
    sh(f"python merge_adapter.py {args}")


@app.function(image=train_image, gpu="H100", volumes=MOUNTS, secrets=[secret], timeout=12 * 3600)
def distill_remote(args: str):
    sh(f"python distill.py {args}")


@app.function(image=serve_image, gpu="H100", volumes=MOUNTS, secrets=[secret], timeout=6 * 3600)
def serve_remote(script: str, args: str):
    sh(f"python {script} {args}")


@app.function(image=train_image, gpu="H100", volumes=MOUNTS, secrets=[secret], timeout=3600)
def spectrum_remote(args: str = ""):
    sh(f"python svd_spectrum.py {args}")


@app.function(image=train_image, volumes=MOUNTS, timeout=1800)
def collect_remote():
    sh("python collect_results.py")


@app.local_entrypoint()
def smoke():
    smoke_adapters.remote()
    for g in ("H100", "A100-80GB", "T4"):
        smoke_gpu.with_options(gpu=g).remote(g)


@app.local_entrypoint()
def smoke_gpus(gpus: str = "A100-80GB,T4"):
    """Smoke test on the listed GPUs only (the H100 was run separately). Builds the tiny adapters first."""
    smoke_adapters.remote()
    for g in gpus.split(","):
        smoke_gpu.with_options(gpu=g).remote(g)


@app.local_entrypoint()
def prep():
    prep_remote.remote()


@app.local_entrypoint()
def train(model: str = "gemma-2b", method: str = "lora", rank: int = 16, gpu: str = "H100", seed: int = 0,
          extra: str = ""):
    n = 2 if gpu == "H100:2" else 1
    g = GPUS[gpu]
    args = f"--model {model} --method {method} --rank {rank} --seed {seed} --gpu {g.split(':')[0]} {extra}"
    train_remote.with_options(gpu=g).remote(args, n)


@app.local_entrypoint()
def merge(model: str, run: str):
    merge_remote.remote(f"--model {model} --run {run}")


@app.local_entrypoint()
def evaluate(model: str, task: str, run: str = "", merged: bool = False, gpu: str = "H100"):
    args = f"--model {model} --task {task}" + (f" --run {run}" if run else "") + (" --merged" if merged else "")
    serve_remote.with_options(gpu=GPUS[gpu]).remote("eval_quality.py", args)


@app.local_entrypoint()
def serve_bench(model: str = "llama-3.1-8b", run: str = "", mode: str = "base", adapters: int = 1,
                gpu: str = "H100", rates: str = ""):
    args = (f"--model {model} --mode {mode} --adapters {adapters} --gpu {gpu}" + (f" --run {run}" if run else "")
            + (f" --rates {rates}" if rates else ""))
    serve_remote.with_options(gpu=GPUS[gpu]).remote("serve_bench.py", args)


@app.local_entrypoint()
def distill(arm: str, student: str = "llama-3.2-1b", make_teacher_data: bool = False):
    if make_teacher_data:
        serve_remote.remote("distill.py", "--arm sft --make-teacher-data")
    else:
        distill_remote.remote(f"--arm {arm} --student {student}")


@app.local_entrypoint()
def spectrum():
    spectrum_remote.remote()


@app.local_entrypoint()
def collect():
    collect_remote.remote()
