# Chapter 11: Navigating the Accelerator and Compiler Landscape

**Status: case study run on 2026-11-15 to 2026-11-23.** The measured results are in `results/`.

Companion code for Chapter 11. The case study runs the RAG pipeline's two generators, Gemma-2B-IT and
Llama-3.1-8B-Instruct, on NVIDIA A100 and H100 GPUs and on Google Cloud TPU v5e and v6e. For each configuration it
measures prefill time, time per output token at batch size 1 and numerical agreement with an FP32 CPU reference,
with PyTorch eager, `torch.compile`, vLLM 0.33.0 (BF16 and FP8 weights), TorchTPU and JAX. Separate experiments
measure batch invariance (determinism) in vLLM, ONNX Runtime against PyTorch, and how DTensor and JAX place the
collectives of a sharded matrix multiplication.

## Scripts

| Script | What it does |
|---|---|
| `common.py` | Shared settings: prompts, prompt length, number of new tokens, environment record |
| `jax_basics.py` | Code 11.1: a two-layer MLP in JAX with `grad`, `vmap` and `jit` (CPU, GPU or TPU) |
| `sharding_compare.py` | Code 11.2: the same row-sharded weight in PyTorch DTensor (`torchrun`, 2 GPUs) and in JAX |
| `colab/gpu_reference_v3.py` | GPU rows (eager, `torch.compile`, vLLM BF16 and FP8), median of repeats, top-1 agreement |
| `colab/determinism_colab.py` | Batch-invariance experiment: default and `VLLM_BATCH_INVARIANT=1` at 1, 16 and 64 concurrent requests |
| `colab/*.ipynb` | Colab notebooks for the GPU rows, the TPU rows and a GPU check |
| `h100/modal_h100_all.py` | All H100 rows and the determinism experiment on Modal in one command (`gpu="H100!"` pins the H100) |
| `h100/modal_ch11_fixups.py` | ONNX Runtime with and without IO binding, and the DTensor check on 2 H100s |
| `h100/modal_jax_shard_check.py` | The JAX half of Code 11.2 on 8 simulated CPU devices |
| `h100/modal_reference.py` | Earlier single-GPU reference script (kept for the commands in its docstring) |
| `tpu/torchtpu_gemma.py` | Code 11.3: TPU rows with TorchTPU (`torch.compile(backend="tpu")`) |
| `tpu/jax_gemma.py`, `tpu/run_jax.sh` | JAX rows on a TPU with KerasHub |
| `onnx/export_and_run.py` | Code 11.4: `torch.export` to ONNX and ONNX Runtime with the CUDA execution provider |
| `parity/cpu_reference.py` | FP32 CPU reference: greedy tokens and teacher-forced logits |
| `parity/compare_logits.py` | Code 11.5: KL divergence and top-1 agreement of a platform's logits with the reference |
| `determinism/run_determinism.py` | Stand-alone version of the batch-invariance experiment |
| `amd/bench_vllm.py`, `amd/mi300x_vllm.sh` | Optional MI300X rows with vLLM on ROCm (not part of the measured case study) |

## Results

| File | Platform and content |
|---|---|
| `results/colab_a100_r3_2026-11-15.json` | A100 80GB (Colab): both models, eager, `torch.compile`, vLLM; median of 3 |
| `results/colab_a100_determinism_2026-11-15.json` | A100: batch invariance, default and invariant mode |
| `results/modal_h100_2026-11-15*.json` | H100 80GB SXM (Modal): both models, all stacks, FP8, determinism |
| `results/modal_fixups_2026-11-16_0109.json` | H100: ONNX Runtime timings with outputs on the host and on the GPU; DTensor collectives |
| `results/cluster_h100_2026-11-23/` | H100 (second host): repeat of the GPU rows with logs |
| `results/colab_tpu_v6e1_*.json`, `results/colab_tpu_v5e1_*.json` | TPU v6e and v5e (Colab): TorchTPU rows and repeats |
| `results/colab_tpu_v5e1_jax_2026-11-16.json` | TPU v5e: JAX rows and the precision comparison |
| `results/bluevela_jax_shard_2026-11-19.json` | JAX sharding check on 8 CPU devices: output sharding and the collectives in the compiled program |

Each result file records the GPU or TPU, the driver and the package versions.

## Stack

vLLM 0.33.0, PyTorch 2.14, transformers 5.x, ONNX Runtime 1.32 (CUDA execution provider); JAX and TorchTPU on Colab
TPU runtimes. Gated models (Gemma, Llama) need a Hugging Face token: on Modal a secret named `huggingface`, on Colab
a Colab secret.
