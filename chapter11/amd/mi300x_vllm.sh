#!/usr/bin/env bash
# Chapter 11 case study, optional MI300X rows (AMD Developer Cloud, one MI300X, about one hour).
# Uses AMD's ROCm vLLM container; HF_TOKEN must already be exported in the shell from your own secret store.
# Compares the portable Triton attention backend with AMD's AITER backend on Gemma-2B-IT.
set -euo pipefail
: "${HF_TOKEN:?export HF_TOKEN first (do not write it into this file)}"
IMG=${IMG:-rocm/vllm:latest}   # record the exact tag and digest printed below in the results
docker pull "$IMG"; docker image inspect "$IMG" --format '{{.Id}}' | tee image_digest.txt
for BACKEND in TRITON_ATTN ROCM_AITER_FA; do
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
    -e HF_TOKEN -e VLLM_ROCM_USE_AITER=$([ "$BACKEND" = ROCM_AITER_FA ] && echo 1 || echo 0) \
    -v "$PWD":/work -w /work "$IMG" \
    python3 amd/bench_vllm.py --backend "$BACKEND" --out "results_mi300x_${BACKEND}.json"
done
echo "Copy results_mi300x_*.json back and merge them as mi300x_vllm_triton and mi300x_vllm_aiter."
