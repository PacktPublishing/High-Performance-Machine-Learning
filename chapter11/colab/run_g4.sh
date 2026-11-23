set -u
for m in google/gemma-2b-it:gemma2b meta-llama/Llama-3.1-8B-Instruct:llama8b; do
  export CH11_MODEL=${m%%:*} CH11_TAG=${m##*:} CH11_REPEATS=3
  python gpu_reference_v3.py torch 2>&1 | grep -E '^rtxpro6000_|Error'
  python gpu_reference_v3.py vllm 2>&1 | grep -E '_vllm_'
  CH11_QUANT=fp8 python gpu_reference_v3.py vllm 2>&1 | grep -E '_vllm_'
done
export CH11_MODEL=meta-llama/Llama-3.1-8B-Instruct CH11_TAG=llama8b CH11_REPEATS=3
CH11_VLLM_MODEL=nvidia/Llama-3.1-8B-Instruct-FP4 CH11_VLABEL=nvfp4 python gpu_reference_v3.py vllm 2>&1 | grep -E '_vllm_' | tee nv.txt
if grep -q '"error"' nv.txt; then CH11_VLLM_MODEL=RedHatAI/Llama-3.1-8B-Instruct-NVFP4 CH11_VLABEL=nvfp4_redhat python gpu_reference_v3.py vllm 2>&1 | grep -E '_vllm_'; fi
python determinism_colab.py default 2>&1 | grep -E "C=|failed|saved"
python determinism_colab.py invariant 2>&1 | grep -E "C=|failed|saved"
echo ALLDONE
