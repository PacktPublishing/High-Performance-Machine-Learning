# Optional: the E8 comparison on Kubernetes with llm-d 1.0

Prerequisites: a Kubernetes cluster with at least 4 GPUs, Helm, a Gateway API implementation supported by llm-d
(for example Istio or kgateway), and a Hugging Face token stored as a Kubernetes secret:

    kubectl create secret generic hf-token --from-literal=HF_TOKEN=<your token>     # typed by you, never committed

Steps (follow the llm-d 1.0 "well-lit path" guides; chart names and values keys change between releases):
1. Install the llm-d "Optimized Baseline" path with `values-baseline.yaml` (4 vLLM pods, load-aware routing).
2. Run `loadgen_agent.py --exp E7` against the gateway URL (set VLM_URL) at the E8 load points.
3. Switch to the "Precise Prefix-Cache Aware Routing" path with `values-precise.yaml` and repeat.
4. Optional: the P/D disaggregation path with 2 prefill and 2 decode pods on the multi-hop questions only.
Record the llm-d chart version, image digests and node GPU type in results/k8s_pins.json.
