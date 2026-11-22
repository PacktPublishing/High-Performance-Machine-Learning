"""Lower-bound estimate of fine-tuning time from FLOPs (Chapter 10).

FLOPs per token: forward 2N; backward 2N for input gradients and, without frozen weights, 2N for weight gradients;
gradient checkpointing adds one more forward pass. MFU is the assumed fraction of peak BF16 throughput.
"""
N = {"Llama-3.1-8B": 8.03e9, "Gemma-2B": 2.51e9}
PEAK = {"H100": 989e12, "A100": 312e12}        # dense BF16 FLOP/s
TOKENS = 2 * 7e6                                # two epochs of the case-study training set
MFU = 0.40


def flops_per_token(n, method, checkpointing=True):
    fwd, bwd_in, bwd_w = 2 * n, 2 * n, (2 * n if method == "full" else 0)
    return fwd + bwd_in + bwd_w + (fwd if checkpointing else 0)


print(f"{'model':13s} {'method':6s} {'GPU':5s} {'PFLOP':>7s} {'hours':>6s}")
for model, n in N.items():
    for method in ("full", "lora"):
        for gpu, peak in PEAK.items():
            total = flops_per_token(n, method) * TOKENS
            print(f"{model:13s} {method:6s} {gpu:5s} {total / 1e15:7.0f} {total / (MFU * peak) / 3600:6.2f}")
