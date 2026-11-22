"""Static training memory per method for the two case-study generators (Chapter 10).

Counts weights, gradients and optimizer states only; activations depend on batch size and sequence length.
Full fine-tuning: BF16 weights and gradients, FP32 master weights, FP32 Adam moments (16 bytes per parameter).
LoRA: frozen BF16 base; adapter in FP32 with gradients and Adam moments (16 bytes per adapter parameter).
QLoRA: linear layers in NF4 with double quantization (4.127 bits per parameter); embeddings in BF16.
"""

CONFIGS = {
    # hidden, intermediate, layers, query heads, KV heads, head dim, vocab, tied embeddings
    "Llama-3.1-8B": dict(d=4096, ff=14336, L=32, h=32, kv=8, hd=128, vocab=128256, tied=False),
    "Gemma-2B": dict(d=2048, ff=16384, L=18, h=8, kv=1, hd=256, vocab=256000, tied=True),
}


def linear_shapes(c):
    q, kv = c["h"] * c["hd"], c["kv"] * c["hd"]
    d, ff = c["d"], c["ff"]
    return [(d, q), (d, kv), (d, kv), (q, d), (d, ff), (d, ff), (ff, d)]  # q, k, v, o, gate, up, down


def count(c, rank):
    shapes = linear_shapes(c)
    linear = c["L"] * sum(i * o for i, o in shapes)
    embed = c["vocab"] * c["d"] * (1 if c["tied"] else 2)
    lora = c["L"] * sum(rank * (i + o) for i, o in shapes)
    return linear, embed, lora


GB = 1e9
print(f"{'model':13s} {'params':>7s} {'full FT':>8s} {'LoRA':>7s} {'QLoRA':>7s} {'adapter':>8s}")
for name, c in CONFIGS.items():
    linear, embed, lora = count(c, rank=16)
    total = linear + embed
    full = 16 * total
    lora_mem = 2 * total + 16 * lora
    qlora_mem = linear * 4.127 / 8 + 2 * embed + 16 * lora
    print(f"{name:13s} {total / GB:6.2f}B {full / GB:6.1f}GB {lora_mem / GB:5.1f}GB {qlora_mem / GB:5.1f}GB"
          f" {lora / 1e6:6.1f}M")
