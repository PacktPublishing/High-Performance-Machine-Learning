"""Singular values of the weight change ΔW in one MLP projection of Gemma-2B (Chapter 10, Figure 10.12).

Compares full fine-tuning with merged LoRA adapters at ranks 16 and 256.
"""
import argparse

import torch
from safetensors import safe_open

from common import MODELS, RESULTS, RUNS, save_json


def load_weight(path, name):
    import glob
    for f in glob.glob(f"{path}/*.safetensors"):
        with safe_open(f, "pt") as s:
            if name in s.keys():
                return s.get_tensor(name).float()
    raise KeyError(name)


ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=9)
ap.add_argument("--proj", default="down_proj")
ap.add_argument("--runs", default="gemma-2b-full-r16-s0-H100,gemma-2b-lora-r16-s0-H100,gemma-2b-lora-r256-s0-H100")
a = ap.parse_args()

from huggingface_hub import snapshot_download  # noqa: E402
base_dir = snapshot_download(MODELS["gemma-2b"], allow_patterns=["*.safetensors", "*.json"])
name = f"model.layers.{a.layer}.mlp.{a.proj}.weight"
w0 = load_weight(base_dir, name)
out = {}
for run in a.runs.split(","):
    path = f"{RUNS}/{run}" if "full" in run else f"{RUNS}/{run}-merged"
    dw = load_weight(path, name) - w0
    s = torch.linalg.svdvals(dw.cuda() if torch.cuda.is_available() else dw).cpu()
    label = "full fine-tuning" if "full" in run else f"LoRA r={run.split('-r')[1].split('-')[0]}"
    out[label] = s[:512].tolist()
save_json(out, f"{RESULTS}/spectrum.json")
print({k: v[:3] for k, v in out.items()})
