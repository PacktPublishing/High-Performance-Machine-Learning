"""Create untrained rank-16 LoRA adapters (B = 0) for the vLLM smoke test, one per model.

For Qwen3-VL the adapter targets only the language model's linear layers.
"""
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from common import MODELS, VOL

for key in ("llama-3.1-8b", "gemma-2b", "llama-3.2-1b", "qwen3-vl-8b", "gemma-4-e2b"):
    out = f"{VOL}/smoke_adapters/{key}"
    if os.path.exists(out):
        continue
    try:
        if key in ("qwen3-vl-8b", "gemma-4-e2b"):
            m = AutoModelForImageTextToText.from_pretrained(MODELS[key], dtype=torch.bfloat16)
            targets = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        else:
            m = AutoModelForCausalLM.from_pretrained(MODELS[key], dtype=torch.bfloat16)
            targets = "all-linear"
        pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=targets))
        pm.save_pretrained(out)
        print("adapter", key, out)
    except Exception as e:
        print("FAILED", key, repr(e)[:300])
