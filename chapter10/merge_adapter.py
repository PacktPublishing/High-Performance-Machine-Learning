"""Merge a LoRA/QLoRA adapter into the 16-bit base model (Chapter 10).

A QLoRA adapter is merged into the BF16 base (not the NF4 weights it was trained against); eval_quality.py --kl
measures the effect of that mismatch.
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODELS, RUNS

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--model", required=True)
a = ap.parse_args()

base = AutoModelForCausalLM.from_pretrained(MODELS[a.model], dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, f"{RUNS}/{a.run}").merge_and_unload()
merged.save_pretrained(f"{RUNS}/{a.run}-merged", safe_serialization=True)
AutoTokenizer.from_pretrained(MODELS[a.model]).save_pretrained(f"{RUNS}/{a.run}-merged")
print("merged ->", f"{RUNS}/{a.run}-merged")
