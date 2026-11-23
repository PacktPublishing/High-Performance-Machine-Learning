"""Serve Qwen3-VL-8B-Instruct with vLLM 0.33.0 on Modal, one web endpoint per experiment configuration.

    modal deploy serve_vlm.py                      # deploys the server class
    EXP=E2 modal run serve_vlm.py::smoke           # starts E2 and sends one image+tool request

The Hugging Face token is read from the Modal secret "huggingface" (HF_TOKEN). Never put a token in code.
"""
import json
import os
import subprocess
import time

import modal
import yaml

VLLM_VERSION = os.environ.get("VLLM_VERSION", "0.33.0")       # R1 sets 0.32.0
GPU = os.environ.get("GPU", "H100")
EXP = os.environ.get("EXP", "E2")

image = (modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.11")
         .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]", "pyyaml", "requests")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
         .add_local_dir(os.path.join(os.path.dirname(__file__), "configs"), "/root/configs"))
if VLLM_VERSION.startswith("0.32"):
    image = (modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
             .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]", "pyyaml", "requests")
             .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
             .add_local_dir(os.path.join(os.path.dirname(__file__), "configs"), "/root/configs"))

hf_cache = modal.Volume.from_name("hpml-hf-cache", create_if_missing=True)
app = modal.App(f"hpml-ch12-vlm-{EXP.lower()}-{GPU.lower()}")


def resolve(exp_id, path="/root/configs/experiments.yaml"):
    cfg = yaml.safe_load(open(path))
    common, exp = cfg["common"], dict(cfg[exp_id])
    while "base" in exp:                      # E8, E9, R1, R2 inherit from another experiment
        base = dict(cfg[exp.pop("base")]); base.update(exp); exp = base
    out = {**common, **{k: v for k, v in exp.items() if k not in ("agent", "retriever")}}
    out["agent"] = {**common["agent"], **exp.get("agent", {})}
    out["retriever"] = {**common["retriever"], **exp.get("retriever", {})}
    out["vllm_args"] = common["vllm_args"] + exp.get("extra_vllm_args", [])
    return out


@app.cls(image=image, gpu=GPU, timeout=3600, scaledown_window=600,
         secrets=[modal.Secret.from_name("huggingface")], volumes={"/root/.cache/huggingface": hf_cache})
@modal.concurrent(max_inputs=512)
class VLM:
    @modal.enter()
    def start(self):
        c = resolve(EXP)
        env = dict(os.environ, **c.get("env", {}))
        cmd = ["vllm", "serve", c["model"], "--served-model-name", c["served_name"], "--port", "8000"]
        if c.get("revision"):
            cmd += ["--revision", c["revision"]]
        cmd += c["vllm_args"]
        print("launch:", " ".join(cmd), "env:", c.get("env", {}))
        self.proc = subprocess.Popen(cmd, env=env)
        import requests
        for _ in range(900):
            try:
                if requests.get("http://localhost:8000/health", timeout=2).ok:
                    return
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("vLLM did not become healthy")

    @modal.web_server(8000, startup_timeout=900)
    def serve(self):
        pass


@app.local_entrypoint()
def smoke():
    """Print the URL; agent.py and loadgen_agent.py read it from VLM_URL."""
    print("deployed URL:", VLM().serve.get_web_url())
