"""Chapter 11 follow-ups on Modal (2026-11-16), after the independent review.

  cd chapter11 && modal run h100/modal_ch11_fixups.py

1. onnx_timing: Code 11.4 timing with outputs kept on the GPU (ONNX Runtime IO binding) next to the earlier
   host-copy timing, and PyTorch with and without copying the logits to the host, so both sides do the same work.
2. dtensor: the PyTorch half of Code 11.2 on 2 H100s with torchrun (DTensor placements and collective counts).
Writes results/modal_fixups_<date>.json.
"""
import json, os, time
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, "..")
app = modal.App("hpml-ch11-fixups")
hf = modal.Volume.from_name("hf-cache", create_if_missing=True)
SECRETS = [modal.Secret.from_name("huggingface")]
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.14.1", "transformers", "onnx", "onnxscript", "onnxruntime-gpu", "datasets", "numpy")
       .add_local_file(os.path.join(CH, "common.py"), "/root/ch11/common.py")
       .add_local_file(os.path.join(CH, "sharding_compare.py"), "/root/ch11/sharding_compare.py"))


@app.function(image=img, gpu="H100!", secrets=SECRETS, volumes={"/root/.cache/huggingface": hf}, timeout=3600)
def onnx_timing():
    import sys, traceback
    sys.path.insert(0, "/root/ch11")
    try:
        import numpy as np, torch, onnxruntime as ort
        from common import MAX_PROMPT, load_prompts, chat, environment
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = "google/gemma-2b-it"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
        model.config.use_cache = False
        ids = tok(chat(tok, load_prompts()[0]), return_tensors="pt", padding="max_length", max_length=MAX_PROMPT,
                  truncation=True).input_ids
        x = ids.cuda()

        def tmed(fn, n=10):
            for _ in range(3): fn()
            ts = []
            for _ in range(n):
                torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) * 1e3)
            return round(sorted(ts)[n // 2], 2)
        with torch.no_grad():
            torch_gpu = tmed(lambda: model(x).logits)
            torch_host = tmed(lambda: model(x).logits.cpu())
        batch, seq = torch.export.Dim("batch", max=16), torch.export.Dim("seq", max=4096)
        prog = torch.onnx.export(model, (x,), dynamo=True, dynamic_shapes=({0: batch, 1: seq},))
        prog.optimize(); prog.save("/tmp/gemma2b.onnx", external_data=True)
        sess = ort.InferenceSession("/tmp/gemma2b.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inp, out = sess.get_inputs()[0].name, sess.get_outputs()[0].name
        ort_host = tmed(lambda: sess.run(None, {inp: ids.numpy()}))
        io = sess.io_binding()
        xo = ort.OrtValue.ortvalue_from_numpy(ids.numpy(), "cuda", 0)
        io.bind_ortvalue_input(inp, xo)
        io.bind_output(out, "cuda")

        def ort_dev():
            sess.run_with_iobinding(io)
        ort_device = tmed(ort_dev)
        return {"torch_fp16_ms_logits_on_gpu": torch_gpu, "torch_fp16_ms_logits_to_host": torch_host,
                "ort_fp16_ms_outputs_to_host": ort_host, "ort_fp16_ms_iobinding_on_gpu": ort_device,
                "logits_mb": round(512 * 256000 * 2 / 1e6), "env": environment({"gpu": "H100"})}
    except Exception as e:
        return {"error": repr(e)[:500], "trace": traceback.format_exc()[-2000:]}


@app.function(image=img, gpu="H100!:2", timeout=1200)
def dtensor():
    import subprocess
    p = subprocess.run("cd /root/ch11 && torchrun --nproc_per_node=2 sharding_compare.py torch", shell=True,
                       capture_output=True, text=True)
    return {"returncode": p.returncode, "stdout": p.stdout[-3000:], "stderr": p.stderr[-2000:]}


@app.local_entrypoint()
def main():
    a, b = onnx_timing.spawn(), dtensor.spawn()
    out = {"source": "Modal H100 (modal_ch11_fixups.py)", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "onnx_timing": a.get(), "dtensor": b.get()}
    path = os.path.join(CH, "results", f"modal_fixups_{time.strftime('%Y-%m-%d_%H%M')}.json")
    json.dump(out, open(path, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "onnx_timing"} | {"onnx": {k: v for k, v in out["onnx_timing"].items() if k != "env"}}, indent=1, default=str)[:3000])
    print("saved", path)
