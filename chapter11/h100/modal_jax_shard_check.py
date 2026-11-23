"""Code 11.2, JAX half, on 8 simulated CPU devices (no GPU needed, about a minute).

  cd chapter11 && modal run h100/modal_jax_shard_check.py

Prints the output sharding of x @ w.T with w row-sharded and the collectives XLA put in the
compiled program. Writes results/modal_jax_shard_<date>_<HHMM>.json.
"""
import json, os, time
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.join(HERE, "..")
app = modal.App("hpml-ch11-jax-shard")
img = (modal.Image.debian_slim(python_version="3.11").pip_install("jax[cpu]")
       .env({"XLA_FLAGS": "--xla_force_host_platform_device_count=8"})
       .add_local_file(os.path.join(CH, "sharding_compare.py"), "/root/sharding_compare.py"))


@app.function(image=img, cpu=4, timeout=600)
def jax_shard():
    import subprocess
    p = subprocess.run("python /root/sharding_compare.py jax", shell=True, capture_output=True, text=True)
    import jax, jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    mesh = jax.make_mesh((jax.device_count(),), ("data",))
    w = jax.device_put(jnp.ones((8192, 4096)), NamedSharding(mesh, P("data", None)))
    f = jax.jit(lambda x, w: x @ w.T)
    hlo = f.lower(jnp.ones((16, 4096)), w).compile().as_text()
    ops = [l.strip()[:160] for l in hlo.splitlines()
           if any(k in l for k in ("all-gather", "all-reduce", "reduce-scatter", "all-to-all", "collective-permute"))]
    return {"jax": jax.__version__, "devices": jax.device_count(), "stdout": p.stdout, "stderr": p.stderr[-1500:],
            "collective_lines": ops}


@app.local_entrypoint()
def main():
    out = jax_shard.remote()
    print(json.dumps(out, indent=1))
    path = os.path.join(CH, "results", f"modal_jax_shard_{time.strftime('%Y-%m-%d_%H%M')}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("saved", path)
