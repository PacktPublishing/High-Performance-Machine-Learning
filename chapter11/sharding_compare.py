"""Code 11.2: the same row-sharded weight in PyTorch DTensor and in JAX.

PyTorch part: torchrun --nproc_per_node=2 sharding_compare.py torch   (2 GPUs)
JAX part:     python sharding_compare.py jax   (all local devices; on CPU set
              XLA_FLAGS=--xla_force_host_platform_device_count=8 to simulate 8 devices)
Both parts compute y = x @ w.T with w split by rows and print where the result lives and which
collectives were inserted. Neither framework needs one here: the result stays split by columns.
"""
import sys

if sys.argv[1:] == ["torch"]:
    import torch, torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard, Replicate
    from torch.distributed.tensor.debug import CommDebugMode
    dist.init_process_group("nccl"); torch.cuda.set_device(dist.get_rank())
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("data",))
    torch.manual_seed(0)
    w = distribute_tensor(torch.randn(8192, 4096), mesh, [Shard(0)])        # rows split
    x = distribute_tensor(torch.randn(16, 4096), mesh, [Replicate()])
    with CommDebugMode() as comm:
        y = x @ w.T                                                          # stays sharded
    print(dist.get_rank(), "local shard", tuple(w.to_local().shape), "output", tuple(y.shape),
          y.placements, "collectives", comm.get_total_counts())
    with CommDebugMode() as comm:
        full = y.full_tensor()                                               # explicit all-gather
    print(dist.get_rank(), "full", tuple(full.shape), "collectives", dict(comm.get_comm_counts()))
    dist.destroy_process_group()
else:
    import jax, jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    mesh = jax.make_mesh((jax.device_count(),), ("data",))
    w = jax.device_put(jnp.ones((8192, 4096)), NamedSharding(mesh, P("data", None)))
    f = jax.jit(lambda x, w: x @ w.T)
    y = f(jnp.ones((16, 4096)), w)
    print("output", y.shape, y.sharding)
    hlo = f.lower(jnp.ones((16, 4096)), w).compile().as_text()
    print("collectives", [op for op in ("all-gather", "all-reduce", "reduce-scatter", "all-to-all") if op in hlo])

    def local_sum(a):
        return jax.lax.psum(a.sum(axis=0), "data")
    total = jax.shard_map(local_sum, mesh=mesh, in_specs=P("data", None), out_specs=P(None))(w)
    print("column sums", total.shape, float(total[0]))
