"""Code 11.1: a two-layer MLP in JAX with grad, vmap and jit. Runs on CPU, GPU or TPU (pip install jax)."""
import jax, jax.numpy as jnp


def mlp(params, x):
    h = jax.nn.gelu(x @ params["w1"])
    return h @ params["w2"]


def loss(params, x, y):
    return jnp.mean((mlp(params, x) - y) ** 2)


k1, k2 = jax.random.split(jax.random.key(0))
params = {"w1": jax.random.normal(k1, (512, 2048)) * 0.02, "w2": jax.random.normal(k2, (2048, 512)) * 0.02}
train_grad = jax.jit(jax.grad(loss))
per_example = jax.vmap(mlp, in_axes=(None, 0))
x = jnp.ones((64, 512)); y = jnp.zeros((64, 512))
g = train_grad(params, x, y)
print(jax.tree.map(jnp.shape, g))
print(per_example(params, x).shape)
print(train_grad.lower(params, x, y).as_text()[:300])
