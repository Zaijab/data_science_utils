import equinox as eqx
import jax
import jax.numpy as jnp
from flax import nnx


@jax.jit
def expensive_computation(x):
    return jnp.exp(x) + (jnp.log(x**2) + 1) ** 2


@jax.jit
def masked_computation(x, mask):
    return jnp.where(mask, expensive_computation(x), jnp.inf)


# jax.jit


# @eqx.filter_jit
def static_arg_array_size(n):
    return jnp.arange(n)


@eqx.filter_jit
def random_size_array(key):
    size = jax.random.poisson(key, lam=10)
    return static_arg_array_size(size)


key = jax.random.key(42)

sizes = jax.random.poisson(key, lam=10, shape=(10,))
jax.tree.map(static_arg_array_size, sizes)

# random_size_array(key)
# static_arg_array_size(10)

# batch_size = 10_000

# key, subkey = jax.random.split(key)
# data = jax.random.normal(subkey, shape=(batch_size,))


# for mask_prob in [0.1, 0.5, 1.0]:
#     print(mask_prob, end=": ")
#     key, subkey = jax.random.split(key)
#     mask = jax.random.uniform(subkey, shape=(batch_size,)) <= mask_prob
#     print(jax.numpy.compress(mask, data).shape)
#     masked_computation(data, mask)  # warmup
#     # %timeit masked_computation(data, mask).block_until_ready()
