import jax
import jax.numpy as jnp


@jax.jit
def expensive_computation(x):
    return jnp.exp(x) + (jnp.log(x**2) + 1) ** 2


@jax.jit
def masked_computation(x, mask):
    return jax.lax.select(mask, expensive_computation(x), jnp.jnp.inf)
    return jnp.where(mask, expensive_computation(x), jnp.inf)


key = jax.random.key(42)
batch_size = 10_000

key, subkey = jax.random.split(key)
data = jax.random.normal(subkey, shape=(batch_size,))


for mask_prob in [0.1, 0.5, 1.0]:
    print(mask_prob, end=": ")
    key, subkey = jax.random.split(key)
    mask = jax.random.uniform(subkey, shape=(batch_size,)) <= mask_prob
    print(jax.numpy.compress(mask, data).shape)
    masked_computation(data, mask)  # warmup
    # %timeit masked_computation(data, mask).block_until_ready()
