import jax
import jax.numpy as jnp

import uuid
print(uuid.uuid4())
def f():
    return jnp.array([jnp.nan])

key = jax.random.key(0)
key, subkey = jax.random.split(key)

arr = jax.random.normal(subkey, shape=(100,100))
idx = jax.random.bernoulli(subkey, shape=(30,100))

idx = jnp.concat([idx, jnp.tile(False, (70,100))])
arr_with_nan = arr.at[idx].set(jnp.nan)
print(jnp.sum(arr_with_nan, axis=1))
# print(idx)
