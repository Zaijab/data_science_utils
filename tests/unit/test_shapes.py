import jax
import jax.numpy as jnp


def f(key):
    return jnp.arange(jax.random.poisson(key, 10.0))

f = jax.jit(f, static_argnums=(0,))
f(jax.random.key(0))
