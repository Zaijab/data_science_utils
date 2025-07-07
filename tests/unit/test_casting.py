import jax
import jax.numpy as jnp

@jax.jit
def f(x):
    return float(x)

f(jnp.asarray(1.0))
