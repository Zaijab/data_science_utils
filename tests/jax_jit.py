import jax
import jax.numpy as jnp
from functools import partial

# Same performance!

@partial(jax.jit, static_argnames=['key'])
def f_compact(x, key=None):
    return x @ x + x @ x

f_compact(jnp.eye(2), jax.random.key(42))
