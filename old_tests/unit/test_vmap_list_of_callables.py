import jax
import jax.numpy as jnp


def f(x):
    return x


funcs = [f] * 100
index = jnp.arange(len(funcs))

jax.tree.map(lambda leaf, index: leaf(index), funcs, (index,))
