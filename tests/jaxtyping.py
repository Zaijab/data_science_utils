from jaxtyping import Float, Array, jaxtyped

from beartype import beartype as typechecker
from functools import partial
from collections.abc import Callable

import jax
import jax.numpy as jnp


 
import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['u'])
def f(x, adder, u=0.9):
    return adder(x, x + u)


my_func = jax.tree_util.Partial(f, u=0.9)
my_adder = jax.tree_util.Partial(lambda x, y: x + y)

my_func(jnp.array([1.0, 2.0]), my_adder)
