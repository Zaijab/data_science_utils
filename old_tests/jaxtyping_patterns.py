"""
This module tests Jaxtyping utilities and common use case patterns.
Namely, we see how to use Jaxtyping with correctly notated inputs, outputs, and statically compiled arguments.
"""

from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['debug'])
def f(x: Float[Array, "state_dim"], debug: bool = False) -> Float[Array, "2*state_dim"]:
    """
    This function takes in an array and outputs an array.
    The type signatures are checked by `beartype` and will raise an exception if the output is not of the expected shape.
    Internal type signatures are not typechecked. So we use a debug flag to get the exact shape after every operation.
    Checking will cause some slowdown so typical use will be having debug on false and all the debug code goes away.
    """
    y = x + 1
    if debug:
        assert isinstance(y, Float[Array, "state_dim"]) # Test passes as expected, and runs if told
        # assert isinstance(y, Float[Array, "state_dim + 10"]) # Uncomment for angry Python
        
    return jnp.zeros(x.shape[0] * 2)


@jaxtyped(typechecker=typechecker)
class myClass(eqx.Module):
    x: Float[Array, "2"]
    # y: Float[Array, "2"] # Not initialized during init

    def __init__(self, myinput):
        self.x = jnp.array([1.2, 1.0])
        # self.x = jnp.array([1, 1]) # Uncomment for angry Python, wrong dtype
        # self.x = jnp.eye(3) # Uncomment for angry Python, wrong shape

    def update(self):
        self.x = self.x + 1

# f(jnp.zeros(2), debug=True)

myClass(jnp.array([1.0,2.0])).update()

