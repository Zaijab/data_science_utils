"""
This module tests Jaxtyping utilities and common use case patterns.
Namely, we see how to use Jaxtyping with correctly notated inputs, outputs, and statically compiled arguments.
"""

from functools import partial

import jax
import jax.numpy as jnp
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


f(jnp.zeros(2), debug=True)
