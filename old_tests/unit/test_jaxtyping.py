import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
@jax.jit
def fn(x: Float[Array, "1"]) -> Float[Array, "1"]:
    return x


fn(jnp.array([1.0]))
