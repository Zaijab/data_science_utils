import jax
import jax.numpy as jnp
from jaxtyping import jaxtyped, Float, Array, Key
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
@jax.jit
def norm_measurement(
        state: Float[Array, "*batch_dim state_dim"],
        key: Key[Array, ""] | None = None,
        covariance: Float[Array, "1 1"] = jnp.array([[1.0]])
) -> Float[Array, "1"]:
    perfect_measurement = jnp.linalg.norm(state)
    noise = 0 if key is None else jnp.sqrt(covariance) * jax.random.normal(key)
    return (perfect_measurement + noise).reshape(-1)

