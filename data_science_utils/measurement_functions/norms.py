import jax
import jax.numpy as jnp
from typing import Optional
from jaxtyping import jaxtyped, Float, Array, Key
from beartype import beartype as typechecker
import equinox as eqx


@jaxtyped(typechecker=typechecker)
@jax.jit
def norm_measurement(state: Float[Array, "*batch_dim state_dim"],
                     key: Optional[Key[Array, "1"]] = None,
                     covariance: Float[Array, "measurement_dim measurement_dim"] = jnp.array([[1.0]])) -> Float[Array, "measurement_dim"]:
    perfect_measurement = jnp.linalg.norm(state)
    noise = 0 if key is None else jnp.sqrt(covariance.value) * jax.random.normal(key)
    return jnp.array([perfect_measurement + noise])


@jaxtyped(typechecker=typechecker)
class Distance(eqx.Module):
    covariance: Float[Array, "measurement_dim measurement_dim"]

    def __call__(self, state, key=None):
        return norm_measurement(state=state, key=key, covariance=self.covariance)
