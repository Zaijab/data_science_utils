import equinox as eqx
import jax
import jax.numpy as np
from jaxtyping import Float, Array, Key, jaxtyped
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
class Projection(eqx.Module):
    """Doing Stuff"""

    measurement_dim: int
    observation_matrix: Float[Array, "measurement_dimension state_dimension"]
    observation_cov: Float[Array, "measurement_dimension measurement_dimension"]
    probability_of_detection: float

    poisson_clutter_rate: float
    clutter_region: jnp.array
    clutter_density: jnp.array


measure = Projection(
    observation_matrix=jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    ),
    observation_covariance=jnp.diag([100.0, 100.0, 100.0]),
)
