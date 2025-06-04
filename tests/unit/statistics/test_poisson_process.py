import jax
import jax.numpy as jnp

from data_science_utils.statistics.poisson_point_process import (
    poisson_point_process_hyperrectangle,
)


def test_poisson_process() -> None:
    poisson_point_process_hyperrectangle(
        jax.random.key(0),
        10.0,
        jnp.array([[0.0, 200.0], [0.0, 200.0], [0.0, 400.0]]),
        1000,
    )
