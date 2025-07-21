from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from data_science_utils.dynamical_systems import (
    AbstractStochasticDiscreteDynamicalSystem,
)


class RandomWalk(AbstractStochasticDiscreteDynamicalSystem):
    """This module implements a standard Random Walk."""

    _dimension: int = eqx.static_field()

    def __init__(self, dimension: int):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def initial_state(self, key):
        return jnp.zeros(self.dimension)

    def forward(
        self,
        state,
        key,
    ):
        return jax.random.multivariate_normal(
            key, mean=state, cov=10 * jnp.eye(state.shape[0])
        )
