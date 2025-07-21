"""
An interesting paper discusses a common ODE form of Lorenz systems. This module defines said systems and numerically integrates it via Diffrax.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from jaxtyping import Array, Float, PyTree, jaxtyped
from beartype import beartype as typechecker
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, ConstantStepSize
import matplotlib.pyplot as plt
from functools import partial
from typing import Tuple, Optional


class GeneralLorenzEquations4(eqx.Module):
    beta: float = 6.0
    rho: float = 8.0
    gamma: float = 0.0
    dt: float = 0.01

    @eqx.filter_jit
    def vector_field(
        self, t: float, y: Float[Array, "4"], args: PyTree
    ) -> Float[Array, "4"]:
        """
        Compute the vector field for the GLE-4 equations.
        """
        X1, X2, X3, X4 = y

        # GLE-4 equations from the paper
        dX1 = X4 * X2 - X3 * X4 + self.beta * (X2 - X4) + self.gamma * (X2**2 - X4 * X1)
        dX2 = (
            self.rho * X1 * X3
            - X4 * X1
            + self.beta * (X3 - X1)
            + self.gamma * (X3**2 - X1 * X2)
        )
        dX3 = (
            X2 * X4
            - self.rho * X1 * X2
            + self.beta * (X4 - X2)
            + self.gamma * (X4**2 - X2 * X3)
        )
        dX4 = X3 * X1 - X2 * X3 + self.beta * (X1 - X3) + self.gamma * (X1**2 - X3 * X4)

        return jnp.array([dX1, dX2, dX3, dX4])

    def forward(
        self,
    ):
        pass
