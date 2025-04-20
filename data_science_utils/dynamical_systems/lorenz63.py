import jax
import jax.numpy as jnp
import equinox as eqx

from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax import lax, random
from jaxtyping import Array, Bool, Float, Key, jaxtyped
import equinox as eqx


class Lorenz63(eqx.Module):
    dt: float 0.01

    @property
    def dimension(self):
        return 3

    @property
    def initial_state(self):
        return jnp.array([8, 0, 0])

    def forward(
        self,
        x: Float[Array, "*batch 2"],
    ) -> Float[Array, "*batch 2"]:
        return ikeda_forward(x, u=self.u)
