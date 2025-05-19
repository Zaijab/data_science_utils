from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from diffrax import ConstantStepSize, ODETerm, SaveAt, Tsit5, diffeqsolve
from data_science_utils.dynamical_systems import AbstractDynamicalSystem
from jax import lax, random
from jaxtyping import Array, Bool, Float, Key, jaxtyped


class Lorenz63(AbstractDynamicalSystem, strict=True):
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01

    @property
    def dimension(self):
        return 3

    def initial_state(
        self,
        key: Key[Array, ...] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        return jnp.array([8.0, 0.0, 0.0])

    def vector_field(self, t, y, args):
        x, y, z = y
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return jnp.array([dx, dy, dz])

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def forward(
        self,
        x: Float[Array, "3"],
        t1: float = 1.0,
        saveat: SaveAt = SaveAt(t1=True),
    ) -> Float[Array, "3"]:
        """Integrate a single point forward in time."""
        term = ODETerm(self.vector_field)
        solver = Tsit5()

        sol = diffeqsolve(
            term,
            solver,
            t0=0,
            t1=t1,
            dt0=self.dt,
            y0=x,
            stepsize_controller=ConstantStepSize(),
            saveat=saveat,
            max_steps=100_000,
        )
        return sol.ys

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def generate(
        self, key: Key[Array, "..."], batch_size: int = 1000, spin_up_steps: int = 100
    ) -> Float[Array, "{batch_size} 3"]:

        initial_states = jnp.ones((batch_size, 3)) * jnp.array([8.0, 1.0, 1.0])
        initial_states = initial_states + 5 * random.normal(key, shape=(batch_size, 3))
        final_states = eqx.filter_vmap(self.forward)(initial_states, spin_up_steps)

        return final_states
