from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped
import equinox as eqx
from data_science_utils.dynamical_systems import AbstractDynamicalSystem


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def lorenz96_derivatives(
    x: Float[Array, "*batch dim"], F: float = 8.0
) -> Float[Array, "*batch dim"]:
    """Compute the derivatives for the Lorenz 96 model."""
    # Handle rolling for periodic boundary conditions
    x_plus_1 = jnp.roll(x, shift=-1, axis=-1)  # X_{i+1}
    x_minus_1 = jnp.roll(x, shift=1, axis=-1)  # X_{i-1}
    x_minus_2 = jnp.roll(x, shift=2, axis=-1)  # X_{i-2}

    # Compute derivatives according to Lorenz 96 formula
    derivatives = (x_plus_1 - x_minus_2) * x_minus_1 - x + F

    return derivatives


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def lorenz96_forward(
    x: Float[Array, "*batch dim"],
    F: float = 8.0,
    dt: float = 0.01,
    steps: int = 12,  # 12 steps of 0.01 = 0.12 time units
) -> Float[Array, "*batch dim"]:
    """Advance the Lorenz 96 system forward by integrating for specified time units using RK4."""

    def rk4_step(state, _):
        # RK4 integration for one step
        k1 = lorenz96_derivatives(state, F)
        k2 = lorenz96_derivatives(state + dt / 2 * k1, F)
        k3 = lorenz96_derivatives(state + dt / 2 * k2, F)
        k4 = lorenz96_derivatives(state + dt * k3, F)

        # Update state
        new_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_state, None

    # Perform multiple steps using scan
    final_state, _ = jax.lax.scan(rk4_step, x, None, length=steps)
    return final_state


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def lorenz96_generate(
    key: Key[Array, "..."],
    batch_size: int = 1000,
    dim: int = 40,
    F: float = 8.0,
    spin_up_steps: int = 100,
    dt: float = 0.01,
    steps: int = 12,
) -> Float[Array, "{batch_size} {dim}"]:
    initial_states = F + 0.01 * jax.random.normal(key, shape=(batch_size, dim))

    # Flow towards attractor
    def body_fn(i, val):
        return lorenz96_forward(val, F, dt, steps)

    # Use fori_loop for spin-up steps
    final_states = jax.lax.fori_loop(0, spin_up_steps, body_fn, initial_states)

    return final_states


class Lorenz96(AbstractDynamicalSystem, strict=True):
    F: float = 8.0
    dt: float = 0.05
    steps: int = 12
    dim: int = 40
    batch_size: int = 1000
    spin_up_steps: int = 100

    @property
    def dimension(self):
        return self.dim

    @property
    def initial_state(self):
        # Standard initial state with a small perturbation on one component
        state = jnp.ones(self.dim)  # * self.F
        state = state.at[0].set(state[0] + 0.01)
        return state

    def forward(
        self,
        x: Float[Array, "*batch {self.dim}"],
    ) -> Float[Array, "*batch {self.dim}"]:
        return lorenz96_forward(x, F=self.F, dt=self.dt, steps=self.steps)

    def generate(
        self,
        key: Key[Array, "..."],
        batch_size: int = None,
    ) -> Float[Array, "{batch_size} {self.dim}"]:
        actual_batch_size = self.batch_size if batch_size is None else batch_size
        return lorenz96_generate(
            key,
            batch_size=actual_batch_size,
            dim=self.dim,
            F=self.F,
            spin_up_steps=self.spin_up_steps,
            dt=self.dt,
            steps=self.steps,
        )
