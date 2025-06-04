import equinox as eqx
import jax
import jax.numpy as jnp
from data_science_utils.dynamical_systems import Lorenz63
from diffrax import ConstantStepSize, ODETerm, SaveAt, Tsit5, diffeqsolve
from jaxtyping import Array, Float


def test_lorenz_63_integrate_single_trajectory() -> None:
    system = Lorenz63()
    x = system.initial_state()
    ts = jnp.linspace(0, 10, 1000)

    trajectory = system.trajectory(0.0, 10.0, x, SaveAt(ts=ts))
    orbit = system.orbit(0.0, 10.0, x, SaveAt(ts=ts))
    flow = system.flow(0.0, 10.0, x)

    assert ts.ndim == 1
    assert x.ndim == 1
    assert trajectory[0].shape == ts.shape
    assert trajectory[1].shape == (ts.shape[0], x.shape[0])
    assert orbit.shape == (ts.shape[0], x.shape[0])
    assert flow.shape == x.shape
