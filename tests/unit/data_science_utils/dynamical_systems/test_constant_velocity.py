import jax.numpy as jnp
from diffrax import SaveAt

from data_science_utils.dynamical_systems.constant_velocity import CVModel


def test_constant_velocity_vo() -> None:
    system = CVModel(3, 1.0, 5.0, "vo")
    state = system.initial_state()
    state = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0, -5.0])
    system.flow(0.0, 10.0, state)
    system.orbit(0.0, 10.0, state, SaveAt(t0=True, t1=True))
    system.trajectory(0.0, 10.0, state, SaveAt(t0=True, t1=True))


def test_constant_velocity_durant() -> None:
    system = CVModel(3, 1.0, 5.0, "durant")
    state = system.initial_state()
    state = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0, -5.0])
    system.flow(0.0, 10.0, state)
    system.orbit(0.0, 10.0, state, SaveAt(t0=True, t1=True))
    system.trajectory(0.0, 10.0, state, SaveAt(t0=True, t1=True))
