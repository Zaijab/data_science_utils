import equinox as eqx
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

def test_constant_velocity_engmphd_paper() -> None:
    system = system = CVModel(
        position_dimension=3, sampling_period=1.0, process_noise_std=5.0, ordering="durant"
    )
    true_state = jnp.array(
        [[50, 50, 50, 0.5, 0.5, 2], [100, 100, 50, -0.5, -0.5, 2]], dtype=jnp.float64
    )
    next_state = eqx.filter_vmap(system.flow)(0.0, 1.0, true_state)

    assert jnp.allclose(next_state[0, :], jnp.array([50.5, 50.5, 52, 0.5, 0.5, 2]))
    assert jnp.allclose(next_state[1, :], jnp.array([99.5, 99.5, 52, -0.5, -0.5, 2]))
