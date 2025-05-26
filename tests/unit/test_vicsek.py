import jax

from data_science_utils.dynamical_systems import Vicsek


def test_vicsek_forward() -> None:
    key = jax.random.key(0)
    system = Vicsek(n_particles=200)

    key, subkey = jax.random.split(key)
    true_state = system.initial_state(subkey)

    key, subkey = jax.random.split(key)
    true_state = system.forward(true_state, subkey)
    true_state = system.flow(jax.random.key(10), 0.0, 10.0, true_state)

    assert true_state.shape == (200, 3)
