import jax
from data_science_utils.dynamical_systems import Ikeda


def test_ikeda_generate() -> None:
    system = Ikeda()
    batch = system.generate(jax.random.key(0))
    assert batch.ndim == 2
    assert batch.shape[0] == system.batch_size
    assert batch.shape[1] == system.dimension


def test_ikeda_initial_state() -> None:
    system = Ikeda()
    assert system.initial_state().ndim == 1
    assert system.initial_state().shape == (2,)


def test_ikeda_forward() -> None:
    system = Ikeda()
    initial_state = system.initial_state()
    initial_state = system.forward(initial_state)
    assert system.initial_state().ndim == 1
    assert system.initial_state().shape == (2,)

    batch = system.generate(jax.random.key(0))
    batch = system.forward(batch)
    assert batch.ndim == 2
    assert batch.shape[0] == system.batch_size
    assert batch.shape[1] == system.dimension
