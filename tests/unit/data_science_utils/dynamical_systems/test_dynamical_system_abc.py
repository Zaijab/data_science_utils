import equinox as eqx
import jax

from data_science_utils.dynamical_systems import Ikeda, Lorenz63, Lorenz96


def test_instantiation() -> None:
    systems = [Ikeda, Lorenz63, Lorenz96]

    for system_class in systems:
        system = system_class()


def test_initial_state() -> None:
    systems = [Ikeda, Lorenz63, Lorenz96]

    for system_class in systems:
        system = system_class()
        noiseless_state = system.initial_state()
        noisy_state = system.initial_state(jax.random.key(0))

        assert noiseless_state.ndim == 1
        assert noiseless_state.shape == (system.dimension,)

        assert noisy_state.ndim == 1
        assert noisy_state.shape == (system.dimension,)


def test_flow() -> None:
    systems = [Ikeda, Lorenz63, Lorenz96]

    for system_class in systems:
        system = system_class()
        initial_state = system.initial_state()
        initial_state = system.flow(0.0, 10.0, initial_state)


def test_generate() -> None:
    systems = [Ikeda, Lorenz63, Lorenz96]

    for system_class in systems:
        system = system_class()
        batch_size = 10
        attractors = system.generate(jax.random.key(0), batch_size=batch_size)
        assert attractors.ndim == 2
        assert attractors.shape == (batch_size, system.dimension)

        attractors = eqx.filter_vmap(system.flow)(0.0, 10.0, attractors)
        assert attractors.ndim == 2
        assert attractors.shape == (batch_size, system.dimension)
