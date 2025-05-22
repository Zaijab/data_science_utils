from data_science_utils.dynamical_systems import Ikeda, Lorenz63, Lorenz96


def test_instantiation() -> None:
    systems = [Ikeda, Lorenz63, Lorenz96]

    for system_class in systems:
        system = system_class()


def test_flow() -> None:
    systems = [Ikeda, Lorenz63, Lorenz96]

    for system_class in systems:
        system = system_class()
        initial_state = system.initial_state()
        initial_state = system.flow(0.0, 10.0, initial_state)
