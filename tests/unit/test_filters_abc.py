import jax
import jax.numpy as jnp
from data_science_utils.dynamical_systems import Ikeda, Lorenz96
from data_science_utils.filters import EnKF, EnGMF, evaluate_filter
from data_science_utils.measurement_systems import RangeSensor


def test_instantiation() -> None:
    filter_classes = [EnKF]
    for filter_class in filter_classes:
        filter_class()


def test_update() -> None:

    key = jax.random.key(1010)
    key, subkey = jax.random.split(key)
    dynamical_system = Lorenz96()
    measurement_system = RangeSensor(jnp.array([[0.25]]))
    filter_update = EnGMF().update
    ensemble_size = 100

    key, subkey = jax.random.split(key)
    initial_ensemble = dynamical_system.generate(subkey, batch_size=ensemble_size)

    key, subkey = jax.random.split(key)
    rmse = evaluate_filter(
        initial_ensemble,
        dynamical_system,
        measurement_system,
        filter_update,
        subkey,
    )
    print(f"RMSE {rmse}")
