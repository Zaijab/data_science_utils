import jax.numpy as jnp

from data_science_utils.measurement_systems import RangeSensor


def test_instantiation() -> None:
    measurement_system_classes = [RangeSensor]
    for measurement_system_class in measurement_system_classes:
        measurement_system_class(jnp.array([[1 / 16]]))


def test_measurement() -> None:
    measurement_system_classes = [RangeSensor]
    for measurement_system_class in measurement_system_classes:
        measurement_system = measurement_system_class(jnp.array([[1 / 16]]))
        measurement_system(jnp.array([1.25, 0]))
