import jax.numpy as jnp

from data_science_utils.filters import EnKF


def test_instantiation() -> None:
    filter_classes = [EnKF]
    for filter_class in filter_classes:
        filter_class()
