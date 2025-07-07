import jax.numpy as jnp

from data_science_utils.measurement_systems.angles_only import AnglesOnly


def test_angles_only_random_data() -> None:
    measurement_system = AnglesOnly()
    measurement_system.covariance
    measurement_system(jnp.array([1,2,3,4,5,6]))
