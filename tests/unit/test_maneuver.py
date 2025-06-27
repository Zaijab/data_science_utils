import uuid

import jax
import jax.numpy as jnp
from diffrax import ConstantStepSize, Tsit5

from data_science_utils.dynamical_systems.crtbp import CR3BP
from data_science_utils.filters import EnGMF, EnKF, evaluate_filter
from data_science_utils.filters.evaluate_imperfect_model import \
    evaluate_filter_imperfect
from data_science_utils.measurement_systems import Radar, RangeSensor
from data_science_utils.measurement_systems.angles_only import AnglesOnly

key = jax.random.key(1001)
key, subkey = jax.random.split(key)

dynamical_system = CR3BP()
measurement_system = RangeSensor()
measurement_system = AnglesOnly()
measurement_system = Radar()

print(uuid.uuid4())

for _ in range(10):
    stochastic_filter = EnGMF()
    key, subkey = jax.random.split(key)
    rmse_no_maneuver = evaluate_filter(
        dynamical_system,
        measurement_system,
        stochastic_filter,
        subkey,
    )
    key, subkey = jax.random.split(key)
    rmse_with_maneuver = evaluate_filter_imperfect(
        dynamical_system,
        measurement_system,
        stochastic_filter,
        subkey,
    )

    print("EnGMF:")
    print(f"\t{rmse_no_maneuver=}")
    print(f"\t{rmse_with_maneuver=}")


    stochastic_filter = EnKF()
    key, subkey = jax.random.split(key)
    rmse_no_maneuver = evaluate_filter(
        dynamical_system,
        measurement_system,
        stochastic_filter,
        subkey,
    )
    key, subkey = jax.random.split(key)
    rmse_with_maneuver = evaluate_filter_imperfect(
        dynamical_system,
        measurement_system,
        stochastic_filter,
        subkey,
    )

    print("EnKF:")
    print(f"\t{rmse_no_maneuver=}")
    print(f"\t{rmse_with_maneuver=}")
    break
