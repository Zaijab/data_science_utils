

import jax
import jax.numpy as jnp

from data_science_utils.dynamical_systems import Ikeda
from data_science_utils.filters import EnGMF, evaluate_filter
from data_science_utils.measurement_systems import RangeSensor


def test_engmf() -> None:
    key = jax.random.key(100)
    key, subkey = jax.random.split(key)
    dynamical_system = Ikeda(u=0.9)
    measurement_system = RangeSensor(jnp.array([[0.25]]))
    stochastic_filter = EnGMF()
    true_state = dynamical_system.initial_state()
    ensemble_size = 100
    ensemble = jax.random.multivariate_normal(
        subkey,
        shape=(ensemble_size,),
        mean=true_state,
        cov=jnp.eye(2),
    )

    return evaluate_filter(
        ensemble,
        dynamical_system,
        measurement_system,
        stochastic_filter.update,
        key,
    )

test_engmf()
