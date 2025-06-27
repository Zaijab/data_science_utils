

import jax
import jax.numpy as jnp
from diffrax import ConstantStepSize, Tsit5

from data_science_utils.dynamical_systems import Ikeda
from data_science_utils.dynamical_systems.crtbp import CR3BP
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

def test_engmf_crtbp() -> None:

    key = jax.random.key(1010)
    key, subkey = jax.random.split(key)


    dynamical_system = CR3BP(solver=Tsit5(), stepsize_contoller=ConstantStepSize())
    measurement_system = RangeSensor()
    stochastic_filter = EnGMF()

    key, subkey = jax.random.split(key)
    rmse = evaluate_filter(
        dynamical_system,
        measurement_system,
        stochastic_filter,
        subkey,
    )
    print(f"RMSE {rmse}")


test_engmf()
