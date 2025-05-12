import jax
import jax.numpy as jnp
from jaxtyping import jaxtyped, Float, Array, Key
from beartype import beartype as typechecker
import equinox as eqx

from typing import Tuple, Callable, Protocol
import jax
import jax.numpy as jnp
from ott.solvers.linear import sinkhorn
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
import equinox as eqx
from jaxtyping import jaxtyped, Float, Array, Key
from beartype import beartype as typechecker

import matplotlib.pyplot as plt

from data_science_utils.dynamical_systems import Ikeda
from data_science_utils.measurement_functions import RangeSensor
from data_science_utils.filtering import etpf_update, enkf_update, evaluate_filter

import uuid

print(uuid.uuid4())


key = jax.random.key(106)

dynamical_system = Ikeda(u=0.9)
measurement_system = RangeSensor(jnp.array([[0.25]]))
true_state = dynamical_system.initial_state

etpf_rmses, enkf_rmses = [], []
for ensemble_size in range(10, 200, 10):
    key, subkey = jax.random.split(key)
    print(ensemble_size)
    ensemble = jax.random.multivariate_normal(
        subkey,
        shape=(ensemble_size,),
        mean=true_state,
        cov=jnp.eye(2),
    )

    etpf_rmse = evaluate_filter(
        ensemble,
        dynamical_system,
        measurement_system,
        etpf_update,
        key,
        debug=False,
    )
    etpf_rmses.append(etpf_rmse)

    enkf_rmse = evaluate_filter(
        ensemble,
        dynamical_system,
        measurement_system,
        enkf_update,
        key,
    )
    enkf_rmses.append(enkf_rmse)


print(f"ETPF: {etpf_rmse} | EnKF: {enkf_rmse}")
