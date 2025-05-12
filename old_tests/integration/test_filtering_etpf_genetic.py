from typing import Any, Callable, Protocol, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from beartype import beartype as typechecker
from data_science_utils.dynamical_systems import Ikeda
from data_science_utils.filtering import etpf_update, evaluate_filter
from data_science_utils.measurement_functions import RangeSensor
from jaxtyping import Array, Float, Key, jaxtyped
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

key = jax.random.key(1010)
key, subkey = jax.random.split(key)
dynamical_system = Ikeda(u=0.9)
measurement_system = RangeSensor(jnp.array([[0.25]]))
true_state = dynamical_system.initial_state
ensemble_size = jnp.arange(10, 120, 10)

# solver = jax.tree_util.Partial(solve_ot_problem_correct)

etpf_rmses = []
for size in ensemble_size:
    key, subkey = jax.random.split(key)
    initial_ensemble = jax.random.multivariate_normal(
        subkey,
        shape=(size,),
        mean=true_state,
        cov=jnp.eye(2),
    )
    key, subkey = jax.random.split(key)
    etpf_rmse = evaluate_filter(
        initial_ensemble,
        dynamical_system,
        measurement_system,
        etpf_update,
        key,
    )
    etpf_rmses.append(etpf_rmse)

    # key, subkey = jax.random.split(key)
    # enkf_rmse = evaluate_filter(
    #     initial_ensemble,
    #     dynamical_system,
    #     measurement_system,
    #     enkf_update,
    #     key,
    # )
    # enkf_rmses.append(enkf_rmse)

    print(f"{size} {etpf_rmse}")
