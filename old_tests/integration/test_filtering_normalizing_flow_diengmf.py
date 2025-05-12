import jax
import jax.numpy as jnp
from data_science_utils.dynamical_systems import Ikeda
from data_science_utils.filtering import evaluate_filter
from data_science_utils.measurement_functions import RangeSensor

key = jax.random.key(1010)
key, subkey = jax.random.split(key)
dynamical_system = Ikeda(u=0.9)
measurement_system = RangeSensor(jnp.array([[0.25]]))
true_state = dynamical_system.initial_state
ensemble_size = jnp.arange(10, 120, 10)

rmses = []
for size in ensemble_size:
    key, subkey = jax.random.split(key)
    initial_ensemble = jax.random.multivariate_normal(
        subkey,
        shape=(size,),
        mean=true_state,
        cov=jnp.eye(2),
    )
    key, subkey = jax.random.split(key)
    rmse = evaluate_filter(
        initial_ensemble,
        dynamical_system,
        measurement_system,
        ,
        key,
    )
    rmses.append(rmse)

    print(f"{size} {rmse}")
