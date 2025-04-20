import jax
import jax.numpy as jnp
from data_science_utils.dynamical_systems import Ikeda
from data_science_utils.measurement_functions import RangeSensor
from data_science_utils.filtering import etpf_update, evaluate_filter


key = jax.random.key(100)
key, subkey = jax.random.split(key)
dynamical_system = Ikeda(u=0.9)
measurement_system = RangeSensor(jnp.array([[0.25]]))
true_state = dynamical_system.initial_state
ensemble_size = 10
ensemble = jax.random.multivariate_normal(
    subkey,
    shape=(ensemble_size,),
    mean=true_state,
    cov=jnp.eye(2),
)

evaluate_filter(
    ensemble,
    dynamical_system,
    measurement_system,
    etpf_update,
    key,
)
