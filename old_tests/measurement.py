import jax
import jax.numpy as jnp
from data_science_utils.measurement_functions import norm_measurement

norm_measurement(jnp.array([[1.25, 0]]), jax.random.key(0), covariance=jnp.array([[1.0]]))
