import jax
import jax.numpy as jnp
from data_science_utils.filtering import etpf_update
from data_science_utils.measurement_systems import RangeSensor

key = jax.random.key(0)

etpf_update(
    key,
    jax.random.normal(key, shape=(2, 3)),
    jax.random.normal(key, shape=(1,)),
    RangeSensor(jnp.array([[1.0]])),
)
