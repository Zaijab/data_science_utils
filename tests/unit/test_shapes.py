import jax
import jax.numpy as jnp
import equinox as eqx
from flax import nnx

def initialize_ensemble(key, batch_size):
    return jax.random.normal(key, (batch_size,))

def create_variable_measurement(key):
    n = jax.random.poisson(key, lam=10.0)
    n = jnp.minimum(n, 40)
    x = jax.random.normal(key, shape=(40,))
    return jax.lax.slice(x, (0,), (n,)) 

# @nnx.jit
@eqx.filter_jit
def create_ensemble_process_measurement(key):
    ensemble = initialize_ensemble(key, 10)
    measurements = create_variable_measurement(key)
    return measurements

print(create_ensemble_process_measurement(jax.random.key(0)))
