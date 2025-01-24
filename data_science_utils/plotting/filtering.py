from functools import partial

import jax
import jax.numpy as jnp
from data_science_utils.dynamical_systems import IkedaSystem, flow, generate
from data_science_utils.filtering import (
    discriminator_ensemble_gaussian_mixture_filter_update_ensemble,
    ensemble_gaussian_mixture_filter_update_ensemble)
from data_science_utils.measurement_functions import Distance
from jax import lax
from tqdm import tqdm

# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_compilation_cache_dir", "/home/zjabbar/.cache/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
jax.config.update("jax_disable_jit", False)

plot = False
key = jax.random.key(100)
key, subkey = jax.random.split(key)
measurement_device = Distance(jnp.array([[1.0]]))
true_state = jnp.array([[1.25, 0.0]])

if plot:
    import matplotlib.pyplot as plt
    key, subkey = jax.random.split(key)
    attractor = generate(subkey)

ensemble_size = 10
silverman_bandwidth = (4 / (ensemble_size * (2 + 2))) ** (2 / (2 + 4))

key, subkey = jax.random.split(key)
filter_ensemble = jax.random.multivariate_normal(key=subkey, shape=(ensemble_size,), mean=true_state, cov=jnp.eye(2))

filter_update = partial(
    ensemble_gaussian_mixture_filter_update_ensemble,
    bandwidth_factor=silverman_bandwidth,
    measurement_device=measurement_device,
)

# filter_update = partial(
#     discriminator_ensemble_gaussian_mixture_filter_update_ensemble,
#     bandwidth_factor=silverman_bandwidth,
#     measurement_device=measurement_device,
#     ninverses=8,
# )


burn_in_time = 500
measurement_time = 10 * burn_in_time
debug = False
covariances, states = [], []

for t in tqdm(range(burn_in_time + measurement_time), leave=False):

    key, subkey = jax.random.split(key)

    filter_ensemble = filter_update(
        state=filter_ensemble,
        key=subkey,
        measurement=measurement_device(true_state),
        debug=debug
    )

    if plot:
        plt.scatter(attractor[:, 0], attractor[:, 1], c='blue', alpha=0.1, s=0.1)
        plt.scatter(prior_ensemble[:, 0], prior_ensemble[:, 1], alpha=0.8, s=10, c='purple', label='Prior')
        plt.scatter(filter_ensemble[:, 0], filter_ensemble[:, 1], alpha=0.8, s=10, c='yellow', label='Posterior')
        plt.scatter(true_state[0, 0], true_state[0, 1], c='lime', s=100, label='True')
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
        plt.show()

    if t >= burn_in_time:
        if plot:
            break

        states.append(true_state - jnp.mean(filter_ensemble, axis=0))
        cov = jnp.cov(filter_ensemble.T)

        if debug:
            try:
                jnp.linalg.cholesky(cov)
            except e:
                assert False, "COVARIANCE MATRIX IS NOT POSITIVE DEFINITE"

        covariances.append(cov)

    filter_ensemble = flow(filter_ensemble)
    true_state = flow(true_state)


if len(states) != 0:
    e = jnp.expand_dims(jnp.array(states), -1)
    if debug:
        assert isinstance(e, Float[Array, f"{measurement_time} 1 2 1"])
    P = jnp.expand_dims(jnp.array(covariances), 1)
    if debug:
        assert isinstance(P, Float[Array, f"{measurement_time} 1 2 2"])

    rmse = jnp.sqrt((1 / (e.shape[0] * e.shape[1] * e.shape[2] * e.shape[3])) * jnp.sum(e * e, axis=(0, 1, 2, 3)))
    snees = (1 / e.size) * jnp.sum(jnp.swapaxes(e, -2, -1) @ jnp.linalg.inv(P) @ e)



    print(f'Batch {ensemble_size} / Time {measurement_time}')
    print(f"RMSE: {rmse}")
    print(f"SNEES: {snees}")

