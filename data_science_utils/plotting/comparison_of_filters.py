from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_compilation_cache_dir", "/home/zjabbar/.cache/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

import matplotlib.pyplot as plt
from data_science_utils.dynamical_systems import flow, generate
from data_science_utils.filtering import (
    discriminator_ensemble_gaussian_mixture_filter_update_ensemble,
    ensemble_gaussian_mixture_filter_update_ensemble)
from data_science_utils.measurement_functions import Distance, norm_measurement
from jax import lax
from tqdm import tqdm

##################

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jax import lax
from jaxtyping import Array, Bool, Float, Key, jaxtyped
import equinox as eqx

from data_science_utils.dynamical_systems import ikeda_attractor_discriminator

def plot_against_ikeda(point, mean, cov):
    rng = jax.random.key(100)
    attractor = generate(rng)
    data = jax.random.multivariate_normal(key=rng, shape=(1000), mean=mean, cov=cov)
    plt.scatter(attractor[:, 0], attractor[:, 1], c='blue')
    plt.scatter(data[:, 0], data[:, 1], c='purple')
    plt.scatter(point[0], point[1], c='lime', s=100)
    plt.show()


@partial(jax.jit, static_argnames=["u", "ninverses"])
def ikeda_rejection_sample_single(
    rng: Key[Array, "1"],
    mean: Float[Array, "state_dim"],
    cov: Float[Array, "state_dim state_dim"],
    ninverses: int = 8,
    u: float = 0.9,
) -> Float[Array, "state_dim"]:
    """
    Samples from the multivariate normal defined by (mean, cov) until
    ikeda_attractor_discriminator(candidate) is True. Returns the first
    candidate that passes.
    """

    def cond_fun(carry):
        # carry = (rng, candidate, accepted_bool)
        return ~carry[2]

    def body_fun(carry):
        rng, sample, pass_sample, count = carry
        count += 1
        rng, subkey_candidate = jax.random.split(rng)
        sample = jax.random.multivariate_normal(subkey_candidate, mean=mean, cov=cov)
        pass_sample = ikeda_attractor_discriminator(sample, ninverses, u)

        #jax.debug.callback(debug_while_loop, carry, sample, mean, cov)

        return (rng, sample, pass_sample, count)

    # Initialize with zero vector and False acceptance
    carry_init = (rng, jnp.zeros_like(mean), False, 0)
    _, final_sample, _, _ = eqx.internal.while_loop(cond_fun, body_fun, carry_init, kind="checkpointed", max_steps=10**4)
    return final_sample

def debug_while_loop(carry, point, mean, cov):
    if carry[-1] > 1500 and (carry[-1] % 500) == 0:
        print(carry)
        plot_against_ikeda(point, mean, cov)


@partial(jax.jit, static_argnames=["ninverses", "u"])
def ikeda_rejection_sample_batch(
    rng: Key[Array, "batch_dim"],
    means: Float[Array, "batch state_dim"],
    covs: Float[Array, "batch state_dim state_dim"],
    ninverses: int = 8,
    u: float = 0.9
) -> Float[Array, "batch state_dim"]:
    """
    For each row in 'means' and 'covs', run rejection_sample_single.
    """
    return jax.vmap(ikeda_rejection_sample_single, in_axes=(0, 0, 0, None, None))(rng, means, covs, ninverses, u)


##############
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped

import equinox as eqx





##################

# TESTING
def is_positive_definite(M):
    try:
        jnp.linalg.cholesky(M)
        return True
    except:
        assert False, "COVARIANCE MATRIX IS NOT POSITIVE DEFINITE"

def has_nan(M):
    assert not jnp.any(jnp.isnan(M))




@partial(jax.jit, static_argnames=["ensemble_size"])
def filter_experiment(
    ensemble_size,
    ensemble_initialization_key,
    measurement_covariance,
    bandwidth,
    sampling_function,
):
    """
    Same signature as your original function, but uses lax.scan instead
    of a Python for-loop. Returns the RMSE over the measurement phase.

    NOTE:
      - 'bandwidth' can be either a scalar or a callable that takes
        ensemble_size and returns a scalar.
      - The 'sampling_function' is presumably a partial of either
        sample_multivariate_normal or rejection_sample_batch, etc.
      - For simplicity, burn_in_time and measurement_time are hard-coded
        internally. Adjust as needed or move them into parameters.
    """
    burn_in_time = 100
    measurement_time = 10 * burn_in_time
    total_steps = burn_in_time + measurement_time

    bandwidth_factor = bandwidth(ensemble_size)

    measurement_device = jax.tree_util.Partial(norm_measurement, covariance=measurement_covariance)

    # Initialize the true state
    true_state_init = jnp.array([[1.25, 0.0]])

    # Initialize the ensemble
    key = ensemble_initialization_key
    key, subkey = jax.random.split(key)
    # filter_ensemble_init = jax.random.multivariate_normal(
    #     subkey,
    #     shape=(ensemble_size,),
    #     mean=true_state_init,
    #     cov=(1 / 4) * jnp.eye(2),
    # )

    filter_ensemble_init = generate(subkey, batch_size=ensemble_size)

    # Build the partial for the filter update
    filter_update = jax.tree_util.Partial(
        ensemble_gaussian_mixture_filter_update_ensemble,  # or if you are storing your filter in sampling_function
        bandwidth_factor=bandwidth_factor,
        measurement_device=measurement_device,
        measurement_device_covariance=measurement_covariance,
        sampling_function=sampling_function,  # The same function that draws the new samples
    )

    #-------------------------------------------
    # single iteration of the filter
    #-------------------------------------------
    def scan_step(carry, step_idx):
        """
        carry = (key, ensemble, true_state)
        step_idx is [0.. total_steps-1]
        """
        (key, ensemble, true_state) = carry
        new_key, subkey = jax.random.split(key)

        # Filter update
        updated_ensemble = filter_update(
            state=ensemble,
            key=subkey,
            measurement=measurement_device(state=true_state),
            debug=debug,  # or True if you want
        )

        # Compute error only if we're past burn-in
        is_measurement_phase = (step_idx >= burn_in_time)
        error = (true_state - jnp.mean(updated_ensemble, axis=0)) * is_measurement_phase

        # Flow forward
        ensemble_next = flow(updated_ensemble)
        true_state_next = flow(true_state)

        new_carry = (new_key, ensemble_next, true_state_next)
        return new_carry, error

    #-------------------------------------------
    # run lax.scan
    #-------------------------------------------
    carry_init = (key, filter_ensemble_init, true_state_init)
    steps_array = jnp.arange(total_steps)  # shape (total_steps,)

    (final_carry, errors_over_time) = lax.scan(scan_step, carry_init, steps_array)
    # errors_over_time has shape (total_steps, 2)

    # We only recorded nonzero error in the measurement phase, so we can just compute:
    rmse = jnp.sqrt(jnp.mean(errors_over_time**2))

    return rmse


debug = False
ensemble_sizes = range(3, 21, 1)

key = jax.random.key(42)
key, *ensemble_keys = jax.random.split(key, 1 + 32)

measurement_covariances = [
   jnp.array([[0.5 ** 2]]),
   jnp.array([[1.0 ** 2]]),
   jnp.array([[2.0 ** 2]]),
]


def silverman_bandwidth(ensemble_size, scale=1):
    return scale * ((4 / (ensemble_size * (2 + 2))) ** (2 / (2 + 4)))


bandwidths = [
   jax.tree_util.Partial(silverman_bandwidth, scale=1/3),
   jax.tree_util.Partial(silverman_bandwidth, scale=2/3),
   jax.tree_util.Partial(silverman_bandwidth, scale=3/3),
]


@jax.jit
@jax.vmap
def sample_gaussian_mixture(key, point, cov):
    return jax.random.multivariate_normal(key, mean=point, cov=cov)


sampling_functions = [
    jax.tree_util.Partial(ikeda_rejection_sample_batch),
    jax.tree_util.Partial(sample_gaussian_mixture),
]

print("WHAT")

records = []

for ensemble_size in tqdm(ensemble_sizes, leave=False):
    for ensemble_key in tqdm(ensemble_keys):
        for covariance in measurement_covariances:
            for bandwidth in bandwidths:
                for sampling_function in sampling_functions:
                    rmse = filter_experiment(
                            ensemble_size,
                            ensemble_key,
                            covariance,
                            bandwidth,
                            sampling_function,
                    )
                    print(rmse)
                    records.append({
                        "ensemble_size": ensemble_size,
                        "random": str(jax.random.key_data(ensemble_key)),
                        "covariance": covariance.item(),
                        "bandwidth": bandwidth.keywords['scale'],
                        "sampling": sampling_function.func.__name__,
                        "RMSE": rmse
                    })

import pandas as pd
df = pd.DataFrame(records)
