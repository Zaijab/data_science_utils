import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial
import itertools

from jaxtyping import Array, Float, Key, jaxtyped
from beartype import beartype as typechecker

from data_science_utils.filtering import ensemble_gaussian_mixture_filter_update_ensemble
from data_science_utils.dynamical_systems import ikeda_attractor_discriminator, flow, generate

import equinox as eqx

# ------------------------------------------------------------------
# 0. Utils [NEED TO REFACTOR BADLY]

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


# -------------------------------------------------------------------
# 1. EnGMF Update
@partial(jax.jit, static_argnames=["debug"])
def engmf_update(key, ensemble, measurement, measurement_device_covariance, measurement_device, bandwidth, debug=False):
    """
    EnGMF update using a Gaussian mixture update.
    
    Parameters:
      key: PRNG key.
      ensemble: Array of shape (batch, state_dim).
      measurement: Array of shape (measurement_dim,).
      bandwidth: Scalar bandwidth factor.
      measurement_covariance: Measurement covariance (e.g. shape (m, m)).
      sampling_function: Function to sample from the Gaussian mixture.
      debug: Optional debug flag.
      
    Returns:
      updated_ensemble: Array of shape (batch, state_dim).
      
    Note: The function splits the key internally. The caller must split the key
          externally between calls to avoid reusing keys.
    """
    subkey, _ = jax.random.split(key)
    bandwidth_factor = bandwidth(ensemble.shape[0])
    updated_ensemble = ensemble_gaussian_mixture_filter_update_ensemble(
        state=ensemble,
        bandwidth_factor=bandwidth_factor,
        key=subkey,
        measurement=measurement,
        measurement_device_covariance=measurement_device_covariance,
        measurement_device=measurement_device,
        sampling_function=jax.tree_util.Partial(sample_gaussian_mixture),
        debug=debug
    )
    return updated_ensemble

# -------------------------------------------------------------------
# 2. DI-EnGMF Update
@partial(jax.jit, static_argnames=["debug"])
def di_engmf_update(key, ensemble, measurement, measurement_device_covariance, measurement_device, bandwidth, debug=False):
    """
    Discriminator-Informed EnGMF update using a rejection sampler.
    
    Parameters:
      key: PRNG key.
      ensemble: Array of shape (batch, state_dim).
      measurement: Array of shape (measurement_dim,).
      bandwidth: Scalar bandwidth factor.
      measurement_covariance: Measurement covariance.
      debug: Optional debug flag.
      
    Returns:
      updated_ensemble: Array of shape (batch, state_dim).
      
    Note: Uses ikeda_rejection_sample_batch as the sampling_function.
    """
    subkey, _ = jax.random.split(key)
    bandwidth_factor = bandwidth(ensemble.shape[0])
    updated_ensemble = ensemble_gaussian_mixture_filter_update_ensemble(
        state=ensemble,
        bandwidth_factor=bandwidth_factor,
        key=subkey,
        measurement=measurement,
        measurement_device_covariance=measurement_device_covariance,
        measurement_device=measurement_device,
        sampling_function=jax.tree_util.Partial(ikeda_rejection_sample_batch),
        debug=debug
    )
    return updated_ensemble

# -------------------------------------------------------------------
# 3. EnKF Update
@partial(jax.jit, static_argnames=["debug"])
def enkf_update(key, ensemble, measurement, measurement_device_covariance, measurement_device, inflation_factor, debug=False):
    """
    Standard EnKF update with inflation.
    
    Parameters:
      key: PRNG key (included for interface consistency; not used here for randomness).
      ensemble: Array of shape (batch, state_dim).
      measurement: Array of shape (measurement_dim,).
      inflation_factor: Scalar inflation factor.
      measurement_covariance: Measurement covariance.
      debug: Optional debug flag.
      
    Returns:
      updated_ensemble: Array of shape (batch, state_dim).
    """
    # Compute ensemble mean and inflate the ensemble.
    mean = jnp.mean(ensemble, axis=0)

    if debug:
        jax.debug.print('{shape}', mean.shape)

    inflated = mean + inflation_factor * (ensemble - mean)
    # Compute the global covariance with a small regularization.
    ensemble_covariance = jnp.cov(inflated.T) + 1e-8 * jnp.eye(inflated.shape[-1])

    @jit
    def update_ensemble_point(point, key):
        point_measurement = measurement_device(point, key)
        measurement_jacobian = jax.jacfwd(measurement_device)(point)
        innovation_covariance = measurement_jacobian @ ensemble_covariance @ measurement_jacobian.T + measurement_device_covariance
        kalman_gain = ensemble_covariance @ measurement_jacobian.T @ jnp.linalg.inv(innovation_covariance)
        point = point + (kalman_gain @ jnp.atleast_2d(measurement - point_measurement)).reshape(-1)
        return point

    keys = jax.random.split(key, ensemble.shape[0])
    updated_ensemble = vmap(update_ensemble_point)(inflated, keys)
    return updated_ensemble

# -------------------------------------------------------------------
# 4. BRUEnKF Update
@partial(jax.jit, static_argnames=["debug"])
def bruenkf_update(key, ensemble, measurement, measurement_device_covariance, measurement_device, inflation_factor, num_bruf_steps, debug=False):
    """
    Standard EnKF update with inflation.
    
    Parameters:
      key: PRNG key (included for interface consistency; not used here for randomness).
      ensemble: Array of shape (batch, state_dim).
      measurement: Array of shape (measurement_dim,).
      inflation_factor: Scalar inflation factor.
      measurement_covariance: Measurement covariance.
      debug: Optional debug flag.
      
    Returns:
      updated_ensemble: Array of shape (batch, state_dim).
    """

    def bruf_update(_, ensemble):
        mean = jnp.mean(ensemble, axis=0)
        inflated = mean + inflation_factor * (ensemble - mean)
        ensemble_covariance = jnp.cov(inflated.T) + 1e-8 * jnp.eye(inflated.shape[-1])

        @jit
        def update_ensemble_point(point, key):
            point_measurement = measurement_device(point, key)
            measurement_jacobian = jax.jacfwd(measurement_device)(point)
            innovation_covariance = measurement_jacobian @ ensemble_covariance @ measurement_jacobian.T + measurement_device_covariance
            kalman_gain = ensemble_covariance @ measurement_jacobian.T @ jnp.linalg.inv(innovation_covariance)
            point = point + (kalman_gain @ jnp.atleast_2d(measurement - point_measurement)).reshape(-1)
            return point

        keys = jax.random.split(key, ensemble.shape[0])
        updated_ensemble = vmap(update_ensemble_point)(inflated, keys)
        return updated_ensemble

    updated_ensemble = jax.lax.fori_loop(0, num_bruf_steps, bruf_update, ensemble)
    return updated_ensemble

@partial(jax.jit, static_argnames=["ensemble_size", "debug"])
def filter_experiment(
        ensemble_update_method,
        ensemble_size,
        ensemble_initialization_key,
        measurement_device_covariance,
        debug=False,
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

    # Initialize the true state
    true_state_init = jnp.array([[1.25, 0.0]])
    measurement_device = jax.tree_util.Partial(norm_measurement, covariance=measurement_covariance)

    # Initialize the ensemble
    key = ensemble_initialization_key
    key, subkey = jax.random.split(key)
    filter_ensemble_init = jax.random.multivariate_normal(
        subkey,
        shape=(ensemble_size,),
        mean=true_state_init,
        cov=(1 / 4) * jnp.eye(2),
    )

    #filter_ensemble_init = generate(subkey, batch_size=ensemble_size)

    # Build the partial for the filter update
    filter_update = jax.tree_util.Partial(
        ensemble_update_method,
        measurement_device=measurement_device,
        measurement_device_covariance=measurement_device_covariance,
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
            ensemble=ensemble,
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


# Initialize the PRNG key.
key = jax.random.key(100)
key, subkey = jax.random.split(key)
true_state = jnp.array([[1.25, 0.0]])
ensemble_size = 10
ensemble = jax.random.multivariate_normal(
    subkey,
    shape=(ensemble_size,),
    mean=true_state,
    cov=(1 / 4) * jnp.eye(2),
)

from data_science_utils.measurement_functions import norm_measurement

measurement_covariance = jnp.array([[0.25]])
measurement_device = jax.tree_util.Partial(norm_measurement, covariance=measurement_covariance)

key, subkey = jax.random.split(key)
measurement = measurement_device(state=true_state, key=jax.random.key(100))


@jax.jit
@jax.vmap
def sample_gaussian_mixture(key, point, cov):
    return jax.random.multivariate_normal(key, mean=point, cov=cov)


def silverman_bandwidth(ensemble_size, scale=1):
    return scale * ((4 / (ensemble_size * (2 + 2))) ** (2 / (2 + 4)))


ensemble_update_methods = [
    # jax.tree_util.Partial(engmf_update, bandwidth=jax.tree_util.Partial(silverman_bandwidth, scale=1 / 3)),
    # jax.tree_util.Partial(engmf_update, bandwidth=jax.tree_util.Partial(silverman_bandwidth, scale=2 / 3)),
    # jax.tree_util.Partial(engmf_update, bandwidth=jax.tree_util.Partial(silverman_bandwidth, scale=3 / 3)),
    # jax.tree_util.Partial(di_engmf_update, bandwidth=jax.tree_util.Partial(silverman_bandwidth, scale=1 / 3)),
    # jax.tree_util.Partial(di_engmf_update, bandwidth=jax.tree_util.Partial(silverman_bandwidth, scale=2 / 3)),
    # jax.tree_util.Partial(di_engmf_update, bandwidth=jax.tree_util.Partial(silverman_bandwidth, scale=3 / 3)),
    jax.tree_util.Partial(enkf_update, inflation_factor=1.01),
    jax.tree_util.Partial(bruenkf_update, inflation_factor=1.01, num_bruf_steps=5),
]

debug = False
ensemble_sizes = range(3, 21, 1)

key = jax.random.key(42)
key, *ensemble_keys = jax.random.split(key, 1 + 32)

measurement_covariances = [
    jnp.array([[0.5 ** 2]]),
    jnp.array([[1.0 ** 2]]),
    jnp.array([[2.0 ** 2]]),
]



product_list = list(
    itertools.product(
        ensemble_sizes,
        # ensemble_keys,
        measurement_covariances,
        ensemble_update_methods,
    )
)

from tqdm import tqdm

records = []
for ensemble_size, measurement_covariance, ensemble_update_method in tqdm(product_list):
    
    print(ensemble_size, measurement_covariance.item(), ensemble_update_method.func.__name__, ensemble_update_method.keywords)
    rmse = jax.vmap(lambda key: filter_experiment(ensemble_update_method, ensemble_size, key, measurement_covariance))(jnp.array(ensemble_keys))
    records.append({'ensemble_size':ensemble_size,
                    'measurement_covariance':measurement_covariance.item(),
                    'update_method':ensemble_update_method.func.__name__,
                    'keywords':ensemble_update_method.keywords,
                    'rmse':rmse})
    print(rmse)
