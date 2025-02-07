from functools import partial

import jax
import jax.numpy as jnp
from data_science_utils.dynamical_systems import IkedaSystem, flow, generate
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

@jax.jit
def sample_multivariate_normal(
    rng: Key[Array, ""],
    mean: Float[Array, "state_dim"],
    cov: Float[Array, "state_dim state_dim"]
) -> Float[Array, "state_dim"]:
    """
    rng: a JAX PRNGKey
    mean: 1D array of shape [state_dim]
    cov: 2D array of shape [state_dim, state_dim]
    """
    return jax.random.multivariate_normal(rng, mean=mean, cov=cov)

#@partial(jax.jit, static_argnames=["ninverses"])
@jax.jit
def ikeda_attractor_discriminator(
    x: Float[Array, "state_dim"],
    ninverses: int = 10,
    u: float = 0.9,
) -> Bool[Array, ""]:
    """
    Returns True if 'x' lies on the Ikeda attractor, computed via ninverses
    inverse iterations. For batch usage, vmap over x.
    """
    def ikeda_inv(zp1):
        """
        Single inverse iteration: Newton's method to invert Ikeda forward step,
        then radius normalization to match the input radius.
        """
        # Shift/unscale
        zn = (zp1 - jnp.array([1.0, 0.0])) / u

        def newton_iteration(zi, _):
            xi, yi = zi[0], zi[1]
            r2 = xi**2 + yi**2
            opx2y2 = 1.0 + r2
            ti = 0.4 - 6.0 / opx2y2
            cti, sti = jnp.cos(ti), jnp.sin(ti)

            # Partial derivatives of ti wrt (x, y)
            dti_dx = 12.0 * xi / (opx2y2**2)
            dti_dy = 12.0 * yi / (opx2y2**2)

            # Jacobian
            J11 = cti - (yi * cti + xi * sti) * dti_dx
            J12 = -sti - (yi * cti + xi * sti) * dti_dy
            J21 = sti + (xi * cti - yi * sti) * dti_dx
            J22 = cti + (xi * cti - yi * sti) * dti_dy
            detJ = J11 * J22 - J12 * J21

            # Residual
            c0 = zn[0] - (xi * cti - yi * sti)
            c1 = zn[1] - (xi * sti + yi * cti)

            dx0 = (J22 * c0 - J12 * c1) / detJ
            dx1 = (-J21 * c0 + J11 * c1) / detJ

            zi_next = jnp.array([xi + dx0, yi + dx1])
            # Enforce same radius as zn
            zn_norm = jnp.linalg.norm(zn)
            zi_norm = jnp.linalg.norm(zi_next)
            zi_next = jnp.where(
                zi_norm > 0,
                zi_next * (zn_norm / zi_norm),
                zi_next
            )
            return zi_next, None

        zi_final, _ = lax.scan(newton_iteration, zn, None, length=10)
        return zi_final

    def body_fn(_, state):
        return ikeda_inv(state)

    # Apply ninverses inverse iterations
    x_inv = lax.fori_loop(0, ninverses, body_fn, x)
    threshold = jnp.sqrt(1.0 / (1.0 - u))
    return (jnp.linalg.norm(x_inv) < threshold)

# ---------------------------------------------------------------------
# Single-sample Rejection: repeatedly draw until point passes discriminator
# ---------------------------------------------------------------------
#@partial(jax.jit, static_argnames=["ninverses"])
@jax.jit
def rejection_sample_single(
    rng: jax.random.PRNGKey,
    mean: Float[Array, "state_dim"],
    cov: Float[Array, "state_dim state_dim"],
    ninverses: int = 10,
    u: float = 0.9,
    bypass_prob: float = 0.0,
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
        rng, sample, accepted = carry
        rng, subkey_candidate, subkey_bypass = jax.random.split(rng, 3)
        candidate = jax.random.multivariate_normal(subkey_candidate, mean=mean, cov=cov)
        # candidate = sample_multivariate_normal(subkey_candidate, mean, cov)
        pass_sample = ikeda_attractor_discriminator(candidate, ninverses, u)

        random_bypass = jax.random.uniform(subkey_bypass) < bypass_prob
        pass_sample = pass_sample | random_bypass
        sample = jnp.where(pass_sample, candidate, sample)
        accepted = accepted | pass_sample
        return (rng, sample, accepted)

    # Initialize with zero vector and False acceptance
    carry_init = (rng, jnp.zeros_like(mean), False)
    _, final_sample, _ = eqx.internal.while_loop(cond_fun, body_fun, carry_init, kind="checkpointed", max_steps=10**6)
    return final_sample

# ---------------------------------------------------------------------
# Batch Rejection: vmap across multiple means/covariances
# ---------------------------------------------------------------------
#@partial(jax.jit, static_argnames=["ninverses"])
@jax.jit
def rejection_sample_batch(
    rng: Key[Array, "batch_dim"],
    means: Float[Array, "batch state_dim"],
    covs: Float[Array, "batch state_dim state_dim"],
    ninverses: int = 10,
    u: float = 0.9
) -> Float[Array, "batch state_dim"]:
    """
    For each row in 'means' and 'covs', run rejection_sample_single.
    """
    batch_size = means.shape[0]
    return jax.vmap(
        lambda sk, m, c: rejection_sample_single(sk, m, c, ninverses, u),
    )(rng, means, covs)


##############
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped

import equinox as eqx


@partial(jax.jit, static_argnames=["debug"])
@jaxtyped(typechecker=typechecker)
def ensemble_gaussian_mixture_filter_update_point(
        point: Float[Array, "state_dim"],
        prior_mixture_covariance: Float[Array, "state_dim state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_device: object,
        measurement_device_covariance,
        debug: bool = False
):
    measurement_jacobian = jax.jacfwd(measurement_device)(point)
    if debug:
        assert isinstance(measurement_jacobian, Float[Array, "measurement_dim state_dim"])
        jax.debug.callback(has_nan, measurement_jacobian)

    kalman_gain = prior_mixture_covariance @ measurement_jacobian.T @ jnp.linalg.inv(measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + measurement_device_covariance)
    if debug:
        assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])
        jax.debug.callback(has_nan, kalman_gain)


    gaussian_mixture_covariance = (jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian) @ prior_mixture_covariance #+ 1e-10 * jnp.eye(point.shape[0])
    if debug:
        assert isinstance(gaussian_mixture_covariance, Float[Array, "state_dim state_dim"])
        # jax.debug.callback(is_positive_definite, gaussian_mixture_covariance)
        jax.debug.callback(has_nan, gaussian_mixture_covariance)

    point = point - (kalman_gain @ jnp.atleast_2d(measurement_device(point) - measurement)).reshape(-1)
    if debug:
        assert isinstance(point, Float[Array, "state_dim"])

    logposterior_weights = jsp.stats.multivariate_normal.logpdf(
        measurement, measurement_device(point),
        measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + measurement_device_covariance
    )
    if debug:
        assert isinstance(logposterior_weights, Float[Array, ""])

    return point, logposterior_weights, gaussian_mixture_covariance


@jax.jit
@jax.vmap
def sample_multivariate_normal(key, point, cov):
    return jax.random.multivariate_normal(key, mean=point, cov=cov)



@partial(jax.jit, static_argnames=["debug"])
@jaxtyped(typechecker=typechecker)
def ensemble_gaussian_mixture_filter_update_ensemble(
        state: Float[Array, "batch_dim state_dim"],
        bandwidth_factor: Float[Array, ""],
        key: Key[Array, ""],
        measurement: Float[Array, "measurement_dim"],
        measurement_device_covariance,
        measurement_device,
        sampling_function,
        debug: bool = False
):
    key: Key[Array, ""]
    subkey: Key[Array, ""]
    subkeys: Key[Array, "batch_dim"]

    key, subkey, *subkeys = jax.random.split(key, 2 + state.shape[0])
    subkeys = jnp.array(subkeys)
    if debug:
        assert isinstance(subkeys, Key[Array, "batch_dim"])

    emperical_covariance = jnp.cov(state.T) + 1e-8 * jnp.eye(2)
    if debug:
        # jax.debug.callback(is_positive_definite, emperical_covariance)
        assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])

    mixture_covariance = bandwidth_factor * emperical_covariance
    if debug:
        assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

    posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(ensemble_gaussian_mixture_filter_update_point,
                                                                               in_axes=(0, None, None, None, None, None))(state,
                                                                                                                          mixture_covariance,
                                                                                                                          measurement,
                                                                                                                          measurement_device,
                                                                                                                          measurement_device_covariance,
                                                                                                                          debug)

    if debug:
        assert isinstance(posterior_ensemble, Float[Array, "batch_dim state_dim"])
        assert isinstance(logposterior_weights, Float[Array, "batch_dim"])
        assert isinstance(posterior_covariances, Float[Array, "batch_dim state_dim state_dim"])
        jax.debug.callback(has_nan, posterior_covariances)


    # Scale Weights
    m = jnp.max(logposterior_weights)
    g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
    posterior_weights = jnp.exp(logposterior_weights - g)
    posterior_weights = (posterior_weights / jnp.sum(posterior_weights))
    if debug:
        assert isinstance(posterior_weights, Float[Array, "batch_dim"])


    # Prevent Degenerate Particles
    variable = jax.random.choice(subkey, state.shape[0], shape=(state.shape[0],), p=posterior_weights)
    posterior_ensemble = posterior_ensemble[variable, ...]
    posterior_covariances = posterior_covariances[variable, ...]
    if debug:
        jax.debug.callback(has_nan, posterior_covariances)

    posterior_samples = sampling_function(subkeys, posterior_ensemble, posterior_covariances)
    if debug:
        assert isinstance(posterior_weights, Float[Array, "batch_dim"])

    return posterior_samples



##################


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_compilation_cache_dir", "/home/zjabbar/.cache/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "all")


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
    filter_ensemble_init = jax.random.multivariate_normal(
        subkey,
        shape=(ensemble_size,),
        mean=true_state_init,
        cov=jnp.eye(2),
    )

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
ensemble_sizes = range(3, 30, 2)

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

sampling_functions = [
    jax.tree_util.Partial(sample_multivariate_normal),
    # jax.tree_util.Partial(rejection_sample_batch, ninverses=8, u=0.9),
]


records = []

for ensemble_size in tqdm(ensemble_sizes, leave=False):
    for ensemble_key in tqdm(ensemble_keys):
        for covariance in measurement_covariances:
            for bandwidth in bandwidths:
                for sampling_function in sampling_functions:
                    records.append({
                        "ensemble_size": ensemble_size,
                        "random": str(jax.random.key_data(ensemble_key)),
                        "covariance": covariance.item(),
                        "bandwidth": bandwidth.keywords['scale'],
                        "sampling": sampling_function.func.__name__,
                        "RMSE": filter_experiment(
                            ensemble_size,
                            ensemble_key,
                            covariance,
                            bandwidth,
                            sampling_function,
                        )
                    })

df = pd.DataFrame(records)
