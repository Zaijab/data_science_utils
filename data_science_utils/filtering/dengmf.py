from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jax import lax
from jaxtyping import Array, Bool, Float, Key, jaxtyped
import equinox as eqx
from data_science_utils.measurement_systems import AbstractMeasurementSystem
from data_science_utils.dynamical_systems import ikeda_attractor_discriminator


@jax.jit
def sample_multivariate_normal(
    rng: jax.random.PRNGKey,
    mean: Float[Array, "state_dim"],
    cov: Float[Array, "state_dim state_dim"],
) -> Float[Array, "state_dim"]:
    """
    rng: a JAX PRNGKey
    mean: 1D array of shape [state_dim]
    cov: 2D array of shape [state_dim, state_dim]
    """
    return jax.random.multivariate_normal(rng, mean=mean, cov=cov)


# @partial(jax.jit, static_argnames=["ninverses"])
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
            zi_next = jnp.where(zi_norm > 0, zi_next * (zn_norm / zi_norm), zi_next)
            return zi_next, None

        zi_final, _ = lax.scan(newton_iteration, zn, None, length=10)
        return zi_final

    def body_fn(_, state):
        return ikeda_inv(state)

    # Apply ninverses inverse iterations
    x_inv = lax.fori_loop(0, ninverses, body_fn, x)
    threshold = jnp.sqrt(1.0 / (1.0 - u))
    return jnp.linalg.norm(x_inv) < threshold


# ---------------------------------------------------------------------
# Single-sample Rejection: repeatedly draw until point passes discriminator
# ---------------------------------------------------------------------
# @partial(jax.jit, static_argnames=["ninverses"])
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
        candidate = sample_multivariate_normal(subkey_candidate, mean, cov)
        pass_sample = ikeda_attractor_discriminator(candidate, ninverses, u)

        random_bypass = jax.random.uniform(subkey_bypass) < bypass_prob
        pass_sample = pass_sample | random_bypass
        sample = jnp.where(pass_sample, candidate, sample)
        accepted = accepted | pass_sample
        return (rng, sample, accepted)

    # Initialize with zero vector and False acceptance
    carry_init = (rng, jnp.zeros_like(mean), False)
    _, final_sample, _ = eqx.internal.while_loop(
        cond_fun, body_fun, carry_init, kind="checkpointed", max_steps=10**6
    )
    return final_sample


# ---------------------------------------------------------------------
# Batch Rejection: vmap across multiple means/covariances
# ---------------------------------------------------------------------
# @partial(jax.jit, static_argnames=["ninverses"])
@jax.jit
def rejection_sample_batch(
    rng: jax.random.PRNGKey,
    means: Float[Array, "batch state_dim"],
    covs: Float[Array, "batch state_dim state_dim"],
    ninverses: int = 10,
    u: float = 0.9,
) -> Float[Array, "batch state_dim"]:
    """
    For each row in 'means' and 'covs', run rejection_sample_single.
    """
    batch_size = means.shape[0]
    subkeys = jax.random.split(rng, batch_size)
    return jax.vmap(lambda sk, m, c: rejection_sample_single(sk, m, c, ninverses, u))(
        subkeys, means, covs
    )


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["debug"])
def ensemble_gaussian_mixture_filter_update_point(
    point: Float[Array, "state_dim"],
    prior_mixture_covariance: Float[Array, "state_dim state_dim"],
    measurement: Float[Array, "measurement_dim"],
    measurement_device: object,
    debug: bool = False,
):
    measurement_jacobian = jax.jacfwd(measurement_device)(point)
    # if debug:
    #     assert isinstance(measurement_jacobian, Float[Array, "measurement_dim state_dim"])
    #     jax.debug.callback(has_nan, measurement_jacobian)

    kalman_gain = (
        prior_mixture_covariance
        @ measurement_jacobian.T
        @ jnp.linalg.inv(
            measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
            + measurement_device.covariance
        )
    )
    # if debug:
    #     assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])
    #     jax.debug.callback(has_nan, kalman_gain)

    gaussian_mixture_covariance = (
        jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian
    ) @ prior_mixture_covariance  # + 1e-10 * jnp.eye(point.shape[0])
    # if debug:
    #     assert isinstance(gaussian_mixture_covariance, Float[Array, "state_dim state_dim"])
    #     # jax.debug.callback(is_positive_definite, gaussian_mixture_covariance)
    #     jax.debug.callback(has_nan, gaussian_mixture_covariance)

    point = point - (
        kalman_gain @ jnp.atleast_2d(measurement_device(point) - measurement)
    ).reshape(-1)
    # if debug:
    #     assert isinstance(point, Float[Array, "state_dim"])

    logposterior_weights = jsp.stats.multivariate_normal.logpdf(
        measurement,
        measurement_device(point),
        measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
        + measurement_device.covariance,
    )
    # if debug:
    #     assert isinstance(logposterior_weights, Float[Array, ""])

    return point, logposterior_weights, gaussian_mixture_covariance


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["debug"])
def discriminator_ensemble_gaussian_mixture_filter_update_ensemble(
    state: Float[Array, "batch_dim state_dim"],
    bandwidth_factor: float,
    key: Key[Array, ""],
    measurement: Float[Array, "measurement_dim"],
    measurement_device,
    ninverses: int,
    debug: bool = False,
) -> Float[Array, "batch_dim state_dim"]:
    key: Key[Array, ""]
    subkey: Key[Array, ""]
    subkeys: Key[Array, "batch_dim"]

    key, subkey, rejection_sample_key, *subkeys = jax.random.split(
        key, 2 + 1 + state.shape[0]
    )
    subkeys = jnp.array(subkeys)
    # if debug:
    #     assert isinstance(subkeys, Key[Array, "batch_dim"])

    emperical_covariance = jnp.cov(state.T)
    # if debug:
    #     # jax.debug.callback(is_positive_definite, emperical_covariance)
    #     assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])

    mixture_covariance = bandwidth_factor * emperical_covariance
    # if debug:
    #     assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

    posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(
        ensemble_gaussian_mixture_filter_update_point,
        in_axes=(0, None, None, None, None),
    )(state, mixture_covariance, measurement, measurement_device, debug)

    # if debug:
    #     assert isinstance(posterior_ensemble, Float[Array, "batch_dim state_dim"])
    #     assert isinstance(logposterior_weights, Float[Array, "batch_dim"])
    #     assert isinstance(posterior_covariances, Float[Array, "batch_dim state_dim state_dim"])
    #     jax.debug.callback(has_nan, posterior_covariances)

    # Scale Weights
    m = jnp.max(logposterior_weights)
    g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
    posterior_weights = jnp.exp(logposterior_weights - g)
    posterior_weights = posterior_weights / jnp.sum(posterior_weights)
    # if debug:
    #     assert isinstance(posterior_weights, Float[Array, "batch_dim"])

    # Prevent Degenerate Particles
    variable = jax.random.choice(
        subkey, state.shape[0], shape=(state.shape[0],), p=posterior_weights
    )
    posterior_ensemble = posterior_ensemble[variable, ...]
    posterior_covariances = posterior_covariances[variable, ...]
    # if debug:
    #     jax.debug.callback(has_nan, posterior_covariances)

    # posterior_samples = sample_multivariate_normal(subkeys, posterior_ensemble, posterior_covariances)
    posterior_samples = rejection_sample_batch(
        rejection_sample_key,
        posterior_ensemble,
        posterior_covariances,
        ninverses=ninverses,
        u=0.9,
    )
    # if debug:
    #     assert isinstance(posterior_weights, Float[Array, "batch_dim"])

    return posterior_samples


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def diengmf_update(
    key: Key[Array, "..."],
    ensemble: Float[Array, "batch_size state_dim"],
    measurement: Float[Array, "measurement_dim"],
    measurement_system: AbstractMeasurementSystem,
    debug: bool = False,
) -> Float[Array, "batch_size state_dim"]:
    pass
