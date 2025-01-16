from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["debug"])
def ensemble_gaussian_mixture_filter_update_point(
        point: Float[Array, "state_dim"],
        prior_mixture_covariance: Float[Array, "state_dim state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_device: object,
        debug: bool = False
):
    measurement_jacobian = jax.jacfwd(measurement_device)(point)
    if debug:
        assert isinstance(measurement_jacobian, Float[Array, "measurement_dim state_dim"])
        jax.debug.callback(has_nan, measurement_jacobian)

    kalman_gain = prior_mixture_covariance @ measurement_jacobian.T @ jnp.linalg.inv(measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + measurement_device.covariance)
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
        measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T + measurement_device.covariance
    )
    if debug:
        assert isinstance(logposterior_weights, Float[Array, ""])

    return point, logposterior_weights, gaussian_mixture_covariance


@jax.jit
@jax.vmap
def sample_multivariate_normal(key, point, cov):
    return jax.random.multivariate_normal(key, mean=point, cov=cov)


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["debug"])
def ensemble_gaussian_mixture_filter_update_ensemble(
        state: Float[Array, "batch_dim state_dim"],
        bandwidth_factor: float,
        key: Key[Array, ""],
        measurement: Float[Array, "measurement_dim"],
        measurement_device,
        debug: bool = False
):
    key: Key[Array, ""]
    subkey: Key[Array, ""]
    subkeys: Key[Array, "batch_dim"]

    key, subkey, *subkeys = jax.random.split(key, 2 + state.shape[0])
    subkeys = jnp.array(subkeys)
    if debug:
        assert isinstance(subkeys, Key[Array, "batch_dim"])

    emperical_covariance = jnp.cov(state.T)
    if debug:
        # jax.debug.callback(is_positive_definite, emperical_covariance)
        assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])

    mixture_covariance = bandwidth_factor * emperical_covariance
    if debug:
        assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

    posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(ensemble_gaussian_mixture_filter_update_point, in_axes=(0, None, None, None, None))(state, mixture_covariance, measurement, measurement_device, debug)

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

    posterior_samples = sample_multivariate_normal(subkeys, posterior_ensemble, posterior_covariances)
    if debug:
        assert isinstance(posterior_weights, Float[Array, "batch_dim"])

    return posterior_covariances, posterior_samples


@dataclass
class EnGMF:
    dynamical_system: object
    measurement_device: object
    state: Float[Array, "batch_dim state_dim"]
    bandwidth_factor: float

    def update(self, key, measurement, debug):
        self.covariences, self.state = ensemble_gaussian_mixture_filter_update_ensemble(state=self.state,
                                                                                        bandwidth_factor=self.bandwidth_factor,
                                                                                        measurement_device=self.measurement_device,
                                                                                        key=key,
                                                                                        measurement=measurement,
                                                                                        debug=debug)

    def predict(self):
        self.state = self.dynamical_system.flow(self.state)
