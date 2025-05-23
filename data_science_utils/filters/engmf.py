import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped

from data_science_utils.filters import AbstractFilter
from data_science_utils.measurement_systems import AbstractMeasurementSystem


@jax.jit
@jax.vmap
def sample_gaussian_mixture(key, point, cov):
    return jax.random.multivariate_normal(key, mean=point, cov=cov)


class EnGMF(AbstractFilter, strict=True):
    debug: bool = False
    sampling_function: callable = jax.tree_util.Partial(sample_gaussian_mixture)

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update_point(
        self,
        point: Float[Array, "state_dim"],
        prior_mixture_covariance: Float[Array, "state_dim state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_device: object,
        measurement_device_covariance,
    ):
        measurement_jacobian = jax.jacfwd(measurement_device)(point)
        if self.debug:
            assert isinstance(
                measurement_jacobian, Float[Array, "measurement_dim state_dim"]
            )
            jax.debug.callback(has_nan, measurement_jacobian)

        kalman_gain = (
            prior_mixture_covariance
            @ measurement_jacobian.T
            @ jnp.linalg.inv(
                measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
                + measurement_device_covariance
            )
        )
        if self.debug:
            assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])
            jax.debug.callback(has_nan, kalman_gain)

        gaussian_mixture_covariance = (
            jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian
        ) @ prior_mixture_covariance  # + 1e-10 * jnp.eye(point.shape[0])
        if self.debug:
            assert isinstance(
                gaussian_mixture_covariance, Float[Array, "state_dim state_dim"]
            )
            # jax.debug.callback(is_positive_definite, gaussian_mixture_covariance)
            jax.debug.callback(has_nan, gaussian_mixture_covariance)

        point = point - (
            kalman_gain @ jnp.atleast_2d(measurement_device(point) - measurement)
        ).reshape(-1)
        if self.debug:
            assert isinstance(point, Float[Array, "state_dim"])

        logposterior_weights = jsp.stats.multivariate_normal.logpdf(
            measurement,
            measurement_device(point),
            measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
            + measurement_device_covariance,
        )
        if self.debug:
            assert isinstance(logposterior_weights, Float[Array, ""])

        return point, logposterior_weights, gaussian_mixture_covariance

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update(
        self,
        key: Key[Array, ""],
        prior_ensemble: Float[Array, "batch_dim state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "batch_size state_dim"]:
        key: Key[Array, ""]
        subkey: Key[Array, ""]
        subkeys: Key[Array, "batch_dim"]

        key, subkey, *subkeys = jax.random.split(key, 2 + prior_ensemble.shape[0])
        subkeys = jnp.array(subkeys)

        if self.debug:
            assert isinstance(subkeys, Key[Array, "batch_dim"])

        bandwidth = (
            (4) / (prior_ensemble.shape[0] * (prior_ensemble.shape[-1] + 2))
        ) ** ((2) / (prior_ensemble.shape[-1] + 4))
        emperical_covariance = jnp.cov(prior_ensemble.T)  # + 1e-8 * jnp.eye(2)

        state_dim = emperical_covariance.shape[0]
        i_indices = jnp.arange(state_dim)[:, None]
        j_indices = jnp.arange(state_dim)[None, :]
        distances = jnp.abs(i_indices - j_indices)

        # Gaussian localization with radius L
        L = 3.0  # or 4.0
        rho = jnp.exp(-(distances**2) / (2 * L**2))

        emperical_covariance = emperical_covariance * rho

        if self.debug:
            jax.debug.callback(is_positive_definite, emperical_covariance)
            assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])

        mixture_covariance = bandwidth * emperical_covariance
        if self.debug:
            assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

        posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(
            self.update_point,
            in_axes=(0, None, None, None, None),
        )(
            prior_ensemble,
            mixture_covariance,
            measurement,
            measurement_system,
            measurement_system.covariance,
        )

        if self.debug:
            assert isinstance(posterior_ensemble, Float[Array, "batch_dim state_dim"])
            assert isinstance(logposterior_weights, Float[Array, "batch_dim"])
            assert isinstance(
                posterior_covariances, Float[Array, "batch_dim state_dim state_dim"]
            )
            jax.self.debug.callback(has_nan, posterior_covariances)

        # Scale Weights
        m = jnp.max(logposterior_weights)
        g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
        posterior_weights = jnp.exp(logposterior_weights - g)
        posterior_weights = posterior_weights / jnp.sum(posterior_weights)
        if self.debug:
            assert isinstance(posterior_weights, Float[Array, "batch_dim"])

        # Prevent Degenerate Particles
        variable = jax.random.choice(
            subkey,
            prior_ensemble.shape[0],
            shape=(prior_ensemble.shape[0],),
            p=posterior_weights,
        )
        posterior_ensemble = posterior_ensemble[variable, ...]
        posterior_covariances = posterior_covariances[variable, ...]

        if self.debug:
            jax.self.debug.callback(has_nan, posterior_covariances)

        posterior_samples = self.sampling_function(
            subkeys, posterior_ensemble, posterior_covariances
        )
        if self.debug:
            assert isinstance(posterior_weights, Float[Array, "batch_dim"])

        return posterior_samples
