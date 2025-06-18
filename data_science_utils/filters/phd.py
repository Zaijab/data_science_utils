import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Key, jaxtyped

from data_science_utils.filters import AbstractFilter
from data_science_utils.measurement_systems import AbstractMeasurementSystem
from data_science_utils.statistics import GMM


@jax.jit
@jax.vmap
def sample_gaussian_mixture(key, point, cov):
    return jax.random.multivariate_normal(key, mean=point, cov=cov)


# class EnGMPHDRFS(eqx.Module, strict=True):
#     debug: bool = False
#     sampling_function: jax.tree_util.Partial = jax.tree_util.Partial(
#         sample_gaussian_mixture
#     )

#     @jaxtyped(typechecker=typechecker)
#     @eqx.filter_jit
#     def update_point(
#         self,
#         point: Float[Array, "state_dim"],
#         prior_mixture_covariance: Float[Array, "state_dim state_dim"],
#         measurement: Float[Array, "measurement_dim"],
#         measurement_device: AbstractMeasurementSystem,
#         measurement_device_covariance,
#     ):
#         def measurement_function(point):
#             measurement, _ = measurement_device(
#                 RFS(jnp.atleast_2d(point[:3]), jnp.tile(True, 1))
#             )
#             return measurement

#         ybar = measurement_function(point)

#         measurement_jacobian = jax.jacfwd(measurement_function)(point)[0, ...]

#         if self.debug:
#             assert isinstance(
#                 measurement_jacobian, Float[Array, "measurement_dim state_dim"]
#             )

#         innovation_cov = (
#             measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
#             + measurement_device_covariance
#         )
#         # innovation_cov = (innovation_cov + innovation_cov.T) / 2  # Symmetrize

#         kalman_gain = (
#             prior_mixture_covariance
#             @ measurement_jacobian.T
#             @ jnp.linalg.inv(innovation_cov)
#         )

#         if self.debug:
#             assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])

#         gaussian_mixture_covariance = (
#             jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian
#         ) @ prior_mixture_covariance  # + 1e-10 * jnp.eye(point.shape[0])

#         if self.debug:
#             assert isinstance(
#                 gaussian_mixture_covariance, Float[Array, "state_dim state_dim"]
#             )

#         point = point - (kalman_gain @ jnp.atleast_2d(ybar - measurement).T).reshape(-1)

#         if self.debug:
#             assert isinstance(point, Float[Array, "state_dim"])

#         log_posterior_cov = (
#             measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
#             + measurement_device_covariance
#         )
#         # log_posterior_cov = (log_posterior_cov + log_posterior_cov.T) / 2
#         logposterior_weights = jsp.stats.multivariate_normal.logpdf(
#             measurement,
#             ybar,
#             log_posterior_cov,
#         )
#         if self.debug:
#             assert isinstance(logposterior_weights, Float[Array, ""])

#         return point, jnp.squeeze(logposterior_weights), gaussian_mixture_covariance

#     @jaxtyped(typechecker=typechecker)
#     @eqx.filter_jit
#     def update(
#         self,
#         key: Key[Array, ""],
#         prior_gmm: GMM,
#         measurements: Float[Array, "num_measurements measurement_dim"],
#         measurements_mask: Bool[Array, "num_measurements"],
#         measurement_system: AbstractMeasurementSystem,
#         clutter_density: float,
#         detection_probability: float,
#     ) -> GMM:
#         subkey: Key[Array, ""]
#         subkeys: Key[Array, "batch_dim"]

#         prior_ensemble = prior_gmm.means
#         prior_covs = prior_gmm.covs
#         prior_weights = prior_gmm.weights

#         key, subkey, *subkeys = jax.random.split(key, 2 + prior_ensemble.shape[0])
#         subkeys = jnp.array(subkeys)

#         if self.debug:
#             assert isinstance(subkeys, Key[Array, "batch_dim"])

#         bandwidth = (
#             (4) / (prior_ensemble.shape[0] * (prior_ensemble.shape[-1] + 2))
#         ) ** ((2) / (prior_ensemble.shape[-1] + 4))
#         emperical_covariance = jnp.cov(prior_ensemble.T)

#         state_dim = emperical_covariance.shape[0]
#         i_indices = jnp.arange(state_dim)[:, None]
#         j_indices = jnp.arange(state_dim)[None, :]
#         distances = jnp.abs(i_indices - j_indices)

#         L = 3.0  # or 4.0
#         rho = jnp.exp(-(distances**2) / (2 * L**2))

#         emperical_covariance = emperical_covariance * rho
#         mixture_covariance = bandwidth * emperical_covariance

#         if self.debug:
#             jax.debug.callback(is_positive_definite, emperical_covariance)
#             assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])
#             assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

#         def process_measurement(measurement, mask):
#             return jax.lax.cond(
#                 mask,
#                 lambda: jax.vmap(
#                     self.update_point, in_axes=(0, None, None, None, None)
#                 )(
#                     prior_gmm.means,
#                     mixture_covariance,
#                     measurement,
#                     measurement_system,
#                     measurement_system.covariance,
#                 ),
#                 lambda: (
#                     jnp.zeros_like(prior_gmm.means),
#                     jnp.full((prior_gmm.weights.shape[0]), jnp.asarray(0.0)),
#                     jnp.zeros_like(prior_gmm.covs),
#                 ),
#             )

#         posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(
#             process_measurement
#         )(measurements, measurements_mask)

#         print(posterior_ensemble)

#         logposterior_weights = logposterior_weights[:, :]
#         print(logposterior_weights)

#         # PHD update formula: log(P_D * w_prior * likelihood)
#         log_detection_terms = (
#             jnp.log(detection_probability)
#             + jnp.log(prior_weights)[None, :]
#             + logposterior_weights
#         )

#         # Denominator: log(clutter + sum(detection_terms)) per measurement
#         log_sum_detection = jax.scipy.special.logsumexp(
#             log_detection_terms, axis=1, keepdims=True
#         )
#         log_denominators = jnp.log(clutter_density + jnp.exp(log_sum_detection))

#         # Final normalized weights
#         normalized_weights = jnp.where(
#             measurements_mask[:, None],
#             jnp.exp(log_detection_terms - log_denominators),
#             0.0,
#         )
#         print(normalized_weights)

#         # Scale Weights
#         # m = jnp.max(logposterior_weights)
#         # g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
#         # posterior_weights = jnp.exp(logposterior_weights - g)
#         # log_normalizer = jax.scipy.special.logsumexp(
#         #     logposterior_weights, axis=1, keepdims=True
#         # )
#         # posterior_weights = jnp.exp(logposterior_weights - log_normalizer)
#         # posterior_weights = jnp.exp(logposterior_weights)

#         # detection_weights = (
#         #     detection_probability * prior_weights[None, :] * posterior_weights
#         # )
#         # denominators = clutter_density + jnp.sum(
#         #     detection_weights, axis=1, keepdims=True
#         # )
#         # normalized_weights = jnp.where(
#         #     measurements_mask[:, None], detection_weights / denominators, 0.0
#         # )

#         key, subkey = jax.random.split(key)

#         # Missed detection
#         missed_weights = (1 - detection_probability) * prior_weights
#         missed_means = prior_gmm.means
#         missed_covs = prior_gmm.covs

#         flat_detection_weights = normalized_weights.reshape(-1)
#         flat_detection_means = posterior_ensemble.reshape(
#             -1, posterior_ensemble.shape[-1]
#         )
#         flat_detection_covs = posterior_covariances.reshape(
#             -1, *posterior_covariances.shape[-2:]
#         )

#         final_weights = jnp.concatenate([missed_weights, flat_detection_weights])
#         final_means = jnp.concatenate([missed_means, flat_detection_means])
#         final_covs = jnp.concatenate([missed_covs, flat_detection_covs])

#         posterior_gmm = sample_from_large_gmm(
#             final_means,
#             final_weights,
#             final_covs,
#             subkey,
#             250,
#         )

#         if self.debug:
#             assert isinstance(posterior_ensemble, Float[Array, "batch_dim state_dim"])
#             assert isinstance(logposterior_weights, Float[Array, "batch_dim"])
#             assert isinstance(
#                 posterior_covariances, Float[Array, "batch_dim state_dim state_dim"]
#             )
#             jax.self.debug.callback(has_nan, posterior_covariances)

#         # Missed Detection and Clutter

#         if self.debug:
#             assert isinstance(posterior_weights, Float[Array, "batch_dim"])

#         # Prevent Degenerate Particles
#         p = posterior_gmm.weights / jnp.sum(posterior_gmm.weights)
#         variable = jax.random.choice(
#             subkey,
#             posterior_gmm.means.shape[0],
#             shape=(posterior_gmm.means.shape[0],),
#             p=p,
#         )
#         posterior_ensemble = posterior_gmm.means[variable, ...]
#         posterior_covariances = posterior_gmm.covs[variable, ...]

#         if self.debug:
#             jax.self.debug.callback(has_nan, posterior_covariances)

#         posterior_samples = self.sampling_function(
#             subkeys, posterior_ensemble, posterior_covariances
#         )

#         if self.debug:
#             assert isinstance(posterior_weights, Float[Array, "batch_dim"])

#         return GMM(posterior_samples, posterior_covariances, posterior_gmm.weights)


class EnGMPHD(eqx.Module, strict=True):
    debug: bool = False
    sampling_function: jax.tree_util.Partial = jax.tree_util.Partial(
        sample_gaussian_mixture
    )
    clutter_density: float = 6.25e-8
    detection_probability: float = 0.98

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update_point(
        self,
        point: Float[Array, "state_dim"],
        prior_mixture_covariance: Float[Array, "state_dim state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_device: AbstractMeasurementSystem,
        measurement_device_covariance,
    ):
        def measurement_function(point):
            return measurement_device(point[:3])

        ybar = measurement_function(point)

        measurement_jacobian = jax.jacfwd(measurement_function)(point)

        if self.debug:
            assert isinstance(
                measurement_jacobian, Float[Array, "measurement_dim state_dim"]
            ), measurement_jacobian.shape

        innovation_cov = (
            measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
            + measurement_device_covariance
        )
        # innovation_cov = (innovation_cov + innovation_cov.T) / 2  # Symmetrize

        kalman_gain = (
            prior_mixture_covariance
            @ measurement_jacobian.T
            @ jnp.linalg.inv(innovation_cov)
        )

        if self.debug:
            assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])

        gaussian_mixture_covariance = (
            jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian
        ) @ prior_mixture_covariance  # + 1e-10 * jnp.eye(point.shape[0])

        if self.debug:
            assert isinstance(
                gaussian_mixture_covariance, Float[Array, "state_dim state_dim"]
            )

        point = point - (kalman_gain @ jnp.atleast_2d(ybar - measurement).T).reshape(-1)

        if self.debug:
            assert isinstance(point, Float[Array, "state_dim"])

        log_posterior_cov = (
            measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
            + measurement_device_covariance
        )
        # log_posterior_cov = (log_posterior_cov + log_posterior_cov.T) / 2
        logposterior_weights = jsp.stats.multivariate_normal.logpdf(
            measurement,
            ybar,
            log_posterior_cov,
        )
        if self.debug:
            assert isinstance(logposterior_weights, Float[Array, ""])

        return point, jnp.squeeze(logposterior_weights), gaussian_mixture_covariance

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update(
        self,
        key: Key[Array, ""],
        prior_gmm: GMM,
        measurements: Float[Array, "num_measurements measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> GMM:
        subkey: Key[Array, ""]
        subkeys: Key[Array, "batch_dim"]

        prior_ensemble = prior_gmm.means
        prior_covs = prior_gmm.covs
        prior_weights = prior_gmm.weights

        key, subkey, *subkeys = jax.random.split(key, 2 + prior_ensemble.shape[0])
        subkeys = jnp.array(subkeys)

        if self.debug:
            assert isinstance(subkeys, Key[Array, "batch_dim"])

        bandwidth = (
            (4) / (prior_ensemble.shape[0] * (prior_ensemble.shape[-1] + 2))
        ) ** ((2) / (prior_ensemble.shape[-1] + 4))
        emperical_covariance = jnp.cov(prior_ensemble.T)

        state_dim = emperical_covariance.shape[0]
        i_indices = jnp.arange(state_dim)[:, None]
        j_indices = jnp.arange(state_dim)[None, :]
        distances = jnp.abs(i_indices - j_indices)

        L = 3.0  # or 4.0
        rho = jnp.exp(-(distances**2) / (2 * L**2))

        emperical_covariance = emperical_covariance * rho
        mixture_covariance = bandwidth * emperical_covariance

        if self.debug:
            # jax.debug.callback(is_positive_definite, emperical_covariance)
            assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])
            assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

        def process_measurement(measurement):
            return jax.vmap(self.update_point, in_axes=(0, None, None, None, None))(
                prior_gmm.means,
                mixture_covariance,
                measurement,
                measurement_system,
                measurement_system.covariance,
            )

        posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(
            process_measurement
        )(measurements)

        print(posterior_ensemble)

        logposterior_weights = logposterior_weights[:, :]
        print(logposterior_weights)

        # Scale Weights
        def logsumexp_per_measurement(logposterior_weights):
            m = jnp.max(logposterior_weights)
            g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
            posterior_weights = jnp.exp(logposterior_weights - g)

        # log_normalizer = jax.scipy.special.logsumexp(
        #     logposterior_weights, axis=1, keepdims=True
        # )
        # posterior_weights = jnp.exp(logposterior_weights - log_normalizer)
        # posterior_weights = jnp.exp(logposterior_weights)

        # detection_weights = (
        #     detection_probability * prior_weights[None, :] * posterior_weights
        # )
        # denominators = clutter_density + jnp.sum(
        #     detection_weights, axis=1, keepdims=True
        # )
        # normalized_weights = jnp.where(
        #     measurements_mask[:, None], detection_weights / denominators, 0.0
        # )

        key, subkey = jax.random.split(key)

        # Missed detection
        missed_weights = (1 - self.detection_probability) * prior_weights
        missed_means = prior_gmm.means
        missed_covs = prior_gmm.covs

        flat_detection_weights = normalized_weights.reshape(-1)
        flat_detection_means = posterior_ensemble.reshape(
            -1, posterior_ensemble.shape[-1]
        )
        flat_detection_covs = posterior_covariances.reshape(
            -1, *posterior_covariances.shape[-2:]
        )

        final_weights = jnp.concatenate([missed_weights, flat_detection_weights])
        final_means = jnp.concatenate([missed_means, flat_detection_means])
        final_covs = jnp.concatenate([missed_covs, flat_detection_covs])

        posterior_gmm = sample_from_large_gmm(
            final_means,
            final_weights,
            final_covs,
            subkey,
            250,
        )

        if self.debug:
            assert isinstance(posterior_ensemble, Float[Array, "batch_dim state_dim"])
            assert isinstance(logposterior_weights, Float[Array, "batch_dim"])
            assert isinstance(
                posterior_covariances, Float[Array, "batch_dim state_dim state_dim"]
            )
            jax.self.debug.callback(has_nan, posterior_covariances)

        # Missed Detection and Clutter

        if self.debug:
            assert isinstance(posterior_weights, Float[Array, "batch_dim"])

        # Prevent Degenerate Particles
        p = posterior_gmm.weights / jnp.sum(posterior_gmm.weights)
        variable = jax.random.choice(
            subkey,
            posterior_gmm.means.shape[0],
            shape=(posterior_gmm.means.shape[0],),
            p=p,
        )
        posterior_ensemble = posterior_gmm.means[variable, ...]
        posterior_covariances = posterior_gmm.covs[variable, ...]

        if self.debug:
            jax.self.debug.callback(has_nan, posterior_covariances)

        posterior_samples = self.sampling_function(
            subkeys, posterior_ensemble, posterior_covariances
        )

        if self.debug:
            assert isinstance(posterior_weights, Float[Array, "batch_dim"])

        return GMM(posterior_samples, posterior_covariances, posterior_gmm.weights)
