import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Key, jaxtyped

from data_science_utils.filters import AbstractFilter
from data_science_utils.measurement_systems import AbstractMeasurementSystem
from data_science_utils.statistics import GMM


class EnGMPHD(eqx.Module, strict=True):
    """
    The Ensemble Gaussian Mixture Probability Hypothesis Density filter.

    SOURCE:
    [1] Dalton Durant and Renato Zanetti. "Kernel-Based Ensemble Gaussian
    Mixture Probability Hypothesis Density Filter." In 28th International
    Conference on Information Fusion, Rio de Janeiro, Brazil. 2025.
    """

    debug: bool = False
    clutter_density: float = 6.25e-8
    lambda_c: float = 10.0
    detection_probability: float = 0.98

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update_point(
        self,
        point: Float[Array, "state_dim"], # x_{k|k-1}^(i)
        prior_mixture_covariance: Float[Array, "state_dim state_dim"], # \hat{P}_{k|k-1}^(i)
        measurement: Float[Array, "measurement_dim"], # z
        measurement_system: AbstractMeasurementSystem, # h
    ):
        ### (eq. 21)
        # H_{k}^{(i)} = \frac{\partial h}{\partial x} (x_{k|k-1}^(i))
        measurement_jacobian = jax.jacfwd(measurement_system)(point)

        if self.debug:
            assert isinstance(
                measurement_jacobian, Float[Array, "measurement_dim state_dim"]
            ), measurement_jacobian.shape
            
        ### (eq. 19)
        # S_k^(i) = H_k^(i) P_{k | k - 1}^(i) H_k^(i) + R

        innovation_cov = (
            measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
            + measurement_system.covariance
        )
        innovation_cov = (innovation_cov + innovation_cov.T) / 2  # Symmetrize
        if self.debug:
            assert isinstance(innovation_cov, Float[Array, "measurement_dim measurement_dim"])

        ### (eq. 18)

        # K_k^(i) = P H.T S^(-1)
        kalman_gain = jax.scipy.linalg.solve(innovation_cov, measurement_jacobian @ prior_mixture_covariance).T

        if self.debug:
            assert isinstance(kalman_gain, Float[Array, "state_dim measurement_dim"])
            # jax.debug.print("Hello {}", jnp.allclose(kalman_gain_unstable, kalman_gain))

        ### (eq. 17)
        
        # \hat{P}_{k | k}^{(i)} = \hat{P}_{k | k - 1}^{(i)} - K_{k}^{(i)} H_{k}^{(i)} \hat{P}_{k | k - 1}^{(i)}
        # We may, of course, factor to the right
        # \hat{P}_{k | k}^{(i)} = ( I - K_{k}^{(i)} H_{k}^{(i)} ) \hat{P}_{k | k - 1}^{(i)}
        posterior_covariance = (
            jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian
        ) @ prior_mixture_covariance

        if self.debug:
            assert isinstance(
                posterior_covariance, Float[Array, "state_dim state_dim"]
            )

        ### (eq. 16)
        
        # \hat{x}_{k | k}^{(i)} = \hat{x}_{k | k - 1}^{(i)} + K_{k}^{(i)} ( z - h(\hat{x}_{k | k - 1}^{(i)}))
        posterior_point = point + kalman_gain @ (measurement - measurement_system(point))

        if self.debug:
            assert isinstance(point, Float[Array, "state_dim"])
            assert measurement_system(point).shape == measurement.shape
            assert posterior_point.shape == point.shape

        ### (eq. 22)
        # \xi_{k}^{(i)} = N(z; \hat{x}_{k | k - 1}^{(i)}, S_{k}^{(i)})
        logposterior_weight = jsp.stats.multivariate_normal.logpdf(
            measurement,
            mean=measurement_system(point),
            cov=innovation_cov
        )

        if self.debug:
            assert isinstance(logposterior_weight, Float[Array, ""])

        return posterior_point, posterior_covariance, logposterior_weight

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update(
        self,
        key: Key[Array, ""],
        prior_gmm: GMM,
        measurements: Float[Array, "num_measurements measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ):
        prior_ensemble = prior_gmm.means
        prior_covs = prior_gmm.covs
        prior_weights = prior_gmm.weights

        if self.debug:
            assert isinstance(prior_gmm.means, Float[Array, "num_components state_dim"])
            assert isinstance(prior_gmm.covs, Float[Array, "num_components state_dim state_dim"])
            assert isinstance(prior_gmm.weights, Float[Array, "num_components"])

        ### Silverman Covariance Estimate
        estimated_cardinality = jnp.ceil(jnp.sum(prior_weights))
        silverman_bandwidth = ((4) / (prior_ensemble.shape[0] * (prior_ensemble.shape[-1] + 2))) ** ((2) / (prior_ensemble.shape[-1] + 4))
        emperical_covariance = jnp.cov(prior_ensemble.T)
        mixture_covariance = (silverman_bandwidth / estimated_cardinality) * emperical_covariance

        if self.debug:
            assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])
            assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

        ### Localization
        state_dim = emperical_covariance.shape[0]
        i_indices = jnp.arange(state_dim)[:, None]
        j_indices = jnp.arange(state_dim)[None, :]
        distances = jnp.abs(i_indices - j_indices)
        L = 3.0
        rho = jnp.exp(-(distances**2) / (2 * L**2))
        mixture_covariance = rho * mixture_covariance

        if self.debug:
            assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])



        ### EnKF Update Per Measurement
        def process_measurement(measurement):
            return jax.vmap(self.update_point, in_axes=(0, None, None, None))(
                prior_gmm.means,
                mixture_covariance,
                measurement,
                measurement_system,
            )

        # \hat{x}_{k | k}^{(i)}, \hat{P}_{k | k}^{(i)}, log(\xi_{k}^{(i)})
        posterior_ensemble, posterior_covariances, logposterior_weights = jax.vmap(
            process_measurement
        )(measurements)
        


        ### Log Sum Exp for w_{k | k}^{(i)}
        log_numerator = (jnp.log(self.detection_probability) + jnp.log(prior_weights)[None, :] + logposterior_weights)
        log_sum_detection = jax.scipy.special.logsumexp(log_numerator, axis=1)
        log_denominator = jnp.log(self.lambda_c * self.clutter_density + jnp.exp(log_sum_detection))
        posterior_weights = jnp.exp(log_numerator - log_denominator[:, None])

        if self.debug:
            assert isinstance(log_numerator, Float[Array, "num_measurements num_components"])
            assert isinstance(log_sum_detection, Float[Array, "num_measurements"])
            assert isinstance(log_denominator, Float[Array, "num_measurements"])
            assert isinstance(posterior_weights, Float[Array, "num_measurements num_components"])

        ### GMM: \sum_{z \in Z_k} \sum_{i = 1}^{J_k} w_{k | k}^{(i)} N( \cdot ; \hat{x}_{k | k}^{(i)}, \hat{P}_{k | k}^{(i)})
        detection_gmm = GMM(
            posterior_ensemble.reshape(-1, posterior_ensemble.shape[-1]),
            posterior_covariances.reshape(-1, *posterior_covariances.shape[-2:]),
            posterior_weights.flatten(),
        )

        ### GMM: (1 - p_d) * v(x_k)
        missed_gmm = GMM(
            prior_gmm.means,
            prior_gmm.covs,
            (1 - self.detection_probability) * prior_gmm.weights
        )

        ### v(x_k | Z_k) = (1 - p_d) * v(x_k) + \sum_{z \in Z_k} \sum_{i = 1}^{J_k}        
        intensity_function = GMM(
            jnp.concatenate([missed_gmm.means, detection_gmm.means]),
            jnp.concatenate([missed_gmm.covs, detection_gmm.covs]),
            jnp.concatenate([missed_gmm.weights, detection_gmm.weights]),
        )

        ### Sampling
        # Measurement * J_k + J_k -> J_k via sampling
        key, subkey = jax.random.split(key)
        posterior_gmm_means = eqx.filter_vmap(intensity_function.sample)(
            jax.random.split(subkey, prior_gmm.means.shape[0])
        )

        posterior_gmm = GMM(
            posterior_gmm_means,
            jnp.tile(
                (silverman_bandwidth / estimated_cardinality) * jnp.cov(posterior_gmm_means.T),
                (prior_gmm.means.shape[0], 1, 1),
            ),
            jnp.full(prior_gmm.means.shape[0], jnp.sum(intensity_function.weights) / prior_gmm.means.shape[0]),
        )
        
        return posterior_gmm
