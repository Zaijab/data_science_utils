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
        # H_{k}^(i) = del h / del x (x_{k|k-1}^(i))
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
        
        # innovation_cov = (innovation_cov + innovation_cov.T) / 2  # Symmetrize
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

        estimated_cardinality = jnp.sum(prior_weights)
        silverman_bandwidth = ((4) / (prior_ensemble.shape[0] * (prior_ensemble.shape[-1] + 2))) ** ((2) / (prior_ensemble.shape[-1] + 4))
        emperical_covariance = jnp.cov(prior_ensemble.T)

        if self.debug:
            assert isinstance(prior_gmm.means, Float[Array, "num_components state_dim"])
            assert isinstance(
                prior_gmm.covs, Float[Array, "num_components state_dim state_dim"]
            )
            assert isinstance(prior_gmm.weights, Float[Array, "num_components"])

        # Localization
        state_dim = emperical_covariance.shape[0]
        i_indices = jnp.arange(state_dim)[:, None]
        j_indices = jnp.arange(state_dim)[None, :]
        distances = jnp.abs(i_indices - j_indices)

        L = 3.0  # or 4.0
        rho = jnp.exp(-(distances**2) / (2 * L**2))

        emperical_covariance = emperical_covariance * rho

        mixture_covariance = (silverman_bandwidth / estimated_cardinality) * emperical_covariance

        if self.debug:
            assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])
            assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

        def process_measurement(measurement):
            return jax.vmap(self.update_point, in_axes=(0, None, None, None))(
                prior_gmm.means,
                mixture_covariance,
                measurement,
                measurement_system,
            )

        posterior_ensemble, posterior_covariances, logposterior_weights = jax.vmap(
            process_measurement
        )(measurements)



        # log_numerator = jnp.log(self.detection_probability) + jnp.log(prior_gmm.weights) + logposterior_weights
        # log_sum_detection = jax.scipy.special.logsumexp(log_numerator, axis=1)
        # log_denominator = jnp.log(self.lambda_c * self.clutter_density + jnp.exp(log_sum_detection))
        # posterior_weights = jnp.exp(log_numerator - log_denominator)
        # Input shapes:
        # logposterior_weights: [num_measurements, max_gaussian_components]  
        # prior_weights: [max_gaussian_components]

        # Step 1: Calculate log numerators
        log_numerator = (jnp.log(self.detection_probability) + jnp.log(prior_weights)[None, :] + 
                        logposterior_weights)
        # Shape: [num_measurements, max_gaussian_components]

        # Step 2: Calculate log sum for each measurement (denominator sum term)
        log_sum_detection = jax.scipy.special.logsumexp(log_numerator, axis=1)
        # Shape: [num_measurements]

        # Step 3: Calculate log denominators for each measurement
        log_denominator = jnp.log(self.lambda_c * self.clutter_density + jnp.exp(log_sum_detection))
        # Shape: [num_measurements]

        # Step 4: Calculate normalized detection weights
        posterior_weights = jnp.exp(log_numerator - log_denominator[:, None])
        # Shape: [num_measurements, max_gaussian_components]

        detection_gmm = GMM(
            posterior_ensemble.reshape(-1, posterior_ensemble.shape[-1]),
            posterior_covariances.reshape(-1, *posterior_covariances.shape[-2:]),
            posterior_weights.flatten(),
        )
        
        # (1 - p_d) * v(x_k)
        missed_gmm = GMM(
            prior_gmm.means,
            prior_gmm.covs,
            (1 - self.detection_probability) * prior_gmm.weights
        )


        # v(x_k | Z_k) = (1 - p_d) * v(x_k) + \sum_{z \in Z_k} \sum_{i = 1}^{J_k}
        
        intensity_function = GMM(
            jnp.concatenate([missed_gmm.means, detection_gmm.means]),
            jnp.concatenate([missed_gmm.covs, detection_gmm.covs]),
            jnp.concatenate([missed_gmm.weights, detection_gmm.weights]),
        )

        ### Sampling 
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

# import equinox as eqx
# import jax
# import jax.numpy as jnp
# import jax.scipy as jsp
# from beartype import beartype as typechecker
# from jaxtyping import Array, Float, Key, jaxtyped

# from data_science_utils.filters import AbstractFilter
# from data_science_utils.measurement_systems import AbstractMeasurementSystem
# from data_science_utils.statistics import GMM


# @jaxtyped(typechecker=typechecker)
# @eqx.filter_jit
# def dual_gmm_sample(
#     key: Key[Array, ""],
#     w1: Float[Array, "n1"],
#     m1: Float[Array, "n1 state_dim"],
#     P1: Float[Array, "n1 state_dim state_dim"],
#     w2: Float[Array, "n2"],
#     m2: Float[Array, "n2 state_dim"],
#     P2: Float[Array, "n2 state_dim state_dim"],
#     num_samples: int,
# ) -> tuple[
#     Float[Array, "num_samples state_dim"],
#     Float[Array, "num_samples"],
#     Float[Array, "num_samples state_dim state_dim"],
# ]:
#     """
#     Paper: Algorithm 2 - Sampling from two Gaussian mixtures
#     Matlab: gen_gms_kdesilv_dual (Support/gen_gms_kdesilv_dual.m:8-39)
#     Expected: Returns num_samples points with uniform weights summing to W1+W2
#     """
#     # Paper: Step 1 - Compute total weights
#     # Matlab: Lines 9-11: W1 = sum(w1); W2 = sum(w2); Z = W1 + W2;
#     # Expected shape: W1, W2, Z are scalars
#     W1 = jnp.sum(w1)
#     W2 = jnp.sum(w2)
#     Z = W1 + W2

#     # Paper: Step 2 - Sample from combined distribution
#     # Matlab: Lines 19-32: Loop sampling from either mixture based on W1/Z threshold
#     # Expected shape: samples (num_samples, state_dim)
#     keys = jax.random.split(key, num_samples + 1)

#     def sample_one(key_i):
#         u_key, component_key, sample_key = jax.random.split(key_i, 3)
#         u = jax.random.uniform(u_key)

#         # Select from first or second mixture
#         def sample_from_first():
#             idx = jax.random.choice(component_key, w1.shape[0], p=w1 / W1)
#             return jax.random.multivariate_normal(sample_key, m1[idx], P1[idx])

#         def sample_from_second():
#             idx = jax.random.choice(component_key, w2.shape[0], p=w2 / W2)
#             return jax.random.multivariate_normal(sample_key, m2[idx], P2[idx])

#         return jax.lax.cond(u < W1 / Z, sample_from_first, sample_from_second)

#     samples = jax.vmap(sample_one)(keys[:-1])

#     # Paper: Eq. 13 - Silverman's bandwidth
#     # Matlab: Line 36: betaS = (betaS_scale/ceil(sum(w_out)))*(4/(num_par*(model.x_dim+2)))^(2/(model.x_dim+4))
#     # Expected: scalar bandwidth parameter
#     state_dim = m1.shape[1]
#     beta_silv = (1 / jnp.ceil(Z)) * (4 / (num_samples * (state_dim + 2))) ** (
#         2 / (state_dim + 4)
#     )

#     # Paper: Eq. 11 - KDE covariance
#     # Matlab: Line 37: P = betaS * ((ex * ex') / (num_par-1))
#     # Expected shape: (state_dim, state_dim)
#     sample_mean = jnp.mean(samples, axis=0)
#     centered = samples - sample_mean
#     kde_cov = beta_silv * (centered.T @ centered) / (num_samples - 1)

#     # Paper: Uniform weights
#     # Matlab: Line 34: w_out = Z * ones(num_par,1) ./ num_par
#     # Expected shape: (num_samples,) with each weight = Z/num_samples
#     weights = jnp.full(num_samples, Z / num_samples)
#     covs = jnp.tile(kde_cov[None, :, :], (num_samples, 1, 1))

#     return samples, weights, covs


# class EnGMPHD(eqx.Module, strict=True):
#     debug: bool = False
#     clutter_density: float = 6.25e-8  # Paper: uniform clutter density
#     lambda_c: float = 10.0  # Paper: λ_c Poisson rate
#     detection_probability: float = 0.98  # Paper: P_D
#     survival_probability: float = 0.99  # Paper: P_S

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
#         """
#         Paper: Eq. 14-22 - EKF update for single component
#         Matlab: update function (lines 147-171)
#         Expected: Returns updated state, weight update term, and covariance
#         """
#         # Paper: h(x) measurement function
#         # Matlab: Line 154: ybar = cfig.h(model,m(:,jj),'noiseless')
#         # Expected shape: (measurement_dim,)
#         ybar = measurement_device(point[:3])

#         # Paper: Eq. 21 - Measurement Jacobian
#         # Matlab: Line 155: Hj = cfig.H(model,m(:,jj))
#         # Expected shape: (measurement_dim, state_dim)
#         H = jax.jacfwd(lambda x: measurement_device(x[:3]))(point)

#         # Paper: Eq. 19 - Innovation covariance
#         # Matlab: Line 158: Pyyj = Hj*Pxxj*Hj' + R
#         # Expected shape: (measurement_dim, measurement_dim)
#         S = H @ prior_mixture_covariance @ H.T + measurement_device_covariance
#         S = (S + S.T) / 2  # Matlab: Line 159 - symmetrize

#         # Paper: Eq. 18 - Kalman gain
#         # Matlab: Line 161: Kj = Pxyj * iPyyj
#         # Expected shape: (state_dim, measurement_dim)
#         K = prior_mixture_covariance @ H.T @ jnp.linalg.inv(S)

#         # Paper: Eq. 16 - State update
#         # Matlab: Line 162: m_update(:,jj) = m(:,jj) + Kj*(y-ybar)
#         # Expected shape: (state_dim,)
#         updated_point = point + K @ (measurement - ybar)

#         # Paper: Eq. 17 - Covariance update (Joseph form)
#         # Matlab: Line 164: P_update(:,:,jj) = (Ij - Kj*Hj) * Pxxj * (Ij - Kj*Hj)' + Kj * R * Kj'
#         # Expected shape: (state_dim, state_dim)
#         I = jnp.eye(point.shape[0])
#         updated_cov = (I - K @ H) @ prior_mixture_covariance @ (
#             I - K @ H
#         ).T + K @ measurement_device_covariance @ K.T

#         # Paper: Eq. 22 - Weight update term (Gaussian likelihood)
#         # Matlab: Lines 167-169: U_update(jj) = -(y-ybar)' * (iPyyj * (y-ybar)) / 2 - log(det_Pyyj) / 2 - log(2*pi) * size(y,1) / 2
#         # Expected shape: scalar
#         innovation = measurement - ybar
#         log_likelihood = jsp.stats.multivariate_normal.logpdf(measurement, ybar, S)

#         return updated_point, log_likelihood, updated_cov

#     @jaxtyped(typechecker=typechecker)
#     @eqx.filter_jit
#     def update(
#         self,
#         key: Key[Array, ""],
#         prior_gmm: GMM,
#         measurements: Float[Array, "num_measurements measurement_dim"],
#         measurement_system: AbstractMeasurementSystem,
#     ) -> GMM:
#         """
#         Paper: Section 3.1 - EnGM-PHD Filter Recursion
#         Matlab: run_filter main loop (lines 41-139)
#         Expected: Returns posterior GMM with 250 components
#         """
#         # Paper: Section 3.1 - Extract prior parameters
#         # Matlab: Prior is stored as w_predict, m_predict, P_predict
#         # Expected shapes: means (J_k, 6), covs (J_k, 6, 6), weights (J_k,)
#         prior_weights = prior_gmm.weights
#         prior_means = prior_gmm.means
#         J_k = prior_means.shape[0]
#         state_dim = prior_means.shape[1]

#         # Paper: Eq. 11-13 - Compute KDE parameters
#         # Matlab: Lines 117-118: betaS calculation and covariance computation
#         # Expected: bandwidth scalar, empirical covariance (state_dim, state_dim)
#         estimated_cardinality = jnp.sum(prior_weights)
#         bandwidth = (1 / jnp.ceil(estimated_cardinality)) * (
#             4 / (J_k * (state_dim + 2))
#         ) ** (2 / (state_dim + 4))

#         # Paper: Sample covariance computation
#         # Matlab: Line 118: mum = mean(m_predict,2); ex = m_predict - mum; Pum = ((ex * ex') / (J_rsp-1))
#         # Expected shape: (state_dim, state_dim)
#         mean_state = (
#             jnp.sum(prior_means * prior_weights[:, None], axis=0)
#             / estimated_cardinality
#         )
#         centered = prior_means - mean_state
#         empirical_cov = (
#             centered.T @ (centered * prior_weights[:, None])
#         ) / estimated_cardinality

#         # Paper: Eq. 20 - Prior mixture covariance
#         # Matlab: Line 119: P_predict = repmat((betaS*Pum)+model.Q,1,1,J_rsp)
#         # Expected shape: (state_dim, state_dim) - same for all components
#         mixture_covariance = bandwidth * empirical_cov  # + process_noise_Q

#         # Paper: Eq. 14 - Missed detection term
#         # Matlab: Lines 68-70: w_update = model.Q_D*w_predict
#         # Expected shape: (J_k,)
#         missed_weights = (1 - self.detection_probability) * prior_weights
#         missed_means = prior_means
#         missed_covs = jnp.tile(mixture_covariance[None, :, :], (J_k, 1, 1))

#         # Paper: Eq. 14-15 - Update for each measurement
#         # Matlab: Lines 73-95: Loop over measurements
#         # Expected: For m measurements, create m*J_k detection components
#         if measurements.shape[0] > 0:
#             # Update all components with all measurements
#             all_updates = jax.vmap(
#                 lambda z: jax.vmap(
#                     self.update_point, in_axes=(0, None, None, None, None)
#                 )(
#                     prior_means,
#                     mixture_covariance,
#                     z,
#                     measurement_system,
#                     measurement_system.covariance,
#                 )
#             )(measurements)

#             updated_means = all_updates[0]  # (m, J_k, state_dim)
#             log_likelihoods = all_updates[1]  # (m, J_k)
#             updated_covs = all_updates[2]  # (m, J_k, state_dim, state_dim)

#             # Paper: Eq. 15 - Weight update with log-sum-exp trick
#             # Matlab: Lines 83-88: Log-sum-exp for numerical stability
#             # Expected shape: detection_weights (m, J_k)
#             log_detection_terms = (
#                 jnp.log(self.detection_probability)
#                 + jnp.log(prior_weights[None, :])
#                 + log_likelihoods
#             )

#             # Paper: Denominator computation
#             # Matlab: Line 86: log_sum_w_temp = max(log_w_temp) + log(sum(exp(log_w_temp - max(log_w_temp))))
#             # Expected: Use log-sum-exp per measurement
#             max_log = jnp.max(log_detection_terms, axis=1, keepdims=True)
#             log_sum = max_log + jnp.log(
#                 jnp.sum(jnp.exp(log_detection_terms - max_log), axis=1, keepdims=True)
#             )
#             log_denom = jnp.log(self.lambda_c * self.clutter_density + jnp.exp(log_sum))

#             # Paper: Final normalized weights
#             # Matlab: Line 88: w_temp = exp(log_w_temp)
#             # Expected shape: (m, J_k) normalized detection weights
#             normalized_detection_weights = jnp.exp(log_detection_terms - log_denom)

#             # Flatten detection components
#             flat_detection_weights = normalized_detection_weights.reshape(-1)
#             flat_detection_means = updated_means.reshape(-1, state_dim)
#             flat_detection_covs = updated_covs.reshape(-1, state_dim, state_dim)
#         else:
#             flat_detection_weights = jnp.array([])
#             flat_detection_means = jnp.zeros((0, state_dim))
#             flat_detection_covs = jnp.zeros((0, state_dim, state_dim))

#         # Paper: Combine missed and detection components
#         # Matlab: Lines 90-95: Concatenate components
#         # Expected: Total components = J_k + m*J_k
#         all_weights = jnp.concatenate([missed_weights, flat_detection_weights])
#         all_means = jnp.concatenate([missed_means, flat_detection_means])
#         all_covs = jnp.concatenate([missed_covs, flat_detection_covs])

#         # Paper: Section 3.2 - Resample to fixed J_k components
#         # Matlab: Line 98: gen_gms_kdesilv for resampling
#         # Expected: Resample to exactly 250 components
#         key, subkey = jax.random.split(key)

#         # Create temporary GMM for sampling
#         temp_gmm = GMM(
#             all_means, all_covs, all_weights, max_components=all_weights.shape[0]
#         )
#         resampled_means = jax.vmap(temp_gmm.sample)(jax.random.split(subkey, 250))

#         # Paper: Eq. 26-27 - Final KDE parameters
#         # Matlab: gen_gms_kdesilv lines 35-37
#         # Expected: Uniform weights summing to estimated cardinality
#         total_weight = jnp.sum(all_weights)
#         final_weights = jnp.full(250, total_weight / 250)

#         # Recompute bandwidth with new ensemble
#         final_bandwidth = (1 / jnp.ceil(total_weight)) * (
#             4 / (250 * (state_dim + 2))
#         ) ** (2 / (state_dim + 4))
#         final_cov = final_bandwidth * jnp.cov(resampled_means.T)
#         final_covs = jnp.tile(final_cov[None, :, :], (250, 1, 1))

