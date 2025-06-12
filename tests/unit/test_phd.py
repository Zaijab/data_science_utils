import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import optax
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Key, jaxtyped

from data_science_utils.dynamical_systems import CVModel, RandomWalk
from data_science_utils.filters.abc import AbstractFilter
from data_science_utils.measurement_systems import AbstractMeasurementSystem
from data_science_utils.statistics import (
    GMM,
    merge_gmms,
    poisson_point_process_rectangular_region,
)
from data_science_utils.statistics.random_finite_sets import RFS

key = jax.random.key(0)
key, subkey = jax.random.split(key)


system = CVModel(
    position_dimension=3, sampling_period=1.0, process_noise_std=5.0, ordering="durant"
)
true_state = RFS(
    state=jnp.array(
        [[50, 50, 50, 0.5, 0.5, 2], [100, 100, 50, -0.5, -0.5, 2]], dtype=jnp.float64
    ),
    mask=jnp.array([True, True]),
)
eqx.filter_vmap(system.flow)(0.0, 1.0, true_state.state)

intensity_function = GMM(
    means=jnp.zeros((250, 6)),
    covs=jnp.tile(jnp.eye(6), (250, 1, 1)),
    weights=jnp.tile(1e-16, (250)),
    max_components=250,
)

birth_means = jnp.array([75, 75, 150, 0, 0, 0], dtype=jnp.float64)
birth_covs = jnp.diag(jnp.array([50, 50, 50, 5, 5, 5]) ** 2)
birth_weights = jnp.array(1 / 100)

birth_gmms = GMM(
    means=jnp.tile(birth_means, (10, 1)),
    covs=jnp.tile(birth_covs, (10, 1, 1)),
    weights=jnp.tile(birth_weights, (10)),
    max_components=10,
)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def sample_from_multiple_gmms(
    means: Float[Array, "num_gmms num_components state_dim"],
    weights: Float[Array, "num_gmms num_components"],
    covs: Float[Array, "num_gmms num_components state_dim state_dim"],
    valid_mask: Bool[Array, "num_gmms"],
    key: Key[Array, ""],
    target_components: int = 250,
) -> GMM:
    # Flatten valid GMMs into single arrays
    valid_means = means[valid_mask].reshape(-1, means.shape[-1])
    valid_weights = weights[valid_mask].reshape(-1)
    valid_covs = covs[valid_mask].reshape(-1, covs.shape[-2], covs.shape[-1])

    # Normalize weights
    total_weight = jnp.sum(valid_weights)
    probs = valid_weights / total_weight

    # Sample components according to weights
    uniform_keys = jax.random.split(key, target_components)
    component_indices = jax.vmap(
        lambda k: jax.random.choice(k, valid_means.shape[0], p=probs)
    )(uniform_keys)

    # Sample from selected components
    samples = jax.vmap(
        lambda k, idx: jax.random.multivariate_normal(
            k, valid_means[idx], valid_covs[idx]
        )
    )(uniform_keys, component_indices)

    # Reconstruct with KDE
    uniform_weights = jnp.full(target_components, total_weight / target_components)
    silverman_beta = ((4 / (means.shape[-1] + 2)) ** (2 / (means.shape[-1] + 4))) * (
        target_components ** (-2 / (means.shape[-1] + 4))
    )
    covariance = silverman_beta * jnp.cov(samples.T)

    return GMM(
        samples, jnp.tile(covariance, (target_components, 1, 1)), uniform_weights
    )


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def sample_from_multiple_gmms(
    means: Float[Array, "num_gmms num_components state_dim"],
    weights: Float[Array, "num_gmms num_components"],
    covs: Float[Array, "num_gmms num_components state_dim state_dim"],
    valid_mask: Bool[Array, "num_gmms"],
    key: Key[Array, ""],
    target_components: int = 250,
) -> GMM:
    # Flatten all arrays
    flat_means = means.reshape(-1, means.shape[-1])
    flat_weights = weights.reshape(-1)
    flat_covs = covs.reshape(-1, covs.shape[-2], covs.shape[-1])

    # Expand mask to match flattened shape and zero out invalid weights
    expanded_mask = jnp.repeat(valid_mask, weights.shape[1])
    masked_weights = jnp.where(expanded_mask, flat_weights, 0.0)

    # Normalize weights
    total_weight = jnp.sum(masked_weights)
    probs = masked_weights / jnp.maximum(total_weight, 1e-10)

    # Sample from mixture
    uniform_keys = jax.random.split(key, target_components)
    indices = jax.vmap(lambda k: jax.random.choice(k, flat_means.shape[0], p=probs))(
        uniform_keys
    )
    samples = jax.vmap(
        lambda k, i: jax.random.multivariate_normal(k, flat_means[i], flat_covs[i])
    )(uniform_keys, indices)

    # Apply KDE
    uniform_weights = jnp.full(target_components, total_weight / target_components)
    beta = ((4 / (means.shape[-1] + 2)) ** (2 / (means.shape[-1] + 4))) * (
        target_components ** (-2 / (means.shape[-1] + 4))
    )
    cov = beta * jnp.cov(samples.T)

    return GMM(samples, jnp.tile(cov, (target_components, 1, 1)), uniform_weights)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def sample_from_large_gmm(
    means: Float[Array, "num_components state_dim"],
    weights: Float[Array, "num_components"],
    covs: Float[Array, "num_components state_dim state_dim"],
    key: Key[Array, ""],
    target_components: int = 250,
) -> GMM:
    # Normalize weights
    probs = weights / jnp.sum(weights)

    # Sample component indices
    keys = jax.random.split(key, target_components)
    indices = jax.vmap(lambda k: jax.random.choice(k, means.shape[0], p=probs))(keys)

    # Sample from selected components
    samples = jax.vmap(
        lambda k, i: jax.random.multivariate_normal(k, means[i], covs[i])
    )(keys, indices)

    # Apply KDE
    beta = ((4 / (means.shape[-1] + 2)) ** (2 / (means.shape[-1] + 4))) * (
        target_components ** (-2 / (means.shape[-1] + 4))
    )
    kde_cov = beta * jnp.cov(samples.T)

    return GMM(
        samples,
        jnp.tile(kde_cov, (target_components, 1, 1)),
        jnp.full(target_components, jnp.sum(weights) / target_components),
    )


@jax.jit
@jax.vmap
def sample_gaussian_mixture(key, point, cov):
    return jax.random.multivariate_normal(key, mean=point, cov=cov)


class EnGMPHD(eqx.Module, strict=True):
    debug: bool = False
    sampling_function: jax.tree_util.Partial = jax.tree_util.Partial(
        sample_gaussian_mixture
    )

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
            measurement, _ = measurement_device(
                RFS(jnp.atleast_2d(point[:3]), jnp.tile(True, 1))
            )
            return measurement

        ybar = measurement_function(point)

        measurement_jacobian = jax.jacfwd(measurement_function)(point)[0, ...]

        if self.debug:
            assert isinstance(
                measurement_jacobian, Float[Array, "measurement_dim state_dim"]
            )

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
        # I_minus_KH = jnp.eye(point.shape[0]) - kalman_gain @ measurement_jacobian
        # gaussian_mixture_covariance = (
        #     I_minus_KH @ prior_mixture_covariance @ I_minus_KH.T
        #     + kalman_gain @ measurement_device_covariance @ kalman_gain.T
        # )
        # gaussian_mixture_covariance = (
        #     gaussian_mixture_covariance + gaussian_mixture_covariance.T
        # ) / 2

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
        measurements_mask: Bool[Array, "num_measurements"],
        measurement_system: AbstractMeasurementSystem,
        clutter_density: float,
        detection_probability: float,
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
            jax.debug.callback(is_positive_definite, emperical_covariance)
            assert isinstance(emperical_covariance, Float[Array, "state_dim state_dim"])
            assert isinstance(mixture_covariance, Float[Array, "state_dim state_dim"])

        def process_measurement(measurement, mask):
            return jax.lax.cond(
                mask,
                lambda: jax.vmap(
                    self.update_point, in_axes=(0, None, None, None, None)
                )(
                    prior_gmm.means,
                    mixture_covariance,
                    measurement,
                    measurement_system,
                    measurement_system.covariance,
                ),
                lambda: (
                    jnp.zeros_like(prior_gmm.means),
                    jnp.full((prior_gmm.weights.shape[0]), jnp.asarray(0.0)),
                    jnp.zeros_like(prior_gmm.covs),
                ),
            )

        posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(
            process_measurement
        )(measurements, measurements_mask)

        print(posterior_ensemble)

        logposterior_weights = logposterior_weights[:, :]
        print(logposterior_weights)

        # PHD update formula: log(P_D * w_prior * likelihood)
        log_detection_terms = (
            jnp.log(detection_probability)
            + jnp.log(prior_weights)[None, :]
            + logposterior_weights
        )

        # Denominator: log(clutter + sum(detection_terms)) per measurement
        log_sum_detection = jax.scipy.special.logsumexp(
            log_detection_terms, axis=1, keepdims=True
        )
        log_denominators = jnp.log(clutter_density + jnp.exp(log_sum_detection))

        # Final normalized weights
        normalized_weights = jnp.where(
            measurements_mask[:, None],
            jnp.exp(log_detection_terms - log_denominators),
            0.0,
        )
        print(normalized_weights)

        # Scale Weights
        # m = jnp.max(logposterior_weights)
        # g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
        # posterior_weights = jnp.exp(logposterior_weights - g)
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
        missed_weights = (1 - detection_probability) * prior_weights
        missed_means = prior_gmm.means
        missed_covs = prior_gmm.covs

        # Detection (flatten your computed arrays)
        flat_detection_weights = normalized_weights.reshape(-1)
        flat_detection_means = posterior_ensemble.reshape(
            -1, posterior_ensemble.shape[-1]
        )
        flat_detection_covs = posterior_covariances.reshape(
            -1, *posterior_covariances.shape[-2:]
        )

        # Concatenate
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


clutter_region = jnp.array([[0.0, 200.0], [0.0, 200.0], [0.0, 400.0]])
clutter_average_rate = 10.0
clutter_max_points = 40


class Radar(AbstractMeasurementSystem, strict=True):
    covariance: Float[Array, "..."]

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def __call__(
        self,
        positions: RFS,
        key: Key[Array, ""] | None = None,
    ) -> tuple[Float[Array, "max_objects 3"], Bool[Array, "max_objects"]]:
        x, y, z = positions.state[:, 0], positions.state[:, 1], positions.state[:, 2]
        rho = jnp.sqrt(x**2 + y**2 + z**2)
        alpha = jnp.arctan2(y, x)
        epsilon = jnp.arcsin(z / rho)

        measurements = jnp.stack([rho, alpha, epsilon], axis=1)

        if key is None:
            return measurements, jnp.tile(jnp.asarray(True), measurements.shape[0])

        if key is not None:
            noise_key, detection_key = jax.random.split(key)
            noise_std = jnp.array([1.0, jnp.deg2rad(0.5), jnp.deg2rad(0.5)])
            measurements += jax.random.normal(noise_key, measurements.shape) * noise_std

            detected = jax.random.bernoulli(
                detection_key, p=0.98, shape=(positions.state.shape[0],)
            )
            detected = detected & positions.mask  # Only detect valid positions
            return measurements, detected


range_std = 1.0
angle_std = 0.5 * jnp.pi / 180  # 0.5 degrees in radians
R = jnp.diag(jnp.array([range_std**2, angle_std**2, angle_std**2]))
measurement_system = Radar(R)

stochastic_filter = EnGMPHD()


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def union(rfs1: RFS, rfs2: RFS) -> RFS:
    """
    Union of two RFS objects by concatenating their state and mask arrays.

    Args:
        rfs1: First RFS
        rfs2: Second RFS

    Returns:
        New RFS with combined states and masks
    """
    assert rfs1.state.shape[1] == rfs2.state.shape[1], "State dimensions must match"

    combined_state = jnp.concatenate([rfs1.state, rfs2.state], axis=0)
    combined_mask = jnp.concatenate([rfs1.mask, rfs2.mask], axis=0)

    assert combined_state.shape[0] == rfs1.state.shape[0] + rfs2.state.shape[0]
    assert combined_mask.shape[0] == rfs1.mask.shape[0] + rfs2.mask.shape[0]

    return RFS(combined_state, combined_mask)


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def compute_ospa_cost_matrix(
    estimates: Float[Array, "m 3"],
    truth: Float[Array, "n 3"],
    cutoff: float = 100.0,
    p: int = 2,
) -> Float[Array, "m n"]:
    """Compute cost matrix for OSPA metric using Euclidean distance on positions."""

    # Extract positions (first 3 dimensions)
    est_pos = estimates[:, :3]  # Shape: (m, 3)
    truth_pos = truth[:, :3]  # Shape: (n, 3)

    assert est_pos.shape[1] == 3
    assert truth_pos.shape[1] == 3

    # Compute pairwise distances
    diff = est_pos[:, None, :] - truth_pos[None, :, :]  # Shape: (m, n, 3)
    distances = jnp.linalg.norm(diff, axis=2)  # Shape: (m, n)

    # Apply cutoff and power
    costs = jnp.minimum(distances, cutoff) ** p

    assert costs.shape == (estimates.shape[0], truth.shape[0])
    return costs


@jaxtyped(typechecker=typechecker)
def compute_ospa_components(estimates, truth, cutoff=100.0, p=2):
    m, n = estimates.shape[0], truth.shape[0]
    max_card = jnp.maximum(m, n)

    if m == 0 and n == 0:
        return 0.0

    if m == 0:
        # No estimates, all truth states contribute cutoff cost
        return cutoff

    if n == 0:
        # No truth states, all estimates contribute cutoff cost
        return cutoff

    cost_matrix = compute_ospa_cost_matrix(estimates, truth, cutoff, p)
    row_indices, col_indices = optax.assignment.hungarian_algorithm(cost_matrix)
    assignment_cost = cost_matrix[row_indices, col_indices].sum()
    localization = (
        (assignment_cost / jnp.minimum(m, n)) ** (1 / p)
        if jnp.minimum(m, n) > 0
        else 0.0
    )
    cardinality = cutoff * jnp.abs(m - n) / jnp.maximum(m, n)
    total = ((assignment_cost + cutoff**p * jnp.abs(m - n)) / jnp.maximum(m, n)) ** (
        1 / p
    )
    return total, localization, cardinality


ospa_distance = []
ospa_localization = []
ospa_cardinality = []

for _ in range(50):
    print(_, end=": ")
    # Births
    # Add 10 more Gaussian Terms
    intensity_function: GMM = merge_gmms(intensity_function, birth_gmms, key)

    # SKIP GATING
    # Generate Clutter
    clutter: RFS = poisson_point_process_rectangular_region(
        subkey,
        clutter_average_rate,
        clutter_region,
        clutter_max_points,
    )
    true_state_position = RFS(true_state.state[:, :3], true_state.mask)

    all_measureables = union(true_state_position, clutter)

    # Generate Measurements based on all available information
    measurements, measurements_mask = measurement_system(all_measureables, key)

    # EnGMF Update Equations
    key, subkey = jax.random.split(key)
    intensity_function = stochastic_filter.update(
        subkey,
        intensity_function,
        measurements,
        measurements_mask,
        measurement_system,
        6.25e-8,
        0.98,
    )
    print(jnp.sort(intensity_function.weights)[:20])

    valid_components = intensity_function.weights > 1e-20
    estimated_cardinality = jnp.floor(
        jnp.sum(jnp.where(valid_components, intensity_function.weights, 0))
    )
    print(estimated_cardinality)
    estimated_weights = jnp.where(valid_components, intensity_function.weights, 0.0)
    estimated_states = jnp.where(
        valid_components[:, None], intensity_function.means, 0.0
    )[:, :3]
    # print(estimated_states)

    max_targets = 10  # compile-time constant

    sorted_indices = jnp.argsort(estimated_weights)[-max_targets:]
    final_estimates = estimated_states[sorted_indices]
    final_weights = estimated_weights[sorted_indices]
    valid_estimates_mask = final_weights > 1e-5

    # ospa = compute_ospa_metric(
    #     final_estimates[:, :3], true_state.state[:, 3], cutoff=100.0, p=2
    # )
    distance, localization, cardinality = compute_ospa_components(
        final_estimates[:, :3], true_state.state[:, :3]
    )
    ospa_distance.append(distance)
    ospa_localization.append(localization)
    ospa_cardinality.append(cardinality)

    key, subkey = jax.random.split(key)
    true_state = RFS(
        eqx.filter_vmap(system.flow)(0.0, 1.0, true_state.state),
        mask=jnp.array([True, True]),
    )

    intensity_function = GMM(
        means=eqx.filter_vmap(system.flow)(0.0, 1.0, intensity_function.means),
        covs=intensity_function.covs,
        weights=0.99 * intensity_function.weights,
    )


plt.plot(ospa_distance)
plt.show()
plt.plot(ospa_localization)
plt.show()
plt.plot(ospa_cardinality)
plt.show()
