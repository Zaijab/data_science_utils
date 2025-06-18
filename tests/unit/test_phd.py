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
from data_science_utils.filters.phd import EnGMPHD as EnGMPHD
from data_science_utils.measurement_systems import AbstractMeasurementSystem
from data_science_utils.measurement_systems.radar import Radar
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
true_state = jnp.array(
    [[50, 50, 50, 0.5, 0.5, 2], [100, 100, 50, -0.5, -0.5, 2]], dtype=jnp.float64
)

eqx.filter_vmap(system.flow)(0.0, 1.0, true_state)

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
def sample_from_large_gmm(
    means: Float[Array, "num_components state_dim"],
    weights: Float[Array, "num_components"],
    covs: Float[Array, "num_components state_dim state_dim"],
    key: Key[Array, ""],
    target_components: int = 2,
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


clutter_region = jnp.array([[0.0, 200.0], [0.0, 200.0], [0.0, 400.0]])
clutter_average_rate = 10.0
clutter_max_points = 40


range_std = 1.0
angle_std = 0.5 * jnp.pi / 180
R = jnp.diag(jnp.array([range_std**2, angle_std**2, angle_std**2]))
measurement_system = Radar(R)
stochastic_filter = EnGMPHD(debug=True)


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

    est_pos = estimates[:, :3]
    truth_pos = truth[:, :3]

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
        return cutoff

    if n == 0:
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

for _ in range(30):
    print(_, end=": ")
    intensity_function: GMM = merge_gmms(intensity_function, birth_gmms, key)

    # SKIP GATING
    # Generate Clutter
    key, subkey = jax.random.split(key)
    clutter: RFS = poisson_point_process_rectangular_region(
        subkey,
        clutter_average_rate,
        clutter_region,
        clutter_max_points,
    )
    clutter = clutter.state[clutter.mask]
    all_measureables = jnp.concat([true_state[:, :3], clutter])

    # Generate Measurements based on all available information
    key, subkey = jax.random.split(key)
    measurements = eqx.filter_vmap(measurement_system)(
        all_measureables, jax.random.split(subkey, all_measureables.shape[0])
    )
    key, subkey = jax.random.split(key)
    detected = jax.random.bernoulli(subkey, p=0.98, shape=(measurements.shape[0],))
    measurements = measurements[detected]

    # EnGMF Update Equations
    key, subkey = jax.random.split(key)
    intensity_function = stochastic_filter.update(
        subkey,
        intensity_function,
        measurements,
        measurement_system,
    )

    valid_components = intensity_function.weights > 0.5
    estimated_cardinality = jnp.sum(intensity_function.weights)
    print(jnp.sum(valid_components))
    estimated_weights = jnp.where(valid_components, intensity_function.weights, 0.0)
    estimated_states = jnp.where(
        valid_components[:, None], intensity_function.means, 0.0
    )[:, :3]
    # print(estimated_states)

    #     max_targets = 10  # compile-time constant

    #     sorted_indices = jnp.argsort(estimated_weights)[-max_targets:]
    #     final_estimates = estimated_states[sorted_indices]
    #     final_weights = estimated_weights[sorted_indices]
    #     valid_estimates_mask = final_weights > 1e-5

    #     # ospa = compute_ospa_metric(
    #     #     final_estimates[:, :3], true_state.state[:, 3], cutoff=100.0, p=2
    #     # )
    #     distance, localization, cardinality = compute_ospa_components(
    #         final_estimates[:, :3], true_state.state[:, :3]
    #     )
    #     ospa_distance.append(distance)
    #     ospa_localization.append(localization)
    #     ospa_cardinality.append(cardinality)

    key, subkey = jax.random.split(key)
    true_state = eqx.filter_vmap(system.flow)(0.0, 1.0, true_state)

    intensity_function = GMM(
        means=eqx.filter_vmap(system.flow)(0.0, 1.0, intensity_function.means),
        covs=intensity_function.covs,
        weights=0.99 * intensity_function.weights,
    )


# plt.plot(ospa_distance)
# plt.show()
# plt.plot(ospa_localization)
# plt.show()
# plt.plot(ospa_cardinality)
# plt.show()
