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
from data_science_utils.filters.evaluate import ospa_metric

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
    weights=jnp.zeros(250).at[0].set(1e-16),
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


clutter_region = jnp.array([[0.0, 200.0], [0.0, 200.0], [0.0, 400.0]])
clutter_average_rate = 10.0
clutter_max_points = 40


range_std = 1.0
angle_std = 0.5 * jnp.pi / 180
R = jnp.diag(jnp.array([range_std**2, angle_std**2, angle_std**2]))
measurement_system = Radar(R)
stochastic_filter = EnGMPHD(debug=True)




@jaxtyped(typechecker=typechecker)
def extract_phd_targets(
    gmm: GMM, threshold: float = 0.5
) -> Float[Array, "num_targets state_dim"]:
    """Extract target states from PHD intensity."""
    # Components with weight > threshold
    valid_mask = gmm.weights > threshold
    valid_means = gmm.means[valid_mask]
    valid_weights = gmm.weights[valid_mask]

    # Number of targets ≈ sum of weights
    n_targets = jnp.round(jnp.sum(gmm.weights)).astype(int)

    # Simple extraction: top n_targets by weight
    sorted_idx = jnp.argsort(valid_weights)[::-1]
    return valid_means[sorted_idx[:n_targets]]

import jax.numpy as jnp


ospa_distance = []
ospa_localization = []
ospa_cardinality = []
cardinality = jnp.zeros((250, 100))

for mc_run in range(20):
    print(mc_run, ":")
    key, subkey = jax.random.split(key)
    
    for time in range(100):
        print("\t", time, end=": ")
        intensity_function: GMM = merge_gmms(intensity_function, birth_gmms, key)
        print(f"Total weight: {jnp.sum(intensity_function.weights):.3f}")

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
        cardinality = cardinality.at[mc_run, time].set(jnp.floor(jnp.sum(intensity_function.weights)))

        # estimates = extract_phd_states(intensity_function)
        # finite_mask = estimates[:, 0] < jnp.inf
        # valid_estimates = estimates[finite_mask]

        # distance, localization, cardinality = ospa_metric(
        #     valid_estimates[:, :3], true_state[:, :3]
        # )

        # ospa_distance.append(distance)
        # ospa_localization.append(localization)
        # ospa_cardinality.append(cardinality)

        key, subkey = jax.random.split(key)
        true_state = eqx.filter_vmap(system.flow)(0.0, 1.0, true_state)

        intensity_function = GMM(
            means=eqx.filter_vmap(system.flow)(0.0, 1.0, intensity_function.means),
            covs=intensity_function.covs,
            weights=0.99 * intensity_function.weights,
        )

if cardinality != []:
    plt.plot(jnp.mean(cardinality, axis=0))
    plt.show()

if ospa_distance is not []:
    plt.title("Distance")
    plt.plot(ospa_distance)
    plt.show()
    plt.title("Localization")
    plt.plot(ospa_localization)
    plt.show()
    plt.title("Cardinality")
    plt.plot(ospa_cardinality)
    plt.show()
