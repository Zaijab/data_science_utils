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
from data_science_utils.models.kmeans import kmeans

key = jax.random.key(42)
key, subkey = jax.random.split(key)
num_components = 250


tracking_duration = 500
mc_runs = 10
cardinality = jnp.zeros((mc_runs, tracking_duration))
ospa_distances = jnp.zeros((mc_runs, tracking_duration))
ospa_localizations = jnp.zeros((mc_runs, tracking_duration))
ospa_cardinalities = jnp.zeros((mc_runs, tracking_duration))

for mc_run in range(mc_runs):
    print(mc_run, ":")
    key, subkey = jax.random.split(key)

    ### Constant Velocity Model - VERIFIED
    system = CVModel(
        position_dimension=3, sampling_period=1.0, ordering="durant"
    )
    true_state = jnp.array(
        [[50, 50, 50, 0.5, 0.5, 2], [100, 100, 50, -0.5, -0.5, 2]], dtype=jnp.float64
    )

    ### Initial Intensity Function 
    intensity_function = GMM(
        means=jnp.zeros((num_components, 6)),
        covs=jnp.zeros((num_components, 6, 6)).at[0].set(jnp.eye(6)),
        weights=jnp.zeros(num_components).at[0].set(1e-16),
        max_components=num_components,
    )


    ### Birth Process
    birth_means = jnp.array([75, 75, 150, 0, 0, 0], dtype=jnp.float64)
    birth_covs = jnp.diag(jnp.array([50, 50, 50, 5, 5, 5]) ** 2)
    birth_weights = jnp.array(1 / 100)

    birth_gmms = GMM(
        means=jnp.tile(birth_means, (10, 1)),
        covs=jnp.tile(birth_covs, (10, 1, 1)),
        weights=jnp.tile(birth_weights, (10)),
        max_components=10,
    )


    ### Clutter Generation
    clutter_region = jnp.array([[0.0, 200.0], [0.0, 200.0], [0.0, 400.0]])
    clutter_average_rate = 10.0
    clutter_max_points = 100

    ### Measurement System
    range_std = 1.0
    angle_std = 0.5 * jnp.pi / 180
    R = jnp.diag(jnp.array([range_std**2, angle_std**2, angle_std**2]))
    measurement_system = Radar(R)


    ### Filtering System
    stochastic_filter = EnGMPHD(debug=True)


    ### State Extraction
    def state_extraction_from_gmm(key, gmm: GMM):
        if jnp.sum(intensity_function.weights) > 0.5:
            points = intensity_function.means
            estimated_cardinality = int((jnp.sum(intensity_function.weights)))
            return kmeans(key, points, estimated_cardinality).centroids
        else:
            return jnp.zeros((0, gmm.means.shape[1]))


    
    for time in range(tracking_duration):
        print("\t", time, end=": ")
        intensity_function: GMM = merge_gmms(intensity_function, birth_gmms, key, target_components=num_components)
        print(f"Total weight: {jnp.sum(intensity_function.weights):.3f}")


        # Generate Clutter
        key, subkey = jax.random.split(key)
        n_points = jax.random.poisson(subkey, lam=clutter_average_rate)
        key, subkey = jax.random.split(key)
        uniform_samples = jax.random.uniform(
            subkey, shape=(n_points, 3), minval=0.0, maxval=1.0
        )
        widths = clutter_region[:, 1] - clutter_region[:, 0]
        clutter = clutter_region[:, 0] + uniform_samples * widths[None, :]
        
        all_measureables = jnp.concat([true_state[:, :3], clutter])
        # all_measureables = true_state[:, :3]
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
        
        key, subkey = jax.random.split(key)
        
        estimates = state_extraction_from_gmm(subkey, intensity_function)
        
        print(f"{estimates=}")
        print(f"{true_state=}")
        
        ospa_distance, ospa_localization, ospa_cardinality = ospa_metric(
            estimates[:, :3], true_state[:, :3]
        )
        print(ospa_distance)
        print(f"OSPA Distance: {ospa_distance:.6f}")
        print(f"OSPA Localization: {ospa_localization:.6f}") 
        print(f"OSPA Cardinality: {ospa_cardinality:.6f}")


        print("\t", ospa_distance)
        ospa_distances = ospa_distances.at[mc_run, time].set(ospa_distance)
        ospa_localizations = ospa_localizations.at[mc_run, time].set(ospa_localization)
        ospa_cardinalities = ospa_cardinalities.at[mc_run, time].set(ospa_cardinality)

        key, subkey = jax.random.split(key)
        true_state = eqx.filter_vmap(system.flow)(0.0, 1.0, true_state)

        intensity_function = GMM(
            means=eqx.filter_vmap(system.flow)(0.0, 1.0, intensity_function.means),
            covs=intensity_function.covs,
            weights=0.99 * intensity_function.weights,
        )

# if cardinality is not []:
#     time_range = jnp.arange(100)
#     for mc_run in range(cardinality.shape[0]):
#         plt.plot(time_range, cardinality[mc_run])
#     plt.show()

if ospa_distance is not []:
    plt.title("Distance")
    plt.plot(jnp.mean(ospa_distances, axis=0))
    plt.show()
    plt.title("Localization")
    plt.plot(jnp.mean(ospa_localizations, axis=0))
    plt.show()
    plt.title("Cardinality")
    plt.plot(jnp.mean(ospa_cardinalities, axis=0))
    plt.show()
