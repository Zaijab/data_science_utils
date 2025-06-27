import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import optax
from beartype import beartype as typechecker
from data_science_utils.filters.abc import AbstractFilter
from data_science_utils.filters.evaluate import ospa_metric
from data_science_utils.filters.phd import EnGMPHD as EnGMPHD
from data_science_utils.statistics import (
    GMM, merge_gmms, poisson_point_process_rectangular_region)
from data_science_utils.statistics.random_finite_sets import RFS
from jaxtyping import Array, Bool, Float, Key, jaxtyped

from data_science_utils.dynamical_systems import CVModel, RandomWalk
from data_science_utils.measurement_systems import AbstractMeasurementSystem
from data_science_utils.measurement_systems.radar import Radar
from data_science_utils.models.kmeans import kmeans

key = jax.random.key(42)
key, subkey = jax.random.split(key)
num_components = 250
tracking_duration = 100
mc_runs = 10
cardinality = jnp.zeros((mc_runs, tracking_duration))
ospa_distances = jnp.zeros((mc_runs, tracking_duration))
ospa_localizations = jnp.zeros((mc_runs, tracking_duration))
ospa_cardinalities = jnp.zeros((mc_runs, tracking_duration))

### Claude
# Add this after your existing imports
from mpl_toolkits.mplot3d import Axes3D

# Initialize storage arrays for visualization (add after line 19)
true_trajectories = jnp.zeros((mc_runs, tracking_duration, 2, 6))  # 2 targets, 6-dim state
estimated_states = []  # Store as list for variable-length estimates
clutter_points = []    # Store clutter for visualization


for mc_run in range(mc_runs):
    print(mc_run, ":")
    key, subkey = jax.random.split(key)

    ### Constant Velocity Model - VERIFIED
    system = CVModel(
        position_dimension=3, sampling_period=1.0, ordering="durant"
    )
    true_state = jnp.array(
        [[50, 50, 50, 0.5, 0.5, 2],
         [100, 100, 50, -0.5, -0.5, 2]],
    ) # X0

    ### Initial Intensity Function 
    intensity_function = GMM(
        means=jnp.zeros((num_components, 6)),
        covs=jnp.zeros((num_components, 6, 6)).at[0].set(jnp.eye(6)),
        weights=jnp.zeros(num_components).at[0].set(1e-16),
        max_components=num_components,
    )


    ### Birth Process
    birth_gmm = GMM(
        means=jnp.array([75.0, 75.0, 150.0, 0.0, 0.0, 0.0])[None, ...],
        covs=jnp.diag(jnp.array([50, 50, 50, 5, 5, 5]) ** 2)[None, ...],
        weights=jnp.array(1 / 100)[None, ...],
        max_components=1,
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


    mc_estimates = []
    mc_clutter = []
    
    for time in range(tracking_duration):
        true_trajectories = true_trajectories.at[mc_run, time].set(true_state)
        intensity_function: GMM = merge_gmms(intensity_function, birth_gmm, key, target_components=(num_components + 10))
        
        
    
        if time > -1:
            print("\t", time, end=": ")
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
        mc_clutter.append(clutter)
        
        all_measureables = jnp.concat([true_state[:, :3], clutter])

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
        mc_estimates.append(estimates)

        # print(f"{estimates=}")
        # print(f"{true_state=}")

        
        ospa_distance, ospa_localization, ospa_cardinality = ospa_metric(
            estimates[:, :3], true_state[:, :3]
        )

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

    estimated_states.append(mc_estimates)
    clutter_points.append(mc_clutter)


try:    
    def plot_figure_1_recreation(mc_run_idx=0):
        """Recreate Figure 1 from EnGM-PHD paper showing 3D tracking scenario"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot true trajectories as black lines
        traj = true_trajectories[mc_run_idx]  # Shape: (100, 2, 6)
        ax.plot(traj[:, 0, 0], traj[:, 0, 1], traj[:, 0, 2], 'k-', linewidth=2, label='True Target 1')
        ax.plot(traj[:, 1, 0], traj[:, 1, 1], traj[:, 1, 2], 'k-', linewidth=2, label='True Target 2')

        # Plot estimated states as colored points
        for time_idx, estimates in enumerate(estimated_states[mc_run_idx]):
            if estimates.shape[0] > 0:
                ax.scatter(estimates[:, 0], estimates[:, 1], estimates[:, 2], 
                          c='red', s=20, alpha=0.6)

        # Plot clutter as gray crosses (subsample for clarity)
        for time_idx in range(0, tracking_duration, 5):  # Every 5th timestep
            clutter = clutter_points[mc_run_idx][time_idx]
            if clutter.shape[0] > 0:
                ax.scatter(clutter[:, 0], clutter[:, 1], clutter[:, 2],
                          c='gray', marker='x', s=10, alpha=0.3)

        # Draw surveillance region boundary
        x_bounds, y_bounds, z_bounds = [0, 200], [0, 200], [0, 400]
        vertices = [
            [x_bounds[0], y_bounds[0], z_bounds[0]], [x_bounds[1], y_bounds[0], z_bounds[0]],
            [x_bounds[1], y_bounds[1], z_bounds[0]], [x_bounds[0], y_bounds[1], z_bounds[0]],
            [x_bounds[0], y_bounds[0], z_bounds[1]], [x_bounds[1], y_bounds[0], z_bounds[1]],
            [x_bounds[1], y_bounds[1], z_bounds[1]], [x_bounds[0], y_bounds[1], z_bounds[1]]
        ]

        # Draw wireframe box for surveillance region
        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        for edge in edges:
            points = jnp.array([vertices[edge[0]], vertices[edge[1]]])
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'r--', alpha=0.5)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position') 
        ax.set_zlabel('Z Position')
        ax.set_title('EnGM-PHD Filter: 3D Multi-Target Tracking with Clutter')
        ax.legend()
        plt.tight_layout()
        plt.show()

    # Call the visualization
    plot_figure_1_recreation(0)

    plt.title("OSPA Distance")
    plt.plot(jnp.mean(ospa_distances, axis=0))
    plt.show()
    plt.title("OSPA Localization")
    plt.plot(jnp.mean(ospa_localizations, axis=0))
    plt.show()
    plt.title("OSPA Cardinality")
    plt.plot(jnp.mean(ospa_cardinalities, axis=0))
    plt.show()
    plt.title("Cardinality")
    plt.plot(jnp.mean(cardinality, axis=0))
    plt.show()
except:
    pass
