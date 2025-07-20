import equinox as eqx
import jax
import jax.numpy as jnp

from data_science_utils.dynamical_systems import CVModel
from data_science_utils.filters.phd import EnGMPHD as EnGMPHD
from data_science_utils.measurement_systems import Radar
from data_science_utils.models.kmeans import kmeans
from data_science_utils.statistics import GMM, merge_gmms

key = jax.random.key(42)
key, subkey = jax.random.split(key)
num_components = 250
tracking_duration = 100
mc_runs = 10
cardinality = jnp.zeros((mc_runs, tracking_duration))
ospa_distances = jnp.zeros((mc_runs, tracking_duration))
ospa_localizations = jnp.zeros((mc_runs, tracking_duration))
ospa_cardinalities = jnp.zeros((mc_runs, tracking_duration))


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

for time in range(tracking_duration):
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

    key, subkey = jax.random.split(key)


    estimates = state_extraction_from_gmm(subkey, intensity_function)


    key, subkey = jax.random.split(key)
    true_state = eqx.filter_vmap(system.flow)(0.0, 1.0, true_state)

    intensity_function = GMM(
        means=eqx.filter_vmap(system.flow)(0.0, 1.0, intensity_function.means),
        covs=intensity_function.covs,
        weights=0.99 * intensity_function.weights,
    )
