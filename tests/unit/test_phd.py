import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Key, jaxtyped
from beartype import beartype as typechecker
from data_science_utils.dynamical_systems import RandomWalk, CVModel
from data_science_utils.filters.abc import AbstractFilter
from data_science_utils.measurement_systems import AbstractMeasurementSystem
from data_science_utils.statistics import GMM

import abc

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
def merge_gmms(
    gmm1: GMM, gmm2: GMM, key: Key[Array, ""], target_components: int = 250
) -> GMM:
    target_components = min(
        target_components, gmm1.weights.shape[0] + gmm2.weights.shape[0]
    )
    # Compute total weights
    W1 = jnp.sum(gmm1.weights)
    W2 = jnp.sum(gmm2.weights)
    Z = W1 + W2

    # Sample target_components points using Algorithm 2 logic
    uniform_keys = jax.random.split(key, target_components)

    def sample_one(key_i):
        u = jax.random.uniform(key_i)
        return jax.lax.cond(
            u < W1 / Z, lambda: gmm1.sample(key_i), lambda: gmm2.sample(key_i)
        )

    samples = jax.vmap(sample_one)(uniform_keys)

    # Reconstruct GMM with uniform weights
    uniform_weights = jnp.full(
        target_components, jnp.array([W1 + W2]) / target_components
    )
    spatial_dimension = gmm1.means.shape[1]
    silverman_beta = (
        ((4) / (spatial_dimension + 2)) ** (2 / (spatial_dimension + 4))
    ) * ((target_components) ** (-(2) / (spatial_dimension + 4)))
    covariance = (silverman_beta / Z) * jnp.cov(samples.T)

    return GMM(
        samples, jnp.tile(covariance, (target_components, 1, 1)), uniform_weights
    )


from data_science_utils.statistics.poisson_point_process import (
    poisson_point_process_hyperrectangle,
)

clutter_region = jnp.array([[0.0, 200.0], [0.0, 200.0], [0.0, 400.0]])
clutter_average_rate = 10.0
clutter_max_points = 40

for _ in range(3):
    # Births
    # Add 10 more Gaussian Terms
    intensity_function = merge_gmms(intensity_function, birth_gmms, key)

    # SKIP GATING
    # Generate Clutter
    clutter, clutter_mask = poisson_point_process_hyperrectangle(
        subkey,
        clutter_average_rate,
        clutter_region,
        clutter_max_points,
    )

    print(clutter[clutter_mask])

    # Detect Everything

    # Update
    #

    key, subkey = jax.random.split(key)
    true_state = eqx.filter_vmap(system.flow)(0.0, 1.0, true_state)
