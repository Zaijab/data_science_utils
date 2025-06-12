import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Key, jaxtyped

from data_science_utils.dynamical_systems import CVModel, RandomWalk
from data_science_utils.filters.abc import AbstractFilter
from data_science_utils.measurement_systems import AbstractMeasurementSystem
from data_science_utils.statistics import GMM, merge_gmms
from data_science_utils.statistics import (
    poisson_point_process_rectangular_region,
)


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


class Radar(AbstractMeasurementSystem, strict=True):
    covariance: Float[Array, "..."]

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def __call__(
        self,
        positions: Float[Array, "max_objects 3"],
        positions_mask: Bool[Array, "max_objects"],
        key: Key[Array, ""],
    ) -> tuple[Float[Array, "max_objects 3"], Bool[Array, "max_objects"]]:
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        rho = jnp.sqrt(x**2 + y**2 + z**2)
        alpha = jnp.arctan2(y, x)
        epsilon = jnp.arcsin(z / rho)

        measurements = jnp.stack([rho, alpha, epsilon], axis=1)
        noise_key, detection_key = jax.random.split(key)
        noise_std = jnp.array([1.0, jnp.deg2rad(0.5), jnp.deg2rad(0.5)])
        measurements += jax.random.normal(noise_key, measurements.shape) * noise_std

        detected = jax.random.bernoulli(
            detection_key, p=0.98, shape=(positions.shape[0],)
        )
        detected = detected & positions_mask  # Only detect valid positions
        return measurements, detected


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
        measurement_jacobian = jax.jacfwd(measurement_device)(point)
        if self.debug:
            assert isinstance(
                measurement_jacobian, Float[Array, "measurement_dim state_dim"]
            )

        kalman_gain = (
            prior_mixture_covariance
            @ measurement_jacobian.T
            @ jnp.linalg.inv(
                measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
                + measurement_device_covariance
            )
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

        point = point - (
            kalman_gain @ jnp.atleast_2d(measurement_device(point, None) - measurement)
        ).reshape(-1)
        if self.debug:
            assert isinstance(point, Float[Array, "state_dim"])

        logposterior_weights = jsp.stats.multivariate_normal.logpdf(
            measurement,
            measurement_device(point, None),
            measurement_jacobian @ prior_mixture_covariance @ measurement_jacobian.T
            + measurement_device_covariance,
        )
        if self.debug:
            assert isinstance(logposterior_weights, Float[Array, ""])

        return point, logposterior_weights, gaussian_mixture_covariance

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
        prior_weights = (1 - detection_probability) * prior_weights

        key, subkey, *subkeys = jax.random.split(key, 2 + prior_ensemble.shape[0])
        subkeys = jnp.array(subkeys)

        if self.debug:
            assert isinstance(subkeys, Key[Array, "batch_dim"])

        bandwidth = (
            (4) / (prior_ensemble.shape[0] * (prior_ensemble.shape[-1] + 2))
        ) ** ((2) / (prior_ensemble.shape[-1] + 4))
        emperical_covariance = jnp.cov(prior_ensemble.T)  # + 1e-8 * jnp.eye(2)

        # Gaussian localization with radius L
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
                    jnp.full(prior_gmm.weights.shape, -jnp.inf),
                    jnp.zeros_like(prior_gmm.covs),
                ),
            )

        detection_results = jax.vmap(process_measurement)(
            measurements, measurements_mask
        )

        return detection_results

        posterior_ensemble, logposterior_weights, posterior_covariances = jax.vmap(
            self.update_point,
            in_axes=(0, None, None, None, None),
        )(
            prior_ensemble,
            mixture_covariance,
            measurement,
            measurement_system,
            measurement_system.covariance,
        )

        # Pruning: Remove components where weight < threshold (e.g., 1e-5)
        # Merging: Combine components that are "close" (Mahalanobis distance < threshold)
        # Capping: Keep only the highest-weighted 250 components

        if self.debug:
            assert isinstance(posterior_ensemble, Float[Array, "batch_dim state_dim"])
            assert isinstance(logposterior_weights, Float[Array, "batch_dim"])
            assert isinstance(
                posterior_covariances, Float[Array, "batch_dim state_dim state_dim"]
            )
            jax.self.debug.callback(has_nan, posterior_covariances)

        # Scale Weights
        m = jnp.max(logposterior_weights)
        g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
        posterior_weights = jnp.exp(logposterior_weights - g)
        posterior_weights = posterior_weights / jnp.sum(posterior_weights)

        # Missed Detection and Clutter

        if self.debug:
            assert isinstance(posterior_weights, Float[Array, "batch_dim"])

        # Prevent Degenerate Particles
        variable = jax.random.choice(
            subkey,
            prior_ensemble.shape[0],
            shape=(prior_ensemble.shape[0],),
            p=posterior_weights,
        )
        posterior_ensemble = posterior_ensemble[variable, ...]
        posterior_covariances = posterior_covariances[variable, ...]

        if self.debug:
            jax.self.debug.callback(has_nan, posterior_covariances)

        posterior_samples = self.sampling_function(
            subkeys, posterior_ensemble, posterior_covariances
        )

        if self.debug:
            assert isinstance(posterior_weights, Float[Array, "batch_dim"])

        return posterior_samples


clutter_region = jnp.array([[0.0, 200.0], [0.0, 200.0], [0.0, 400.0]])
clutter_average_rate = 10.0
clutter_max_points = 40
measurement_system = Radar(jnp.zeros(1))

filter = EnGMPHD()

for _ in range(10):
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

    # Generate Measurements based on all available information
    measurements, measurements_mask = measurement_system(
        all_positions, all_positions_mask, key
    )

    # EnGMF Update Equations
    key, subkey = jax.random.split(key)
    updated_result = filter.update(
        subkey,
        intensity_function,
        measurements,
        measurements_mask,
        measurement_system,
        6.25e-8,
        0.98,
    )
    print(updated_result)

    key, subkey = jax.random.split(key)
    true_state = eqx.filter_vmap(system.flow)(0.0, 1.0, true_state)
