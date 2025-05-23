import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from data_science_utils.filters import AbstractFilter
from data_science_utils.measurement_systems import AbstractMeasurementSystem
from jaxtyping import Array, Float, Key, jaxtyped


@jaxtyped(typechecker=typechecker)
class EnKF(AbstractFilter, strict=True):
    inflation_factor: float = 1.01
    debug: bool = False

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update(
        self,
        key: Key[Array, "..."],
        ensemble: Float[Array, "batch_size state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "batch_size state_dim"]:
        mean = jnp.mean(ensemble, axis=0)

        if self.debug:
            jax.debug.print("{shape}", mean.shape)

        inflated = mean + self.inflation_factor * (ensemble - mean)

        ensemble_covariance = jnp.cov(inflated.T)

        @jax.jit
        def update_ensemble_point(
            point: Float[Array, "state_dim"], key: Key[Array, "..."]
        ) -> Float[Array, "state_dim"]:
            point_measurement = measurement_system(point)
            measurement_jacobian = jax.jacfwd(measurement_system)(point)
            innovation_covariance = (
                measurement_jacobian @ ensemble_covariance @ measurement_jacobian.T
                + measurement_system.covariance
            )
            kalman_gain = (
                ensemble_covariance
                @ measurement_jacobian.T
                @ jnp.linalg.inv(innovation_covariance)
            )
            point = point + (
                kalman_gain @ jnp.atleast_2d(measurement - point_measurement)
            ).reshape(-1)
            return point

        keys = jax.random.split(key, ensemble.shape[0])
        updated_ensemble = jax.vmap(update_ensemble_point)(inflated, keys)
        return updated_ensemble
