import equinox as eqx
import abc


class AbstractFilter(eqx.Module, strict=True):

    @abc.abstractmethod
    def update(
        self,
        key: Key[Array, "..."],
        ensemble: Float[Array, "batch_size state_dim"],
        measurement: Float[Array, "batch_size measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "batch_size state_dim"]:
        pass
