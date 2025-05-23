import equinox as eqx
import abc
from jaxtyping import Key, Float, Array
from data_science_utils.measurement_systems import AbstractMeasurementSystem


class AbstractFilter(eqx.Module, strict=True):

    @abc.abstractmethod
    def update(
        self,
        key: Key[Array, "..."],
        prior_ensemble: Float[Array, "batch_size state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "batch_size state_dim"]:
        raise NotImplementedError
