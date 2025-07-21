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

class AbstractFilterReimplementation(eqx.Module, strict=True):
    """
    This is a reimplementation of the filtering class.
    This will succeed the previous filtering methods in a way which is much more mathematically faithful.

    The filter method will contain three main methods in order to implement the Bayesian Recursive Relations.
    The main data type being passed around is a distreqx
    """
    @abc.abstractmethod
    def update(
        self,
        key: Key[Array, "..."],
        prior_ensemble: Float[Array, "batch_size state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "batch_size state_dim"]:
        raise NotImplementedError
