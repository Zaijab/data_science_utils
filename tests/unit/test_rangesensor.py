"""
This module tests the RangeSensor class which is a measurement system for a HMM.

The main thing is checking if the abstract/final pattern holds correctly with my current implementation.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from data_science_utils.measurement_functions import RangeSensor
from data_science_utils.measurement_functions import AbstractMeasurementSystem
from jaxtyping import jaxtyped, Float, Array
from beartype import beartype as typechecker

measurement_system = RangeSensor(jnp.array([[1.0]]))


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def f(x: AbstractMeasurementSystem) -> Float[Array, "1"]:
    return jnp.array([1.0])


f(measurement_system)
# Error
# f(jnp.array([]))
