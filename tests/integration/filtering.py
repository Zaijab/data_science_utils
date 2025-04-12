import jax
import jax.numpy as jnp

from data_science_utils.dynamical_systems import Ikeda

system = Ikeda()

system.forward(jnp.array([1.25, 0]))
