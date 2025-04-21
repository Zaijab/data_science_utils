import jax
import jax.numpy as jnp

from data_science_utils.dynamical_systems import Ikeda

system = Ikeda()

true_state = jnp.array([1.25, 0])

true_state = system.forward(true_state)
true_state = system.forward(true_state)
true_state = system.forward(true_state)
