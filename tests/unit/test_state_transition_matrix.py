import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float

from data_science_utils.dynamical_systems.crtbp import CR3BP


dynamical_system = CR3BP()
state = dynamical_system.initial_state()
stm_1 = jax.jacrev(dynamical_system.flow, argnums=(2,))(0.0, 1.0, state)[0]
stm_2 = jax.jacrev(dynamical_system.flow, argnums=(2,))(1.0, 2.0, dynamical_system.flow(0.0, 1.0, state))[0]
stm_3 = jax.jacrev(dynamical_system.flow, argnums=(2,))(0.0, 2.0, state)[0]

print(stm_2 @ stm_1)
print(stm_3)

stm_identity = jax.jacrev(dynamical_system.flow, argnums=(2,))(1.0, 1.0, state)[0]
print(stm_identity)
