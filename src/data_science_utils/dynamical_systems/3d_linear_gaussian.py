"""
This dynamical system is a birth-death process
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Key
from functools import partial


@jax.jit
def flow(
        state: Float[Array, "2*spatial_dim"],
        dt: float = 0.1,
        key: None | Key[Array, "1"] = None
) -> Float[Array, "spatial_dim"]:
    spatial_dim = state.shape[0] / 2
    single_dimension_state_transition = jnp.array([[1.0, dt],
                                                   [0, 1.0]])
    state_transition = jnp.kron(jnp.eye(spatial_dim), single_dimension_state_transition)

    perfect_transitioned_state = state_transition @ state
    
    # Process noise: acceleration noise is scaled by sigma_v.
    sigma_v: float = 5.0
    # Define B0 (2x1) for one spatial dimension: position noise = (dt^2)/2 and velocity noise = dt.
    B0: Float[Array, "2 1"] = jnp.array([[dt ** 2 / 2.0],
                                          [dt]])
    # Build the 6x3 matrix B by placing B0 in a block-diagonal manner for each of the 3 spatial dimensions.
    B: Float[Array, "6 3"] = jnp.block([
        [B0, jnp.zeros((2, 1)), jnp.zeros((2, 1))],
        [jnp.zeros((2, 1)), B0, jnp.zeros((2, 1))],
        [jnp.zeros((2, 1)), jnp.zeros((2, 1)), B0]
    ])

    # Sample acceleration noise for each spatial dimension.
    # The noise is drawn from a standard normal scaled by sigma_v; shape: (3,)
    w_acc: Float[Array, "3"] = sigma_v * jax.random.normal(key, shape=(3,))

    # Compute the full process noise vector (6,) via B @ w_acc.
    noise: Float[Array, "6"] = (B @ w_acc).flatten()

    # Return the updated state: the deterministic part plus process noise.
    return F @ state + noise

# Example usage:
state_vector: Float[Array, "6"] = jnp.array([0, 15, 0, -10, -10, 3])
new_state: Float[Array, "6"] = flow(state_vector, dt=1, key=None)
print(new_state)
