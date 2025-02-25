import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

"""
This dynamical system is a birth-death process
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key
from functools import partial

@jax.jit
def flow(state: Float[Array, "6"], dt: float = 0.1, key: Key[Array, "1"] = jax.random.key(42)) -> Float[Array, "6"]:
    # Construct the state transition matrix F (6x6) for a constant velocity model.
    # F = [ I3   dt * I3 ]
    #     [ 0    I3     ]
    F: Float[Array, "6 6"] = jnp.block([
        [jnp.eye(3), dt * jnp.eye(3)],
        [jnp.zeros((3, 3)), jnp.eye(3)]
    ])

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
new_state: Float[Array, "6"] = flow(state_vector, dt=0.1, key=jax.random.key(42))
print(new_state)
