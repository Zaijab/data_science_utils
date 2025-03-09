import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from data_science_utils.dynamical_systems import IkedaSystem, ikeda_attractor_discriminator

# key = jax.random.key(0)
# key, subkey = jax.random.split(key)
# ikeda = IkedaSystem(state=jnp.array([[1.0, 2.0]]), u=0.9)
# batch = ikeda.generate(subkey)
# batch = ikeda.flow(batch)
# ikeda.iterate()

##############

# Implement Ikeda Discriminator
# Import NNX normalizing flow discriminator

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from jaxtyping import Float, Bool, Array, jaxtyped
from beartype import beartype as typechecker

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from data_science_utils.dynamical_systems import ikeda_attractor_discriminator

grid_spacing = 1000
x = jnp.linspace(-0.5, 2, grid_spacing)
y = jnp.linspace(-2.5, 1, grid_spacing)
XX, YY = jnp.meshgrid(x, y)
grid = jnp.dstack([XX, YY])
grid_points = grid.reshape(-1, 2)

@partial(jax.jit, static_argnames=["u"])
def ikeda_inverse(
    x: Float[Array, "*batch 2"],
    u: float = 0.9
) -> Bool[Array, "*batch"]:
    x1, x2 = x[..., 0], x[..., 1]
    x1, x2 = (x1 - 1) / u, (x2) / u
    t = (0.4 - (6 / (1 + x1**2 + x2**2)))
    sin_t, cos_t = jnp.sin(-t), jnp.cos(t)
    x1_new = (x1 * cos_t - x2 * sin_t)
    x2_new = (x1 * sin_t + x2 * cos_t)
    return jnp.stack((x1_new, x2_new), axis=-1)

@partial(jax.jit, static_argnames=["u"])
def ikeda_inverse(
    x: Float[Array, "*batch 2"],
    u: float = 0.9
) -> Float[Array, "*batch 2"]:
    # Unscale the coordinates
    x1_unscaled = (x[..., 0] - 1) / u
    x2_unscaled = x[..., 1] / u
    
    # Calculate the angle parameter t
    t = 0.4 - (6 / (1 + x1_unscaled**2 + x2_unscaled**2))
    
    # Apply the rotation matrix inverse (transpose)
    # [cos(t)  sin(t)]^T = [cos(t) -sin(t)]
    # [-sin(t) cos(t)]    [sin(t)  cos(t)]
    sin_t, cos_t = jnp.sin(t), jnp.cos(t)
    
    # The inverse rotation
    x1_prev = x1_unscaled * cos_t + x2_unscaled * sin_t
    x2_prev = -x1_unscaled * sin_t + x2_unscaled * cos_t
    
    return jnp.stack((x1_prev, x2_prev), axis=-1)

@partial(jax.jit, static_argnames=["u", "ninverses"])
def ikeda_attractor_discriminator(
    x: Float[Array, "*batch 2"],
    ninverses: int = 10,
    u: float = 0.9
) -> Bool[Array, "*batch"]:


    def apply_inverse_n_times(x_):
        def body_fn(_, state):
            return ikeda_inverse(state, u)
        return lax.fori_loop(0, ninverses, body_fn, x_)

    x_inv = apply_inverse_n_times(x)
    threshold = jnp.sqrt(1.0 / (1.0 - u))
    return jnp.linalg.norm(x_inv, axis=-1) < threshold

@partial(jax.jit, static_argnames=["u", "ninverses"])
def ikeda_attractor_discriminator(
    x: Float[Array, "*batch 2"],
    ninverses: int = 10,
    u: float = 0.9
) -> Bool[Array, "*batch"]:
    # Pre-compute threshold
    threshold_squared = 1.0 / (1.0 - u)
    
    def body_fn(i, state):
        x_curr, is_outside = state
        x_inv = ikeda_inverse(x_curr, u)
        norm_squared = jnp.sum(x_inv**2, axis=-1)
        return x_inv, is_outside | (norm_squared > threshold_squared)
    
    # Initialize with False for "is_outside"
    init_state = (x, jnp.zeros(x.shape[:-1], dtype=bool))
    x_final, is_outside = lax.fori_loop(0, ninverses, body_fn, init_state)
    
    # Return inverse: if ANY iteration went outside the threshold, it's not on attractor
    return ~is_outside

labels = ikeda_attractor_discriminator(grid_points, ninverses=10)
labels_grid = labels.reshape(grid_spacing, grid_spacing)

plt.figure(figsize=(8, 6))
plt.contourf(XX, YY, labels_grid)
plt.title('Discriminator Output')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
