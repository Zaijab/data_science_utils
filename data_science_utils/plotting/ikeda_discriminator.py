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
from beartype import beartype as typechecker
from jax import lax, random
from jaxtyping import Array, Float, Bool, jaxtyped


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['u'])
def flow(x: Float[Array, "*batch 2"], u: float = 0.9) -> Float[Array, "*batch 2"]:
    x1, x2 = x[..., 0], x[..., 1]
    t = 0.4 - (6 / (1 + x1**2 + x2**2))
    sin_t, cos_t = jnp.sin(t), jnp.cos(t)
    x1_new = 1 + u * (x1 * cos_t - x2 * sin_t)
    x2_new = u * (x1 * sin_t + x2 * cos_t)
    return jnp.stack((x1_new, x2_new), axis=-1)


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['batch_size', 'u'])
def generate(key, batch_size: int = 10**5, u: float = 0.9) -> Array:
    def body_fn(i, val):
        return flow(val, u)
    initial_state = random.uniform(key, shape=(batch_size, 2), minval=-0.25, maxval=0.25)
    return lax.fori_loop(0, 15, body_fn, initial_state)


@dataclass
class IkedaSystem:
    u: float = 0.9
    state: Float[Array, "*batch 2"] = jnp.array([[1.25, 0]])

    def __post_init__(self):
        self.flow: Callable[[Float[Array, "*batch 2"]], Float[Array, "*batch 2"]] = partial(flow, u=self.u)
        self.generate: Callable[[Array, int, float], Array] = partial(generate, u=self.u)

    def iterate(self):
        self.state = self.flow(self.state)


@partial(jax.jit, static_argnames=["u", "ninverses"])
def ikeda_attractor_discriminator(
    x: Float[Array, "*batch 2"],
    ninverses: int = 10,
    u: float = 0.9
) -> Bool[Array, "*batch"]:
    """
    Returns a boolean array indicating which points in x are on the Ikeda map's attractor.
    Replicates the logic of the MATLAB 'onIkedaAttractor' function using repeated inverse iterations.
    """

    @jax.jit
    def ikeda_inv(zp1: Float[Array, "*batch 2"]) -> Float[Array, "*batch 2"]:
        """
        Inverts the Ikeda map via a fixed number of Newton's method steps and normalizes
        to preserve the radius as in the MATLAB code.
        """

        # Unshift and unscale
        zn = (zp1 - jnp.array([1.0, 0.0])) / u

        def newton_iteration(zi, _):
            xi, yi = zi[..., 0], zi[..., 1]

            opx2y2 = 1.0 + xi**2 + yi**2
            ti = 0.4 - 6.0 / opx2y2
            cti, sti = jnp.cos(ti), jnp.sin(ti)

            dti_dx = 12.0 * xi / (opx2y2**2)
            dti_dy = 12.0 * yi / (opx2y2**2)

            # Jacobian terms
            # J = [[J11, J12],
            #      [J21, J22]]
            # Each is shape (...,) for batch
            J11 = cti - (yi * cti + xi * sti) * dti_dx
            J12 = -sti - (yi * cti + xi * sti) * dti_dy
            J21 = sti + (xi * cti - yi * sti) * dti_dx
            J22 = cti + (xi * cti - yi * sti) * dti_dy

            # Residual c = zn - forward(zi)
            # forward(zi) = [xi*cos(ti)-yi*sin(ti), xi*sin(ti)+yi*cos(ti)]
            c0 = zn[..., 0] - (xi * cti - yi * sti)
            c1 = zn[..., 1] - (xi * sti + yi * cti)

            # Solve 2x2 system J * dZi = c
            detJ = J11 * J22 - J12 * J21
            dx0 = (J22 * c0 - J12 * c1) / detJ
            dx1 = (-J21 * c0 + J11 * c1) / detJ
            zi_next = jnp.stack([xi + dx0, yi + dx1], axis=-1)

            # Enforce the same radius as zn
            zn_norm = jnp.linalg.norm(zn, axis=-1, keepdims=True)
            zi_norm = jnp.linalg.norm(zi_next, axis=-1, keepdims=True)
            zi_next = jnp.where(zi_norm > 0, zn_norm * zi_next / zi_norm, zi_next)

            return zi_next, None

        # Initialize with zn and do 12 Newton steps
        zi_final, _ = lax.scan(newton_iteration, zn, None, length=8)
        return zi_final

    def apply_inverse_n_times(x_):
        def body_fn(_, state):
            return ikeda_inv(state)
        return lax.fori_loop(0, ninverses, body_fn, x_)

    x_inv = apply_inverse_n_times(x)
    threshold = jnp.sqrt(1.0 / (1.0 - u))
    return jnp.linalg.norm(x_inv, axis=-1) < threshold


##############

grid_spacing = 1000
x = jnp.linspace(-0.5, 2, grid_spacing)
y = jnp.linspace(-2.5, 1, grid_spacing)
XX, YY = jnp.meshgrid(x, y)
grid = jnp.dstack([XX, YY])
grid_points = grid.reshape(-1, 2)
labels = ikeda_attractor_discriminator(grid_points)
labels_grid = labels.reshape(grid_spacing, grid_spacing)

plt.figure(figsize=(8, 6))
plt.contourf(XX, YY, labels_grid)
plt.title('Discriminator Output')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
