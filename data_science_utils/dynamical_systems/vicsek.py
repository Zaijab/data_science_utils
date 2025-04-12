import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=typechecker)
class VicsekModel(eqx.Module):
    n_particles: int = 100

    def __init__(
        self, key, n_particles, box_size, velocity, radius, noise_strength, dt
    ):
        # Split the key for position and orientation
        key_pos, key_theta = jax.random.split(key)

        # Randomly distribute particles in the box
        positions = jax.random.uniform(key_pos, shape=(n_particles, 2)) * box_size

        # Random initial directions (angles in radians)
        thetas = jax.random.uniform(key_theta, shape=(n_particles,)) * 2 * jnp.pi

        return positions, thetas

    pass


def animate_vicsek():
    pass
