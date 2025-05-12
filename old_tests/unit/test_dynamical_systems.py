"""
This test suite will set up and run all dynamical systems defined in data_science_utils/dynamical_systems.

The purpose is manifold:

(1) Homogenize developer facing API
(2) Display how the updates appear
(3) Easy copy and paste for plotting

Every dynamical system contains:

(1) a forward function
(2) a state vector dimention

Dynamical systems may optionally implement:

(1) a backward function
(2) an attractor generator function

In the work of stochastic filtering, we find it worthwhile to test an algorithm in many different cases.
To test on many different dynamical systems easily, we find the object oriented pattern suits this well.

system = DYNAMICAL_SYSTEM(PARAMETERS)

To be easiest to use with JAX, it seems like we should use pure functions.
However, this means needing to find the names of every 
"""

import jax
import jax.numpy as jnp

# Find every implemented system

### Claude

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Key, jaxtyped
from beartype import beartype as typechecker
from functools import partial
from typing import Tuple


@jaxtyped(typechecker=typechecker)
class VicsekModel(eqx.Module):
    """
    Vicsek model for collective motion simulation.

    Each particle moves at constant speed and adjusts direction according to
    the average direction of neighboring particles within interaction radius,
    plus random noise.

    Updates follow:
    - positions: x(t+1) = x(t) + v(t) * dt
    - velocities: v(t+1) = v0 * (avg_direction_of_neighbors + noise)
    """

    dimension: int  # Spatial dimension (2D or 3D)
    num_particles: int
    speed: float  # Constant speed magnitude
    radius: float  # Interaction radius
    noise_strength: float
    dt: float  # Time step

    def __init__(
        self,
        dimension: int = 2,
        num_particles: int = 100,
        speed: float = 0.03,
        radius: float = 1.0,
        noise_strength: float = 0.1,
        dt: float = 1.0,
    ):
        self.dimension = dimension
        self.num_particles = num_particles
        self.speed = speed
        self.radius = radius
        self.noise_strength = noise_strength
        self.dt = dt

    @staticmethod
    @jax.jit
    def _normalize(v: Float[Array, "... d"]) -> Float[Array, "... d"]:
        """Normalize vectors row-wise."""
        norm = jnp.sqrt(jnp.sum(v**2, axis=-1, keepdims=True))
        # Handle zero vectors
        norm = jnp.where(norm == 0, 1.0, norm)
        return v / norm

    @staticmethod
    @jax.jit
    def _distance_matrix(positions: Float[Array, "n d"]) -> Float[Array, "n n"]:
        """Compute pairwise distances between all particles."""
        diff = positions[:, None, :] - positions[None, :, :]  # (n, n, d)
        return jnp.sqrt(jnp.sum(diff**2, axis=2))  # (n, n)

    @staticmethod
    @jax.jit
    def _compute_neighbors_mask(
        distances: Float[Array, "n n"], radius: float
    ) -> Float[Array, "n n"]:
        """Create a mask for identifying neighbors within the radius."""
        return (distances < radius).astype(jnp.float32)

    @jaxtyped(typechecker=typechecker)
    @partial(jax.jit, static_argnames=["debug"])
    def compute_average_direction(
        self,
        positions: Float[Array, "{self.num_particles} {self.dimension}"],
        velocities: Float[Array, "{self.num_particles} {self.dimension}"],
        debug: bool = False,
    ) -> Float[Array, "{self.num_particles} {self.dimension}"]:
        """Compute the average direction of neighbors for each particle."""

        # Compute pairwise distances
        distances = self._distance_matrix(positions)

        # Create mask for neighbors
        neighbors_mask = self._compute_neighbors_mask(distances, self.radius)

        # Set diagonal to 0 (particle is not its own neighbor)
        neighbors_mask = neighbors_mask.at[
            jnp.arange(self.num_particles), jnp.arange(self.num_particles)
        ].set(0)

        # Compute sum of neighbor directions
        neighbor_sum = neighbors_mask[:, :, None] * velocities[None, :, :]  # (n, n, d)
        neighbor_sum = jnp.sum(neighbor_sum, axis=1)  # (n, d)

        # Count number of neighbors
        num_neighbors = jnp.sum(neighbors_mask, axis=1, keepdims=True)  # (n, 1)

        # Compute average direction, handling case of no neighbors
        avg_direction = jnp.where(
            num_neighbors > 0,
            neighbor_sum / num_neighbors,
            velocities,  # Keep current direction if no neighbors
        )

        # Normalize to get unit vectors
        avg_direction = self._normalize(avg_direction)

        if debug:
            assert avg_direction.shape == (self.num_particles, self.dimension)

        return avg_direction

    @jaxtyped(typechecker=typechecker)
    @partial(jax.jit, static_argnames=["debug"])
    def forward(
        self,
        state: Tuple[
            Float[Array, "{self.num_particles} {self.dimension}"],  # positions
            Float[Array, "{self.num_particles} {self.dimension}"],  # velocities
        ],
        key: Key[Array, "..."],
        debug: bool = False,
    ) -> Tuple[
        Float[Array, "{self.num_particles} {self.dimension}"],  # next positions
        Float[Array, "{self.num_particles} {self.dimension}"],  # next velocities
    ]:
        """
        Update the state of the Vicsek model.

        Args:
            state: Tuple of (positions, velocities)
            key: JAX random key
            debug: Enable debug assertions

        Returns:
            Tuple of (next_positions, next_velocities)
        """
        positions, velocities = state

        if debug:
            assert positions.shape == (self.num_particles, self.dimension)
            assert velocities.shape == (self.num_particles, self.dimension)

        # Compute average direction of neighbors
        avg_direction = self.compute_average_direction(
            positions, velocities, debug=debug
        )

        # Add noise to the direction
        key1, key2 = jax.random.split(key)

        if self.dimension == 2:
            # For 2D: add noise as a random angle perturbation
            noise = self.noise_strength * jax.random.uniform(
                key1, (self.num_particles, 1), minval=-jnp.pi, maxval=jnp.pi
            )

            # Convert directions to angles
            angles = jnp.arctan2(avg_direction[:, 1], avg_direction[:, 0])

            # Add noise to angles
            noisy_angles = angles + noise[:, 0]

            # Convert back to cartesian coordinates
            next_velocities = self.speed * jnp.column_stack(
                (jnp.cos(noisy_angles), jnp.sin(noisy_angles))
            )
        else:
            # For 3D or higher: add noise as a random vector perturbation
            noise = self.noise_strength * jax.random.normal(
                key1, (self.num_particles, self.dimension)
            )

            # Add noise and normalize
            noisy_direction = avg_direction + noise
            next_velocities = self.speed * self._normalize(noisy_direction)

        # Update positions
        next_positions = positions + next_velocities * self.dt

        if debug:
            assert next_positions.shape == (self.num_particles, self.dimension)
            assert next_velocities.shape == (self.num_particles, self.dimension)

        return next_positions, next_velocities

    @jaxtyped(typechecker=typechecker)
    @partial(jax.jit, static_argnames=["num_steps", "debug"])
    def simulate(
        self,
        initial_state: Tuple[
            Float[Array, "{self.num_particles} {self.dimension}"],  # positions
            Float[Array, "{self.num_particles} {self.dimension}"],  # velocities
        ],
        key: Key[Array, "..."],
        num_steps: int = 100,
        debug: bool = False,
    ) -> Tuple[
        Float[
            Array, "{num_steps} {self.num_particles} {self.dimension}"
        ],  # position history
        Float[
            Array, "{num_steps} {self.num_particles} {self.dimension}"
        ],  # velocity history
    ]:
        """
        Simulate the Vicsek model for a given number of steps.

        Args:
            initial_state: Tuple of (positions, velocities)
            key: JAX random key
            num_steps: Number of simulation steps
            debug: Enable debug assertions

        Returns:
            Tuple of (position_history, velocity_history)
        """

        def step_fn(carry, i):
            state, step_key = carry
            step_key, next_key = jax.random.split(step_key)
            next_state = self.forward(state, step_key, debug=debug)
            return (next_state, next_key), (next_state[0], next_state[1])

        # Initialize the scan loop
        keys = jax.random.split(key, num_steps)
        init_carry = (initial_state, keys[0])

        # Run the simulation
        _, (position_history, velocity_history) = jax.lax.scan(
            step_fn, init_carry, jnp.arange(num_steps)
        )

        if debug:
            assert position_history.shape == (
                num_steps,
                self.num_particles,
                self.dimension,
            )
            assert velocity_history.shape == (
                num_steps,
                self.num_particles,
                self.dimension,
            )

        return position_history, velocity_history

    @staticmethod
    @jaxtyped(typechecker=typechecker)
    def initialize_random_state(
        key: Key[Array, "..."],
        num_particles: int,
        dimension: int,
        box_size: float = 10.0,
        speed: float = 0.03,
    ) -> Tuple[
        Float[Array, "{num_particles} {dimension}"],  # positions
        Float[Array, "{num_particles} {dimension}"],  # velocities
    ]:
        """
        Initialize a random state for the Vicsek model.

        Args:
            key: JAX random key
            num_particles: Number of particles
            dimension: Space dimension
            box_size: Size of the bounding box
            speed: Speed magnitude

        Returns:
            Tuple of (positions, velocities)
        """
        pos_key, vel_key = jax.random.split(key)

        # Random positions in a box
        positions = box_size * jax.random.uniform(pos_key, (num_particles, dimension))

        # Random directions with constant speed
        if dimension == 2:
            angles = jax.random.uniform(
                vel_key, (num_particles,), minval=0, maxval=2 * jnp.pi
            )
            velocities = speed * jnp.column_stack((jnp.cos(angles), jnp.sin(angles)))
        else:
            # For 3D or higher
            random_dirs = jax.random.normal(vel_key, (num_particles, dimension))
            unit_dirs = VicsekModel._normalize(random_dirs)
            velocities = speed * unit_dirs

        return positions, velocities


### Example usage
import jax
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize random key
key = jax.random.key(42)

# Create model
model = VicsekModel(
    dimension=2, num_particles=100, speed=0.03, radius=1.0, noise_strength=0.1, dt=1.0
)

# Initialize random state
initial_state = VicsekModel.initialize_random_state(
    key=key,
    num_particles=model.num_particles,
    dimension=model.dimension,
    box_size=10.0,
    speed=model.speed,
)

# Simulate
simulation_key, _ = jax.random.split(key)
position_history, velocity_history = model.simulate(
    initial_state=initial_state, key=simulation_key, num_steps=200
)

# Visualization (2D only)
fig, ax = plt.subplots(figsize=(10, 10))


def update(frame):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title(f"Vicsek Model - Frame {frame}")

    # Extract positions and velocities
    pos = position_history[frame]
    vel = velocity_history[frame]

    # Plot particles as points
    ax.scatter(pos[:, 0], pos[:, 1], c="blue", s=30)

    # Plot velocity vectors
    ax.quiver(
        pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], color="red", scale=0.1, width=0.005
    )

    return ax


# Create animation
anim = FuncAnimation(fig, update, frames=len(position_history), interval=50)
plt.show()
