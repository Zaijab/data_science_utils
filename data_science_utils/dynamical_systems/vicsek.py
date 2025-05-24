import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from data_science_utils.dynamical_systems import AbstractDiscreteDynamicalSystem

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import equinox as eqx


@jax.jit
def points_in_epsilon_ball(point_cloud, query_points, radius):
    """
    Find indices of points in point_cloud within radius of each query point

    Args:
        point_cloud: jnp.array of shape (n_points, dim)
        query_points: jnp.array of shape (n_queries, dim)
        radius: float, distance threshold

    Returns:
        mask: jnp.array of shape (n_queries, n_points) where mask[i, j] is True
              if point_cloud[j] is within radius of query_points[i]
    """
    # Compute squared distances between query points and all points
    # Expanding dimensions to enable broadcasting
    pc_expanded = jnp.expand_dims(point_cloud, axis=0)  # (1, n_points, dim)
    qp_expanded = jnp.expand_dims(query_points, axis=1)  # (n_queries, 1, dim)

    # Calculate squared distances
    squared_diffs = (pc_expanded - qp_expanded) ** 2  # (n_queries, n_points, dim)
    squared_distances = jnp.sum(squared_diffs, axis=-1)  # (n_queries, n_points)

    # Create mask for points within radius
    mask = squared_distances <= radius**2  # (n_queries, n_points)

    return mask


import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Key, jaxtyped
from beartype import beartype as typechecker
from functools import partial


@jaxtyped(typechecker=typechecker)
class VicsekModel(AbstractDiscreteDynamicalSystem):
    """
    2D Vicsek Model for collective motion.

    State representation:
    - state: shape (n_particles, 3) where:
      - state[:, 0:2] are x,y positions
      - state[:, 2] are orientations in radians
    """

    # Model parameters
    n_particles: int
    speed: float
    noise_strength: float
    interaction_radius: float
    domain_size: Float[Array, "2"]

    def __init__(
        self,
        n_particles: int,
        speed: float = 0.03,
        noise_strength: float = 0.1,
        interaction_radius: float = 1.0,
        domain_size: Float[Array, "2"] = None,
    ):
        """
        Initialize the Vicsek model.

        Args:
            n_particles: Number of particles in the system
            speed: Constant speed of all particles
            noise_strength: Magnitude of angular noise (in radians)
            interaction_radius: Radius within which particles interact
            domain_size: Size of the periodic domain [width, height]
        """
        self.n_particles = n_particles
        self.speed = speed
        self.noise_strength = noise_strength
        self.interaction_radius = interaction_radius

        # Default domain size is a 10x10 square if not specified
        if domain_size is None:
            self.domain_size = jnp.array([10.0, 10.0])
        else:
            self.domain_size = domain_size

    @jaxtyped(typechecker=typechecker)
    def initialize_state(self, key: Key[Array, ""]) -> Float[Array, "n_particles 3"]:
        """
        Initialize random positions and orientations.

        Args:
            key: PRNG key for random generation

        Returns:
            state: Initial state with random positions and orientations
        """
        # Split key for positions and orientations
        pos_key, ori_key = jax.random.split(key)

        # Random positions uniformly distributed in the domain
        positions = (
            jax.random.uniform(
                pos_key, shape=(self.n_particles, 2), minval=0.0, maxval=1.0
            )
            * self.domain_size
        )

        # Random orientations in [0, 2π)
        orientations = jax.random.uniform(
            ori_key, shape=(self.n_particles,), minval=0.0, maxval=2 * jnp.pi
        )

        # Combine into a single state matrix
        state = jnp.zeros((self.n_particles, 3))
        state = state.at[:, 0:2].set(positions)
        state = state.at[:, 2].set(orientations)

        return state

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def forward(
        self, state: Float[Array, "n_particles 3"], key: Key[Array, ""]
    ) -> Float[Array, "n_particles 3"]:
        """
        Perform one step of the Vicsek model:
        1. Find neighbors within interaction radius
        2. Compute mean orientation of neighbors
        3. Add noise to orientations
        4. Update positions based on new orientations
        5. Apply periodic boundary conditions

        Args:
            state: Current state (positions and orientations)
            key: PRNG key for noise generation

        Returns:
            Updated state
        """
        # Extract positions and orientations
        positions = state[:, 0:2]
        orientations = state[:, 2]

        # Find neighbors within interaction radius
        # Use positions as both point_cloud and query_points
        neighbor_mask = points_in_epsilon_ball(
            positions, positions, self.interaction_radius
        )

        # Remove self-connections (diagonal elements of the mask)
        neighbor_mask = neighbor_mask & ~jnp.eye(self.n_particles, dtype=bool)

        # Calculate average orientation of neighbors using vector averaging
        # Convert angles to unit vectors for proper circular averaging
        cos_theta = jnp.cos(orientations)
        sin_theta = jnp.sin(orientations)

        # Create velocity vector components for each particle
        vel_x = cos_theta  # shape: (n_particles,)
        vel_y = sin_theta  # shape: (n_particles,)

        # Compute sum of velocity components for neighbors
        # Use masked sum to only include neighbors
        # First expand vel_x and vel_y to match the mask shape
        vel_x_expanded = jnp.expand_dims(vel_x, 0)  # (1, n_particles)
        vel_y_expanded = jnp.expand_dims(vel_y, 0)  # (1, n_particles)

        # Apply mask and sum over neighbors
        sum_cos = jnp.sum(vel_x_expanded * neighbor_mask, axis=1)  # (n_particles,)
        sum_sin = jnp.sum(vel_y_expanded * neighbor_mask, axis=1)  # (n_particles,)

        # Add the particle's own orientation
        sum_cos = sum_cos + cos_theta
        sum_sin = sum_sin + sin_theta

        # Calculate new orientation angle
        # Handle case where a particle has no neighbors (including itself)
        neighbor_count = jnp.sum(neighbor_mask, axis=1) + 1  # +1 for self

        # Calculate average cos and sin, avoiding division by zero
        # If no neighbors, keep current orientation
        avg_cos = jnp.where(neighbor_count > 0, sum_cos / neighbor_count, cos_theta)
        avg_sin = jnp.where(neighbor_count > 0, sum_sin / neighbor_count, sin_theta)

        # Convert back to angle
        new_orientation = jnp.arctan2(avg_sin, avg_cos)

        # Add noise to orientation
        noise_key, _ = jax.random.split(key)
        noise = jax.random.uniform(
            noise_key,
            shape=(self.n_particles,),
            minval=-self.noise_strength / 2,
            maxval=self.noise_strength / 2,
        )
        new_orientation = new_orientation + noise

        # Ensure orientation stays in [0, 2π)
        new_orientation = jnp.mod(new_orientation, 2 * jnp.pi)

        # Update positions based on new orientations
        velocity_x = self.speed * jnp.cos(new_orientation)
        velocity_y = self.speed * jnp.sin(new_orientation)

        new_positions = positions + jnp.column_stack((velocity_x, velocity_y))

        # Apply periodic boundary conditions
        new_positions = jnp.mod(new_positions, self.domain_size)

        # Construct new state
        new_state = jnp.zeros_like(state)
        new_state = new_state.at[:, 0:2].set(new_positions)
        new_state = new_state.at[:, 2].set(new_orientation)

        return new_state


system = VicsekModel(n_particles=200)

key = jax.random.key(0)
key, subkey = jax.random.split(key)
true_state = system.initialize_state(subkey)

key, subkey = jax.random.split(key)
system.forward(true_state, subkey)
system.forward(true_state, subkey)

import datetime

print(datetime.datetime.now())
system.forward(true_state, subkey)


def run_simulation(model, initial_state, n_steps, seed=0):
    """
    Run the Vicsek model for n_steps and return the trajectory.
    """
    # Initialize trajectory array to store all states
    states = jnp.zeros((n_steps + 1, model.n_particles, 3))
    states = states.at[0].set(initial_state)

    # Create initial key
    key = jax.random.key(seed)

    # Run simulation
    for i in range(n_steps):
        key, step_key = jax.random.split(key)
        states = states.at[i + 1].set(model.forward(states[i], step_key))

    return states


def create_animation(states, model, figsize=(10, 10), interval=50, show_velocity=True):
    """
    Create animation of Vicsek model simulation.
    """
    n_steps = states.shape[0]

    # Convert to numpy for matplotlib
    states_np = np.array(states)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Set axis limits with some padding
    padding = 0.5
    ax.set_xlim(-padding, model.domain_size[0] + padding)
    ax.set_ylim(-padding, model.domain_size[1] + padding)

    # Plot domain boundary
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            model.domain_size[0],
            model.domain_size[1],
            fill=False,
            edgecolor="gray",
            linestyle="--",
        )
    )

    # Initialize scatter plot for particles
    particles = ax.scatter([], [], s=25, color="blue", alpha=0.7)
    # Get initial positions and orientations
    initial_positions = states_np[0, :, 0:2]
    initial_orientations = states_np[0, :, 2]
    initial_vel_x = np.cos(initial_orientations)
    initial_vel_y = np.sin(initial_orientations)

    # Initialize quiver plot with initial data
    velocities = ax.quiver(
        initial_positions[:, 0],
        initial_positions[:, 1],
        initial_vel_x,
        initial_vel_y,
        scale=15,
        width=0.005,
    )

    # # Initialize quiver plot for velocities if requested
    # if show_velocity:
    #     velocities = ax.quiver([], [], [], [], scale=15, width=0.005)

    # Add time text
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")

    # Add title and labels
    ax.set_title(
        f"Vicsek Model (N={model.n_particles}, R={model.interaction_radius}, η={model.noise_strength})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Animation update function
    def update(frame):
        # Extract positions and orientations
        positions = states_np[frame, :, 0:2]
        orientations = states_np[frame, :, 2]

        # Update particles
        particles.set_offsets(positions)

        # Update velocities if requested
        if show_velocity:
            vel_x = np.cos(orientations)
            vel_y = np.sin(orientations)
            velocities.set_offsets(positions)
            velocities.set_UVC(vel_x, vel_y)

        # Update time text
        time_text.set_text(f"Step: {frame}")

        if show_velocity:
            return particles, velocities, time_text
        else:
            return particles, time_text

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=n_steps, interval=interval, blit=True
    )

    plt.close(fig)  # To avoid displaying the figure twice

    return anim, fig


# Run the simulation and create animation
def simulate_vicsek_model(
    n_particles=200,
    speed=0.05,
    noise=0.2,
    radius=0.5,
    n_steps=100,
    domain_size=None,
    seed=0,
    save_gif=True,
):
    """
    Run a full Vicsek model simulation and return the animation.

    Parameters:
    -----------
    n_particles : int
        Number of particles
    speed : float
        Particle speed
    noise : float
        Noise strength (in radians)
    radius : float
        Interaction radius
    n_steps : int
        Number of simulation steps
    domain_size : array-like, optional
        Size of the domain [width, height]
    seed : int
        Random seed
    save_gif : bool
        Whether to save the animation as a GIF

    Returns:
    --------
    animation object
    """
    # Create model
    if domain_size is None:
        domain_size = jnp.array([10.0, 10.0])

    model = VicsekModel(
        n_particles=n_particles,
        speed=speed,
        noise_strength=noise,
        interaction_radius=radius,
        domain_size=domain_size,
    )

    # Initialize state
    key = jax.random.key(seed)
    initial_state = model.initialize_state(key)

    # Run simulation
    states = run_simulation(model, initial_state, n_steps)

    # Create animation
    anim, fig = create_animation(states, model, interval=100)

    # Save GIF if requested
    if save_gif:
        try:
            anim.save("vicsek_model.gif", writer="pillow", fps=10)
            print("Animation saved as 'vicsek_model.gif'")
        except Exception as e:
            print(f"Failed to save GIF: {e}")

    return anim, fig, states


# Example usage
anim, fig, states = simulate_vicsek_model(
    n_particles=200, speed=0.05, noise=0.2, radius=0.5, n_steps=100, save_gif=True
)


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
