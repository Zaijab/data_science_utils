import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from data_science_utils.dynamical_systems.abc import (
    AbstractStochasticDiscreteDynamicalSystem,
)

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
    # mask[i, j]
    # True if query_point[i] < point_cloud[j]
    return mask


import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Key, jaxtyped
from beartype import beartype as typechecker
from functools import partial


@jaxtyped(typechecker=typechecker)
class Vicsek(AbstractStochasticDiscreteDynamicalSystem):
    """
    2D Vicsek Model for collective motion.

    State representation:
    - state: shape (n_particles, 3) where:
      - state[:, 0:2] are x,y positions
      - state[:, 2] are orientations in radians
    """

    # Model parameters
    # Number of particles in the system
    n_particles: int = 100

    # Constant speed of all particles
    speed: float = 0.03

    # Magnitude of angular noise (in radians)
    noise_strength: float = 0.1

    # Radius within which particles interact
    interaction_radius: float = 1.0

    # Size of the periodic domain [width, height]
    domain_width: float = 10.0
    domain_height: float = 10.0

    @property
    def dimension(self):
        return 2

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def initial_state(
        self, key: Key[Array, "..."] | None = None, **kwargs
    ) -> Float[Array, "n_particles 3"]:

        # Split key for positions and orientations
        pos_key, ori_key = jax.random.split(key)

        # Random positions uniformly distributed in the domain
        positions = jax.random.uniform(
            pos_key, shape=(self.n_particles, 2), minval=0.0, maxval=1.0
        ) * jnp.array([self.domain_width, self.domain_height])

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
        self, state: Float[Array, "{self.n_particles} 3"], key: Key[Array, "..."]
    ) -> Float[Array, "{self.n_particles} 3"]:
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
        new_positions = jnp.mod(
            new_positions, jnp.array([self.domain_width, self.domain_height])
        )

        # Construct new state
        new_state = jnp.zeros_like(state)
        new_state = new_state.at[:, 0:2].set(new_positions)
        new_state = new_state.at[:, 2].set(new_orientation)

        return new_state
