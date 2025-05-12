import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


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


def plot_query(point_cloud, query_points, epsilon_mask, radius):
    """
    Plot points and highlight those within radius of query points.

    Args:
        point_cloud: jnp.array of shape (n_points, dim)
        query_points: jnp.array of shape (n_queries, dim)
        epsilon_mask: jnp.array of shape (n_queries, n_points) from points_in_epsilon_ball
        radius: float, distance threshold
    """
    # Convert to numpy for matplotlib compatibility
    points_np = jnp.asarray(point_cloud)
    query_np = jnp.asarray(query_points)
    mask_np = jnp.asarray(epsilon_mask)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all points in gray
    ax.scatter(
        points_np[:, 0],
        points_np[:, 1],
        s=10,
        color="gray",
        alpha=0.5,
        label="All Points",
    )

    # Colors for different query points
    colors = ["red", "blue", "green", "orange", "purple"]

    # For each query point
    for i, query in enumerate(query_np):
        color = colors[i % len(colors)]

        # Get points within epsilon
        nearby_indices = jnp.where(mask_np[i])[0]
        nearby_points = points_np[nearby_indices]

        # Plot nearby points
        if len(nearby_points) > 0:
            ax.scatter(
                nearby_points[:, 0],
                nearby_points[:, 1],
                s=30,
                color=color,
                alpha=0.7,
                label=f"Neighbors of query {i}",
            )

        # Plot query point
        ax.scatter(
            query[0],
            query[1],
            s=100,
            color=color,
            marker="*",
            edgecolor="black",
            label=f"Query point {i}",
        )

        # Draw circle with radius
        circle = plt.Circle(
            (query[0], query[1]),
            radius,
            color=color,
            fill=False,
            linestyle="--",
            alpha=0.5,
        )
        ax.add_patch(circle)

    # Add grid and labels
    ax.grid(alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Points within radius {radius} of query points")

    # Add legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))

    # Set equal aspect ratio and show
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


# Scenario 1: Uniform Grid
x, y = jnp.mgrid[0:5, 0:5]
point_cloud = jnp.c_[x.ravel(), y.ravel()]
query_points = jnp.array([[2, 0], [3, 3]])
radius = 1.415
epsilon_mask = points_in_epsilon_ball(point_cloud, query_points, radius)
plot_query(point_cloud, query_points, epsilon_mask, radius)

# Scenario 2: Normal Random
point_cloud = 2 * jax.random.normal(jax.random.key(0), shape=(1000, 2)) + jnp.array(
    [3, 3]
)
query_points = jnp.array([[2, 0], [3, 3]])
radius = 1

epsilon_mask = points_in_epsilon_ball(point_cloud, query_points, radius)
plot_query(point_cloud, query_points, epsilon_mask, radius)
