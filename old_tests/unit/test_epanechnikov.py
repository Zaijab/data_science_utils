import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from data_science_utils.statistics import pdf_epanechnikov, sample_epanechnikov

# Set the parameters
mu = jnp.zeros(2)
sigma = jnp.eye(2)

# Create a grid of points in 2D space
x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
X, Y = np.meshgrid(x, y)

# Flatten the grid to pass through the pdf function
xy_points = np.vstack([X.ravel(), Y.ravel()]).T

# Calculate the PDF values
Z = jax.vmap(pdf_epanechnikov, in_axes=(0, None, None))(xy_points, mu, sigma)
Z = Z.reshape(X.shape)

# Plot the contour plot
plt.figure(figsize=(8, 8))
plt.contour(X, Y, Z, levels=10, cmap="viridis")
plt.title("Contour Plot of Epanechnikov PDF")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()

# Plot the surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
ax.set_title("Surface Plot of Epanechnikov PDF")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Density")
ax.set_box_aspect(None, zoom=0.90)
plt.tight_layout()
plt.show()


# Set the parameters
samples = sample_epanechnikov(
    jax.random.key(0), jnp.zeros(2) + 10, jnp.array([[10, 1], [1, 1]]), 10000
)


sns.jointplot(x=samples[:, 0], y=samples[:, 1])
plt.show()
