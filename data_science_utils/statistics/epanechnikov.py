"""
In this file, we provide a number of functions relating to the Epanechnikov RVs.
For some reason, very little stats packages provide such utilities.
We also provide functions which return the pdf and logpdf values up to a constant for speed.
"""

# import jax.numpy as jnp
# from jax import jit
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

@jit
def pdf_epanechnikov(x, mu, sigma):
    n = mu.shape[0]
    x_centered = x - mu
    mahalanobis_dist = jnp.sum(x_centered * jnp.linalg.solve(sigma, x_centered.T).T, axis=1)
    constant = (n + 2) / 2 / (jnp.pi ** (n / 2)) / jnp.exp(gammaln(n / 2 + 1)) / ((n + 4) ** (n / 2))
    density = jnp.where(
        mahalanobis_dist <= (n + 4),
        constant * (1 - mahalanobis_dist / (n + 4)),
        0.0
    )
    return density

@jit
def logpdf_epanechnikov(x, mu, sigma):
    return jnp.log(pdf_epanechnikov(x, mu, sigma))


@jit
def logpdf_epanechnikov(x, mu, sigma):
    n = mu.shape[0]
    x_centered = x - mu
    # Compute the Mahalanobis distance
    
    
    L = jnp.linalg.cholesky(sigma)
    y = jnp.linalg.solve(L, x_centered.T).T
    mahalanobis_dist = jnp.sum(y ** 2, axis=1)
    
    #mahalanobis_dist = jnp.sum(x_centered * jnp.linalg.solve(sigma, x_centered.T).T, axis=1)
    # Compute the logarithm of the constant term
    log_constant = (
        jnp.log(n + 2)
        - jnp.log(2)
        - (n / 2) * jnp.log(jnp.pi)
        - gammaln(n / 2 + 1)
        - (n / 2) * jnp.log(n + 4)
    )
    # Compute the logarithm of the density
    log_density = jnp.where(
        mahalanobis_dist <= (n + 4),
        log_constant + jnp.log1p(-mahalanobis_dist / (n + 4)),
        -jnp.inf  # Logarithm of zero for points outside the support
    )
    return log_density


# # Set the parameters
# mu = jnp.array([0.0, 0.0])
# sigma = jnp.array([[1.0, 0.0], [0.0, 1.0]])

# # Create a grid of points in 2D space
# x = np.linspace(-3, 3, 1000)
# y = np.linspace(-3, 3, 1000)
# X, Y = np.meshgrid(x, y)

# # Flatten the grid to pass through the pdf function
# xy_points = np.vstack([X.ravel(), Y.ravel()]).T

# # Calculate the PDF values
# Z = pdf_epanechnikov(xy_points, mu, sigma)
# Z = Z.reshape(X.shape)

# # Plot the contour plot
# plt.figure(figsize=(8, 8))
# plt.contour(X, Y, Z, levels=10, cmap='viridis')
# plt.title('Contour Plot of Epanechnikov PDF')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.grid(True)
# plt.show()

# # Plot the surface plot
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
# ax.set_title('Surface Plot of Epanechnikov PDF')
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Density')
# ax.set_box_aspect(None, zoom=0.90)
# plt.tight_layout()
# plt.show()


