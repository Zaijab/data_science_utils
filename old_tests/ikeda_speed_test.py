import jax.numpy as jnp
import matplotlib.pyplot as plt

from data_science_utils.dynamical_systems import ikeda_attractor_discriminator

grid_spacing = 1000
x = jnp.linspace(-0.5, 2, grid_spacing)
y = jnp.linspace(-2.5, 1, grid_spacing)
XX, YY = jnp.meshgrid(x, y)
grid = jnp.dstack([XX, YY])
grid_points = grid.reshape(-1, 2)

labels = ikeda_attractor_discriminator(grid_points, ninverses=10)
labels_grid = labels.reshape(grid_spacing, grid_spacing)

plt.figure(figsize=(8, 6))
plt.contourf(XX, YY, labels_grid)
plt.title('Discriminator Output')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
