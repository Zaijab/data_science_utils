import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ikeda import IkedaSystem

key = jax.random.key(0)
key, subkey = jax.random.split(key)
ikeda = IkedaSystem(state=jnp.array([[1.0,2.0]]), u=0.9)
batch = ikeda.generate(subkey)
batch = ikeda.flow(batch)
ikeda.iterate()
print(ikeda.state.shape)
grid_spacing = 500
x = jnp.linspace(-0.5, 2, grid_spacing)
y = jnp.linspace(-2.5, 1, grid_spacing)
XX, YY = jnp.meshgrid(x, y)
grid = jnp.dstack([XX, YY])
grid_points = grid.reshape(-1, 2)
threshold = 0.01

import matplotlib.pyplot as plt

labels = ikeda_attractor_discriminator(grid_points)
labels_grid = labels.reshape(grid_spacing,grid_spacing)
plt.figure(figsize=(8, 6))
plt.contourf(XX, YY, labels_grid)
plt.title('Discriminator Output')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
