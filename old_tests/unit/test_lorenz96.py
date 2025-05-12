import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from data_science_utils.dynamical_systems import Lorenz96

from mpl_toolkits.mplot3d import Axes3D
# Initialize the Lorenz96 system
system = Lorenz96(dim=40, F=8.0, steps=1)

# Get initial state
x0 = system.initial_state

# Simulate for 3000 steps
num_steps = 3000
trajectory = jnp.zeros((num_steps + 1, system.dimension))
trajectory = trajectory.at[0].set(x0)

# Perform simulation
x = x0
for i in range(1, num_steps + 1):
    x = system.forward(x[None, ...])[0]  # Add and remove batch dimension
    trajectory = trajectory.at[i].set(x)

# Create 3D plot of first three variables
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
ax.set_title("Lorenz96 System: First Three Variables")
plt.savefig("plots/lorenz96.")
plt.show()

