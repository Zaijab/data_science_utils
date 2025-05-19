import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Key, jaxtyped
from beartype import beartype as typechecker
from jax import random, lax
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, ConstantStepSize


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax
import jax.numpy as jnp


lorenz = Lorenz63()


attractor = lorenz.generate(jax.random.key(0), batch_size=10_000, spin_up_steps=3000)

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Plot the attractor points
ax.scatter(
    attractor[:, 0],
    attractor[:, 1],
    attractor[:, 2],
    s=5,
    alpha=0.5,
)
plt.show()
