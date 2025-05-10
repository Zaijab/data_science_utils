from data_science_utils.statistics import logpdf_epanechnikov
from data_science_utils.models import InvertibleNN

import equinox as eqx
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt


grid_spacing = 500
x = jnp.linspace(-0.5, 2, grid_spacing)
y = jnp.linspace(-2.5, 1, grid_spacing)
XX, YY = jnp.meshgrid(x, y)
grid = jnp.dstack([XX, YY])
grid_points = grid.reshape(-1, 2)
threshold = 0.01
model = InvertibleNN(
    hidden_dim=32, num_coupling_layers=6, num_hidden_layers=2, key=subkey
)
model = eqx.tree_deserialise_leaves("cache/ikeda_nf.eqx", model)


@jax.jit
def discriminator(points, threshold=0.1, normalizing_flow=model):
    points = jnp.atleast_2d(points)
    z, log_det_jacobian = eqx.filter_vmap(model.inverse)(points)
    log_det_jacobian = jnp.nan_to_num(log_det_jacobian)
    logpdf_values = jax.vmap(logpdf_epanechnikov, in_axes=(0, None, None))(
        z, jnp.zeros(2), jnp.eye(2)
    )
    total_logprob = logpdf_values + log_det_jacobian
    p_x = jnp.exp(total_logprob)
    if threshold is None:
        return p_x
    else:
        return jnp.where(p_x > threshold, 1, 0)


labels = discriminator(grid_points)
labels_grid = labels.reshape(grid_spacing, grid_spacing)
plt.figure(figsize=(8, 6))
plt.contourf(XX, YY, labels_grid)
plt.title("Discriminator Output")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
