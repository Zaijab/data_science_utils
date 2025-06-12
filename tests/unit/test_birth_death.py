import equinox as eqx
import jax
import jax.numpy as jnp


rfs = RFS(
    jnp.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [3, 6, 9]], dtype=jnp.float64),
    jnp.array([True, False, True, False]),
)


key = jax.random.key(0)
key, subkey = jax.random.split(key)
clutter_region = jnp.array([[0.0, 200.0], [0.0, 200.0], [0.0, 400.0]])
clutter_average_rate = 10.0
clutter_max_points = 40

clutter_rfs = poisson_point_process_rectangular_region(
    subkey,
    clutter_average_rate,
    clutter_region,
    clutter_max_points,
)
print(clutter_rfs)

# This works so far.
rfs = rfs.insert(jnp.array([[10, 20, 30], [10, 20, 30]], dtype=jnp.float64))
print(rfs.mask)
print(rfs.state)
