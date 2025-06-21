import jax
import jax.numpy as jnp
from data_science_utils.measurement_systems import Radar


def test_radar_callable_transform() -> None:
    range_std = 1.0
    angle_std = 0.5 * jnp.pi / 180
    R = jnp.diag(jnp.array([range_std**2, angle_std**2, angle_std**2]))
    measurement_system = Radar(R)
    x = jnp.ones(6)
    measurement_system(x)
    jax.jacfwd(measurement_system)(x)

range_std = 1.0
angle_std = 0.5 * jnp.pi / 180
R = jnp.diag(jnp.array([range_std**2, angle_std**2, angle_std**2]))
measurement_system = Radar(R)
key = jax.random.key(0)
x, y = jax.random.normal(key, shape=(6,)), jax.random.normal(key, shape=(6,))
x, y = measurement_system(x), measurement_system(y)

K = jax.random.normal(key, shape=(6,3))
(K @ (x - y)).shape
