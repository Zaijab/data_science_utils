import jax
import jax.numpy as jnp

from data_science_utils.dynamical_systems.crtbp import CR3BP
from data_science_utils.filters import EnGMF, evaluate_filter
from data_science_utils.measurement_systems import Radar, RangeSensor

key = jax.random.key(1010)
key, subkey = jax.random.split(key)
# dynamical_system = Ikeda()


dynamical_system = CR3BP()
measurement_system = RangeSensor(jnp.array([[0.25]]))
measurement_system = Radar(jnp.diag(jnp.array([
    0.25,
    (0.05 * jnp.pi / 180) ** 2,
    (0.05 * jnp.pi / 180) ** 2
])))
# measurement_system = Radar()

filter_update = EnGMF().update
ensemble_size = 1000

key, subkey = jax.random.split(key)
initial_ensemble = dynamical_system.generate(subkey, batch_size=ensemble_size)

key, subkey = jax.random.split(key)
rmse = evaluate_filter(
    initial_ensemble,
    dynamical_system,
    measurement_system,
    filter_update,
    subkey,
)
print(f"RMSE {rmse}")
