import jax
import jax.numpy as jnp

from data_science_utils.dynamical_systems.crtbp import CR3BP
from data_science_utils.filters import EnGMF, evaluate_filter
from data_science_utils.filters.evaluate_imperfect_model import evaluate_filter_imperfect
from data_science_utils.measurement_systems import Radar, RangeSensor
from diffrax import Tsit5, ConstantStepSize

key = jax.random.key(1010)
key, subkey = jax.random.split(key)


dynamical_system = CR3BP()
measurement_system = RangeSensor()
stochastic_filter = EnGMF()



key, subkey = jax.random.split(key)
rmse = evaluate_filter(
    dynamical_system,
    measurement_system,
    stochastic_filter,
    subkey,
)

print(f"RMSE {rmse}")
print("Hellora")
