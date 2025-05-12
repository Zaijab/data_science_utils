import jax
import jax.numpy as jnp
from data_science_utils.dynamical_systems import Ikeda


def test_import_ikeda():
    system = Ikeda()
    true_state = system.initial_state
    ensemble_size = 10
    key = jax.random.key(0)
    initial_ensemble = jax.random.multivariate_normal(
        key,
        shape=(ensemble_size,),
        mean=true_state,
        cov=1 * jnp.eye(true_state.shape[-1]),
    )

    initial_ensemble
