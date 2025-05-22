import equinox as eqx
import jax
import jax.numpy as jnp
from collections.abc import Callable
from jaxtyping import jaxtyped, Float, Array, Key
from beartype import beartype as typechecker
from data_science_utils.dynamical_systems import AbstractDynamicalSystem
from data_science_utils.measurement_functions import AbstractMeasurementSystem
import matplotlib.pyplot as plt


# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def filter_update(
    carry: tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"]],
    key: Key[Array, "..."],
    dynamical_system: AbstractDynamicalSystem,
    measurement_system: AbstractMeasurementSystem,
    update: Callable[
        [
            Key[Array, "..."],
            Float[Array, "batch_size state_dim"],
            Float[Array, "measurement_dim"],
            AbstractMeasurementSystem,
            bool,
        ],
        Float[Array, "batch_size state_dim"],
    ],
    debug: bool = False,
) -> tuple[
    tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"]],
    Float[Array, "state_dim"],
]:
    (prior_ensemble, true_state) = carry
    ensemble_updating_key, measurement_key = jax.random.split(key)
    updated_ensemble = update(
        ensemble=prior_ensemble,
        measurement=measurement_system(state=true_state, key=measurement_key),
        measurement_system=measurement_system,
        key=ensemble_updating_key,
        debug=debug,
    )
    error = true_state - jnp.mean(updated_ensemble, axis=0)
    if debug:
        # jax.debug.print("Hello")
        jax.debug.callback(plot_update, prior_ensemble, updated_ensemble, true_state)
    ensemble_next = dynamical_system.forward(updated_ensemble)
    true_state_next = dynamical_system.forward(true_state)
    new_carry = (ensemble_next, true_state_next)
    return new_carry, error


# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def evaluate_filter(
    initial_ensemble: Float[Array, "..."],
    dynamical_system: AbstractDynamicalSystem,
    measurement_system: AbstractMeasurementSystem,
    update: Callable[
        [
            Key[Array, "..."],
            Float[Array, "batch_size state_dim"],
            Float[Array, "measurement_dim"],
            AbstractMeasurementSystem,
            bool,
        ],
        Float[Array, "batch_size state_dim"],
    ],
    key: Key[Array, "..."],
    debug: bool = False,
) -> Float[Array, "..."]:

    burn_in_time = 100
    measurement_time = 10 * burn_in_time
    total_steps = burn_in_time + measurement_time

    initial_true_state = dynamical_system.initial_state

    keys = jax.random.split(key, num=(total_steps,))

    scan_step = jax.tree_util.Partial(
        filter_update,
        dynamical_system=dynamical_system,
        measurement_system=measurement_system,
        update=update,
        debug=debug,
    )

    (final_carry, errors_over_time) = jax.lax.scan(
        scan_step, (initial_ensemble, initial_true_state), keys
    )

    errors_past_burn_in = errors_over_time[burn_in_time:]
    rmse = jnp.sqrt(jnp.mean(errors_past_burn_in**2))

    return rmse


def plot_update(prior_ensemble, posterior_ensemble, true_state):
    print(prior_ensemble.shape)
    print(posterior_ensemble.shape)
    print(true_state.shape)
    print(jnp.mean(prior_ensemble, axis=0).shape)
    print(true_state - jnp.mean(prior_ensemble, axis=0))
    prior_error = jnp.sqrt(
        jnp.mean((true_state - jnp.mean(prior_ensemble, axis=0)) ** 2)
    )
    posterior_error = jnp.sqrt(
        jnp.mean((true_state - jnp.mean(posterior_ensemble, axis=0)) ** 2)
    )
    plt.title(f"Prior Error {prior_error:.4f} | Posterior Error {posterior_error:.4f}")
    plt.scatter(prior_ensemble[:, 0], prior_ensemble[:, 1], c="red", label="Prior")
    plt.scatter(
        posterior_ensemble[:, 0], posterior_ensemble[:, 1], c="blue", label="Posterior"
    )
    plt.scatter(true_state[..., 0], true_state[..., 1], c="lime", label="True", s=100)
    plt.legend(bbox_to_anchor=(1.3, 1), loc="upper right")
    plt.show()
