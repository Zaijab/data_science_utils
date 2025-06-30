from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from beartype import beartype as typechecker
from data_science_utils.dynamical_systems import AbstractDynamicalSystem
from data_science_utils.measurement_systems import AbstractMeasurementSystem
from jaxtyping import Array, Float, Key, jaxtyped

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
    (prior_ensemble, true_state, fuel) = carry
    ensemble_updating_key, measurement_key, to_maneuver_key, maneuver_key = jax.random.split(key, 4)
    # p(y|GMM) = Σ_i w_i * N(y; H*μ_i, H*Σ_i*H^T + R)
    # measurement_probability_wrt_gmm = 10
    # jax.debug.print()
    updated_ensemble = update(
        prior_ensemble=prior_ensemble,
        measurement=measurement_system(true_state, measurement_key),
        measurement_system=measurement_system,
        key=ensemble_updating_key,
    )
    error = true_state - jnp.mean(updated_ensemble, axis=0)
    
    if debug:
        jax.debug.callback(plot_update, prior_ensemble, updated_ensemble, true_state)
    ensemble_next = eqx.filter_vmap(dynamical_system.flow)(0.0, 1.0, updated_ensemble)
    true_state_next = dynamical_system.flow(0.0, 1.0, true_state)

    def apply_maneuver(true_state_next, fuel, maneuver_key):
        return_value = maneuver_normal_with_fuel(true_state_next, fuel, maneuver_key)
        # jax.debug.print("I moved {} {}", fuel, return_value[1])
        return return_value

    def no_maneuver(true_state_next, fuel, maneuver_key):
        # jax.debug.print("I didn't move. {}", fuel)
        return true_state_next, fuel

    true_state_next, fuel = jax.lax.cond(
        jax.random.bernoulli(to_maneuver_key),
        apply_maneuver,
        no_maneuver,
        true_state_next, fuel, maneuver_key
    )
    new_carry = (ensemble_next, true_state_next, fuel)
    return new_carry, error


# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def evaluate_filter_imperfect(
    dynamical_system: AbstractDynamicalSystem,
    measurement_system: AbstractMeasurementSystem,
    update: Callable[
        [
            Key[Array, "..."],
            Float[Array, "batch_size state_dim"],
            Float[Array, "measurement_dim"],
            AbstractMeasurementSystem,
        ],
        Float[Array, "batch_size state_dim"],
    ],
    key: Key[Array, "..."],
    debug: bool = False,
) -> Float[Array, "..."]:
    # Should Prob Be static args ^
    ensemble_size = 100
    burn_in_time = 100
    measurement_time = 10 * burn_in_time
    total_steps = burn_in_time + measurement_time

    key, subkey = jax.random.split(key)
    initial_ensemble = dynamical_system.generate(subkey, batch_size=ensemble_size)
    initial_true_state = dynamical_system.initial_state()

    keys = jax.random.split(key, num=(total_steps,))

    scan_step = jax.tree_util.Partial(
        filter_update,
        dynamical_system=dynamical_system,
        measurement_system=measurement_system,
        update=update.update,
    )

    (final_carry, errors_over_time) = jax.lax.scan(
        scan_step, (initial_ensemble, initial_true_state, 10000.0), keys
    )

    errors_past_burn_in = errors_over_time[burn_in_time:]
    rmse = jnp.sqrt(jnp.mean(errors_past_burn_in**2))

    return rmse


# Change total fuel to be 10 or 100
# Use JPL CR3BP look for initial conditions identity matrix initial covariance
# Filter with sparse measurements
# Use measurement covariances from Schroeder
# Read through sensor tasking algorithms
# Implement the windowing for tracking

# 2 column 6 page limit

def maneuver_normal_with_fuel(
    state: Float[Array, "6"],
    fuel: float,
    key: Key[Array, "..."]
) -> tuple[Float[Array, "6"], float]:
    # jax.debug.print("Fuel {}", fuel)
    delta_v_raw = jax.random.normal(key, (3,))
    delta_v_magnitude = jnp.linalg.norm(delta_v_raw)
    maneuver_magnitude = jnp.minimum(delta_v_magnitude, fuel)
    delta_v = jnp.where(fuel > 0, maneuver_magnitude * delta_v_raw / delta_v_magnitude, jnp.zeros(3))
    maneuvered_state = state.at[3:6].add(delta_v)
    return maneuvered_state, fuel - maneuver_magnitude
