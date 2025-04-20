import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def filter_update(
    carry,
    key,
    dynamical_system,
    measurement_system,
    update,
    debug=False,
):
    """
    carry = (prior_ensemble, true_state)
    key = prngkey for noisy measurement + noisy dynamics
    dynamical_system = eqx.Module representing forward dynamics
    measurement_system = eqx.Module representing measurements
    """
    (prior_ensemble, true_state) = carry

    updated_ensemble = update(
        ensemble=prior_ensemble,
        measurement=measurement_system(state=true_state),
        measurement_system=measurement_system,
        key=key,
        debug=debug,
    )

    error = true_state - jnp.mean(updated_ensemble, axis=0)

    ensemble_next = dynamical_system.forward(updated_ensemble)
    true_state_next = dynamical_system.forward(true_state)

    new_carry = (ensemble_next, true_state_next)
    return new_carry, error


@eqx.filter_jit
def evaluate_filter(
    initial_ensemble,  # prior belief
    dynamical_system,  # state system, predict step
    measurement_system,  # measurement system
    update,  # Update
    key,  # omega
    debug=False,
):
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
