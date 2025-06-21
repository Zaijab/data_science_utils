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
    (prior_ensemble, true_state) = carry
    ensemble_updating_key, measurement_key = jax.random.split(key)
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
        ],
        Float[Array, "batch_size state_dim"],
    ],
    key: Key[Array, "..."],
    debug: bool = False,
) -> Float[Array, "..."]:

    burn_in_time = 100
    measurement_time = 10 * burn_in_time
    total_steps = burn_in_time + measurement_time

    initial_true_state = dynamical_system.initial_state()

    keys = jax.random.split(key, num=(total_steps,))

    scan_step = jax.tree_util.Partial(
        filter_update,
        dynamical_system=dynamical_system,
        measurement_system=measurement_system,
        update=update,
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

@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def ospa_metric(
    X: Float[Array, "m state_dim"],
    Y: Float[Array, "n state_dim"],
    c: float = 100.0,
    p: float = 2.0,
) -> tuple[
    float | Float[Array, ""], float | Float[Array, ""], float | Float[Array, ""]
]:
    """
    X: estimated states
    Y: ground truth states
    c: cut-off parameter
    p: order parameter
    """
    m, n = X.shape[0], Y.shape[0]

    # Empty set cases
    if m == 0 and n == 0:
        return 0.0, 0.0, 0.0
    if m == 0:
        # d_p^(c)(∅, Y) = c
        return c, 0.0, c
    if n == 0:
        # d_p^(c)(X, ∅) = c
        return c, c, 0.0

    # Compute distance matrix D_{ij} = ||x_i - y_j||
    D = jnp.sqrt(jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))

    # Apply cut-off: D^(c)_{ij} = min(D_{ij}, c)
    D_cut = jnp.minimum(D, c)

    # Cost matrix for assignment: C_{ij} = (D^(c)_{ij})^p
    C = D_cut**p

    # Solve assignment problem: π* = argmin_π Σ C_{i,π(i)}
    row_ind, col_ind = optax.assignment.hungarian_algorithm(C)

    # Assignment cost: Σ_{i=1}^{min(m,n)} (d^(c)(x_i, y_{π*(i)}))^p
    assignment_cost = C[row_ind, col_ind].sum()

    # Localization error: (1/min(m,n) * assignment_cost)^(1/p)
    e_loc = (assignment_cost / jnp.minimum(m, n)) ** (1 / p)

    # Cardinality error: c * |m - n| / max(m,n)
    e_card = c * jnp.abs(m - n) / jnp.maximum(m, n)

    # Total OSPA: ((assignment_cost + c^p * |m-n|) / max(m,n))^(1/p)
    ospa = ((assignment_cost + c**p * jnp.abs(m - n)) / jnp.maximum(m, n)) ** (1 / p)

    return ospa, e_loc, e_card
