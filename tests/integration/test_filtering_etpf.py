import jax
import jax.numpy as jnp
from jaxtyping import jaxtyped, Float, Array, Key
from beartype import beartype as typechecker
import equinox as eqx

from typing import Tuple, Callable, Protocol
import jax
import jax.numpy as jnp
from ott.solvers.linear import sinkhorn
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
import equinox as eqx
from jaxtyping import jaxtyped, Float, Array, Key
from beartype import beartype as typechecker

from data_science_utils.dynamical_systems import Ikeda
from data_science_utils.measurement_functions import RangeSensor
from data_science_utils.filtering import etpf_update, evaluate_filter


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def solve_optimal_transport(
    ensemble: Float[Array, "batch_size state_dim"],
    weights: Float[Array, "batch_size"],
    epsilon: float = 0.005,
) -> Float[Array, "batch_size batch_size"]:
    ensemble_size: int = ensemble.shape[0]
    a = jnp.ones(ensemble_size) / ensemble_size
    assert isinstance(a, Float[Array, "batch_size"])

    b = weights
    assert isinstance(b, Float[Array, "batch_size"])

    geom = pointcloud.PointCloud(ensemble, ensemble, epsilon=epsilon)
    ot_problem = linear_problem.LinearProblem(geom, a=a, b=b)
    solver = sinkhorn.Sinkhorn(
        inner_iterations=1,
        threshold=0.001,
        max_iterations=1_000_000,
        lse_mode=True,
    )
    ot_result = solver(ot_problem)
    jax.debug.print("{}", ot_result.converged)
    transport_matrix = ot_result.matrix

    assert isinstance(transport_matrix, Float[Array, "batch_size batch_size"])

    # jax.debug.callback(
    #     inspect_marginals,
    #     transport_matrix,
    #     jnp.ones_like(weights) / len(weights),
    #     weights,
    # )

    return transport_matrix


# Create callbacks to inspect values during runtime
def inspect_marginals(matrix, a, b):
    row_sums = jnp.sum(matrix, axis=1)
    col_sums = jnp.sum(matrix, axis=0)
    row_err = jnp.max(jnp.abs(row_sums - a))
    col_err = jnp.max(jnp.abs(col_sums - b))
    print(f"Row marginal error: {row_err}")
    print(f"Column marginal error: {col_err}")


def compute_weights(
    ensemble: Float[Array, "batch_size state_dim"],
    measurement: Float[Array, "measurement_dim"],
    measurement_system: RangeSensor,
    key: Key[Array, "..."],
) -> Float[Array, "batch_size"]:
    keys = jax.random.split(key, ensemble.shape[0])

    def compute_likelihood(
        state: Float[Array, "state_dim"], sub_key: Key[Array, "..."]
    ) -> Float[Array, "1"]:
        predicted_measurement = measurement_system(state, None)
        innovation = measurement - predicted_measurement
        exponent = -0.5 * (
            innovation.T @ jnp.linalg.inv(measurement_system.covariance) @ innovation
        )
        return jnp.exp(exponent)

    likelihoods = jax.vmap(compute_likelihood)(ensemble, keys)
    weights = likelihoods / jnp.sum(likelihoods)
    return weights


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def etpf_update(
    key: Key[Array, "..."],
    ensemble: Float[Array, "batch_size state_dim"],
    measurement: Float[Array, "measurement_dim"],
    measurement_system: RangeSensor,
    debug: bool = False,
) -> Float[Array, "batch_size state_dim"]:
    weights = compute_weights(ensemble, measurement, measurement_system, key)
    assert isinstance(weights, Float[Array, "batch_size"])

    transport_matrix = solve_optimal_transport(ensemble, weights)
    assert isinstance(transport_matrix, Float[Array, "batch_size batch_size"])

    updated_ensemble = (ensemble.shape[0] * transport_matrix) @ ensemble
    assert isinstance(updated_ensemble, Float[Array, "batch_size state_dim"])

    return updated_ensemble


@eqx.filter_jit
def filter_update(
    carry: Tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"]],
    key: Key[Array, "..."],
    dynamical_system: Ikeda,
    measurement_system: RangeSensor,
    update: Callable,
    debug: bool = False,
) -> Tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"]]:
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
    ensemble_next = dynamical_system.forward(updated_ensemble)
    true_state_next = dynamical_system.forward(true_state)
    new_carry = (ensemble_next, true_state_next)
    return new_carry, error


@eqx.filter_jit
def evaluate_filter(
    initial_ensemble: Float[Array, "batch_size state_dim"],  # prior belief
    dynamical_system: Ikeda,  # state system, predict step
    measurement_system: RangeSensor,  # measurement system
    update: Callable,  # Update
    key: Key[Array, "..."],  # omega
    debug: bool = False,
) -> Float[Array, "..."]:
    burn_in_time = 10
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


@eqx.filter_jit
def enkf_update(
    key: Key[Array, "..."],
    ensemble: Float[Array, "batch_size state_dim"],
    measurement: Float[Array, "batch_size measurement_dim"],
    measurement_system: RangeSensor,
    inflation_factor: float = 1.01,
    debug: bool = False,
) -> Float[Array, "batch_size state_dim"]:
    mean = jnp.mean(ensemble, axis=0)

    if debug:
        jax.debug.print("{shape}", mean.shape)

    inflated = mean + inflation_factor * (ensemble - mean)

    ensemble_covariance = jnp.cov(inflated.T)

    @jax.jit
    def update_ensemble_point(
        point: Float[Array, "state_dim"], key: Key[Array, "..."]
    ) -> Float[Array, "state_dim"]:
        point_measurement = measurement_system(point)
        measurement_jacobian = jax.jacfwd(measurement_system)(point)
        innovation_covariance = (
            measurement_jacobian @ ensemble_covariance @ measurement_jacobian.T
            + measurement_system.covariance
        )
        kalman_gain = (
            ensemble_covariance
            @ measurement_jacobian.T
            @ jnp.linalg.inv(innovation_covariance)
        )
        point = point + (
            kalman_gain @ jnp.atleast_2d(measurement - point_measurement)
        ).reshape(-1)
        return point

    keys = jax.random.split(key, ensemble.shape[0])
    updated_ensemble = jax.vmap(update_ensemble_point)(inflated, keys)
    return updated_ensemble


key = jax.random.key(105)
key, subkey = jax.random.split(key)
dynamical_system = Ikeda(u=0.9)
measurement_system = RangeSensor(jnp.array([[0.25]]))
true_state = dynamical_system.initial_state
ensemble_size = 150
ensemble = jax.random.multivariate_normal(
    subkey,
    shape=(ensemble_size,),
    mean=true_state,
    cov=0.25 * jnp.eye(2),
)


etpf_rmse = evaluate_filter(
    ensemble,
    dynamical_system,
    measurement_system,
    etpf_update,
    key,
)

enkf_rmse = evaluate_filter(
    ensemble,
    dynamical_system,
    measurement_system,
    enkf_update,
    key,
)


print(f"ETPF: {etpf_rmse} | EnKF: {enkf_rmse}")
