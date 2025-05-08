import jax
import jax.numpy as jnp
from ott.solvers.linear import sinkhorn
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
import equinox as eqx
from jaxtyping import jaxtyped, Float, Array, Key
from beartype import beartype as typechecker
from typing import Any
from data_science_utils.measurement_functions import AbstractMeasurementSystem
from evosax import CMA_ES


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def solve_ot_problem_correct(
    ensemble: Float[Array, "..."], a: Float[Array, "..."], b: Float[Array, "..."]
) -> Float[Array, "..."]:
    epsilon = 0.05
    # cost_matrix = jnp.sum((ensemble[:, None, :] - ensemble[None, :, :]) ** 2, axis=-1)
    geom = pointcloud.PointCloud(ensemble, ensemble, epsilon=epsilon)
    ot_problem = linear_problem.LinearProblem(geom, a=a, b=b)

    solver = sinkhorn.Sinkhorn(
        inner_iterations=10,
        threshold=1e-6,
        max_iterations=10_000_000,
        lse_mode=True,
    )
    ot_result = solver(ot_problem)
    transport_matrix = ot_result.matrix
    return transport_matrix


@eqx.filter_jit
def solve_ot_problem_genetic(ensemble: Any, a: Any, b: Any) -> Any:

    rng = jax.random.key(100)
    ensemble_size = ensemble.shape[0]
    max_gens = 10000
    population_size = 30
    tol = 1e-6
    strategy = CMA_ES(num_dims=ensemble_size**2, popsize=population_size)
    es_params = strategy.default_params
    state = strategy.initialize(rng, es_params)
    state = state.replace(best_fitness=-jnp.finfo(jnp.float64).max)

    # cost_matrix = jnp.sum((ensemble[:, None, :] - ensemble[None, :, :]) ** 2, axis=-1)

    @jax.vmap
    def transport_matrix_fitness(T: Any) -> Any:
        T = jax.nn.relu(T.reshape(ensemble_size, ensemble_size))
        # transport_cost = jnp.sum(cost_matrix * T)
        a_diff = jnp.sum(T, axis=1) - a
        b_diff = jnp.sum(T, axis=0) - b
        return jnp.sum((a_diff**2) + (b_diff**2))  # + 0.05 * transport_cost

    def cond_fun(carry):
        _, state, iteration = carry
        return state.best_fitness > tol

    def body_fun(carry):
        rng, state, iteration = carry
        rng, rng_ask = jax.random.split(rng)
        x, state = strategy.ask(rng_ask, state, es_params)
        fitness = transport_matrix_fitness(x)
        state = strategy.tell(x, fitness, state, es_params)
        return (rng, state, iteration + 1)

    init_carry = (rng, state, 0)
    _, final_state, _ = eqx.internal.while_loop(
        cond_fun, body_fun, init_carry, max_steps=max_gens, kind="bounded"
    )

    T = final_state.best_member.reshape(ensemble_size, ensemble_size)
    T = jax.nn.relu(T)
    return T


solver = jax.tree_util.Partial(solve_ot_problem_correct)


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

    transport_matrix = solver(ensemble, a, b)
    # jax.debug.callback(inspect_marginals, transport_matrix, a, b)

    assert isinstance(transport_matrix, Float[Array, "batch_size batch_size"])

    return transport_matrix


def compute_weights(
    ensemble: Float[Array, "batch_size state_dim"],
    measurement: Float[Array, "measurement_dim"],
    measurement_system: AbstractMeasurementSystem,
) -> Float[Array, "batch_size"]:

    def compute_log_likelihood(state: Float[Array, "state_dim"]) -> Float[Array, "1"]:
        predicted_measurement = measurement_system(state)
        innovation = measurement - predicted_measurement
        log_likelihood = -0.5 * (
            innovation.T @ jnp.linalg.inv(measurement_system.covariance) @ innovation
        )
        return log_likelihood

    log_likelihoods = jax.vmap(compute_log_likelihood)(ensemble)
    log_likelihoods -= jnp.max(log_likelihoods)
    weights = jnp.exp(log_likelihoods)
    return weights / jnp.sum(weights)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def etpf_update(
    key: Key[Array, "..."],
    ensemble: Float[Array, "batch_size state_dim"],
    measurement: Float[Array, "measurement_dim"],
    measurement_system: AbstractMeasurementSystem,
    debug: bool = False,
) -> Float[Array, "batch_size state_dim"]:
    weights = compute_weights(ensemble, measurement, measurement_system)
    assert isinstance(weights, Float[Array, "batch_size"])

    transport_matrix = solve_optimal_transport(ensemble, weights)
    assert isinstance(transport_matrix, Float[Array, "batch_size batch_size"])

    P = ensemble.shape[0] * transport_matrix
    updated_ensemble = P @ ensemble
    assert isinstance(updated_ensemble, Float[Array, "batch_size state_dim"])

    # Rejuvenation
    tau = 0.2  # alpha = \sqrt{1 + tau} = ~1.01
    cov_matrix = tau * jnp.cov(updated_ensemble, rowvar=False) + jnp.ones(2) * 1e-7

    noise_key, new_key = jax.random.split(key)
    noise = jax.random.multivariate_normal(
        noise_key,
        mean=jnp.zeros(ensemble.shape[1]),
        cov=cov_matrix,
        shape=(ensemble.shape[0],),
    )

    updated_ensemble = updated_ensemble + noise

    return updated_ensemble


# Create callbacks to inspect values during runtime
def inspect_marginals(matrix, a, b):
    row_sums = jnp.sum(matrix, axis=1)
    col_sums = jnp.sum(matrix, axis=0)
    row_err = jnp.max(jnp.abs(row_sums - a))
    col_err = jnp.max(jnp.abs(col_sums - b))
    print(f"Row marginal Error: {row_err}")
    print(f"Column marginal Error: {col_err}")


def markov_marginals(matrix, ensemble):
    row_sums = jnp.sum(matrix, axis=1)
    col_sums = jnp.sum(matrix, axis=0)
    print(f"Row marginal error: {row_sums}")
    print(f"Column marginal error: {col_sums}")
    print(ensemble.shape)
