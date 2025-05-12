import jax
import jax.numpy as jnp
import equinox as eqx

from ott.solvers.linear import sinkhorn
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem

from evosax import CMA_ES, RmES, SimAnneal, OpenES
from typing import Any

key = jax.random.key(0)


ensemble_size = 10
key, subkey = jax.random.split(key)
ensemble = jax.random.multivariate_normal(
    subkey, shape=(ensemble_size,), mean=jnp.array([1.0, 0]), cov=jnp.eye(2)
)
a = jnp.ones(ensemble_size) / ensemble_size
b = jax.random.uniform(subkey, shape=(ensemble_size,))
b = b / jnp.sum(b)


@eqx.filter_jit
def solve_ot_problem_correct(ensemble: Any, a: Any, b: Any) -> Any:
    epsilon = 0.01
    geom = pointcloud.PointCloud(ensemble, ensemble, epsilon=epsilon)
    ot_problem = linear_problem.LinearProblem(geom, a=a, b=b)

    solver = sinkhorn.Sinkhorn(
        inner_iterations=1,
        threshold=1e-6,
        max_iterations=1_000_000,
        lse_mode=True,
    )
    ot_result = solver(ot_problem)
    transport_matrix = ot_result.matrix
    return transport_matrix


@eqx.filter_jit
def solve_ot_problem_genetic(ensemble: Any, a: Any, b: Any) -> Any:

    rng = jax.random.key(101)
    ensemble_size = ensemble.shape[0]
    max_gens = 1_000_000
    population_size = 25
    tol = 1e-8
    strategy = CMA_ES(num_dims=ensemble_size**2, popsize=population_size)
    es_params = strategy.default_params

    state = strategy.initialize(rng, es_params)
    state = state.replace(best_fitness=jnp.finfo(jnp.float64).max)
    # cost_matrix = jnp.sum((ensemble[:, None, :] - ensemble[None, :, :]) ** 2, axis=-1)

    @jax.vmap
    def transport_matrix_fitness(T: Any) -> Any:
        T = T.reshape(ensemble_size, ensemble_size)
        # transport_cost = jnp.sum(cost_matrix * T)
        # probability_mass = jnp.sum(T)
        a_diff = jnp.sum(T, axis=1) - a
        b_diff = jnp.sum(T, axis=0) - b

        return (
            jnp.sum((a_diff**2) + (b_diff**2))
            # + 0.005 * transport_cost
            # + ((probability_mass - 1) ** 2)
        )

    def cond_fun(carry):
        _, state, iteration = carry
        jax.debug.print("{}", state.best_fitness)
        return state.best_fitness > tol

    def body_fun(carry):
        rng, state, iteration = carry
        rng, rng_ask = jax.random.split(rng)
        x, state = strategy.ask(rng_ask, state, es_params)
        x = jax.nn.relu(x)
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


transport_matrix_correct = solve_ot_problem_correct(ensemble, a, b)

print(transport_matrix_correct)

transport_matrix_genetic = solve_ot_problem_genetic(ensemble, a, b)

print(transport_matrix_genetic)

print(jnp.isclose(jnp.sum(transport_matrix_genetic, axis=1), a, atol=1e-4))
print(jnp.isclose(jnp.sum(transport_matrix_genetic, axis=0), b, atol=1e-4))
print(jnp.isclose(transport_matrix_genetic, transport_matrix_correct, atol=1e-4))
print(jnp.sum(transport_matrix_genetic, axis=1), a)
print(jnp.sum(transport_matrix_genetic, axis=0), b)
