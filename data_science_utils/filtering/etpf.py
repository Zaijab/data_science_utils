import jax
import jax.numpy as jnp
from ott.solvers.linear import sinkhorn
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
import equinox as eqx
from jaxtyping import jaxtyped, Float, Array, Key
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def solve_optimal_transport(
    ensemble: Float[Array, "batch_size state_dim"],
    weights: Float[Array, "batch_size"],
    epsilon: float = 0.0001,
):
    ensemble_size: int = ensemble.shape[0]
    a = jnp.ones(ensemble_size) / ensemble_size
    assert isinstance(a, Float[Array, "batch_size"])

    b = weights
    assert isinstance(b, Float[Array, "batch_size"])

    geom = pointcloud.PointCloud(ensemble, ensemble, epsilon=epsilon)
    ot_problem = linear_problem.LinearProblem(geom, a=a, b=b)
    solver = sinkhorn.Sinkhorn(inner_iterations=200)
    ot_result = solver(ot_problem)
    transport_matrix = ot_result.matrix
    assert isinstance(transport_matrix, Float[Array, "batch_size batch_size"])

    return transport_matrix


def compute_weights(ensemble, measurement, measurement_system, key):
    keys = jax.random.split(key, ensemble.shape[0])

    def compute_likelihood(state, sub_key):
        predicted_measurement = measurement_system(state, sub_key)
        innovation = measurement - predicted_measurement
        exponent = -0.5 * (
            innovation.T @ jnp.linalg.inv(measurement_system.covariance) @ innovation
        )
        return jnp.exp(exponent)

    likelihoods = jax.vmap(compute_likelihood)(ensemble, keys)
    weights = likelihoods / jnp.sum(likelihoods)
    return weights


@eqx.filter_jit
def etpf_update(
    key,
    ensemble: Float[Array, "batch_size state_dim"],
    measurement,
    measurement_system,
    debug=False,
):
    weights = compute_weights(ensemble, measurement, measurement_system, key)
    if debug:
        jax.debug.print("Weights: {weights}", weights=weights)
    transport_matrix = solve_optimal_transport(ensemble, weights)
    print(transport_matrix)
    assert isinstance(transport_matrix, Float[Array, "batch_size batch_size"])

    jax.debug.callback(lambda: print("AHHHH"))
    # updated_ensemble = ensemble.T @ (ensemble.shape[0] * transport_matrix)
    return ensemble
