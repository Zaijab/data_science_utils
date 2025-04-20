import jax

from ott.solvers.linear import sinkhorn
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
import equinox as eqx


@eqx.filter_jit
def solve_optimal_transport(ensemble, weights, epsilon=0.0001):
    ensemble_size = ensemble.shape[0]
    a = jnp.ones(ensemble_size) / ensemble_size
    b = weights
    geom = pointcloud.PointCloud(ensemble, ensemble, epsilon=epsilon)
    ot_problem = linear_problem.LinearProblem(geom, a=a, b=b)
    solver = sinkhorn.Sinkhorn(inner_iterations=200)
    ot_result = solver(ot_problem)
    transport_matrix = ot_result.matrix
    return transport_matrix


def compute_weights(ensemble, measurement, measurement_system, key):
    keys = jax.random.split(key, ensemble.shape[0])

    # For each ensemble member, compute likelihood π_Y(y_0|x_i^f)
    def compute_likelihood(state, sub_key):
        predicted_measurement = measurement_system(state, sub_key)
        innovation = measurement - predicted_measurement
        # For Gaussian measurement errors, likelihood is proportional to:
        # exp(-0.5 * (innovation)^T * R^-1 * (innovation))
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
    ensemble,
    measurement,
    measurement_system,
    debug=False,
):
    # 1. Compute importance weights
    weights = compute_weights(ensemble, measurement, measurement_system, key)

    if debug:
        jax.debug.print("Weights: {weights}", weights=weights)

    # 2. Solve optimal transport problem
    transport_matrix = solve_optimal_transport(ensemble, weights)

    # 3. Apply transformation to generate analysis ensemble
    # Z^a = Z^f @ D^T (matrix form)
    updated_ensemble = ensemble.T @ transport_matrix.T

    return updated_ensemble.T
