import equinox as eqx
import jax
import jax.numpy as jnp
from data_science_utils.dynamical_systems import Ikeda
from data_science_utils.measurement_functions import RangeSensor


@eqx.filter_jit
def enkf_update(
    key,
    ensemble,
    measurement,
    measurement_system,
    inflation_factor,
    debug=False,
):
    mean = jnp.mean(ensemble, axis=0)

    if debug:
        jax.debug.print("{shape}", mean.shape)

    inflated = mean + inflation_factor * (ensemble - mean)

    ensemble_covariance = jnp.cov(inflated.T)  # + 1e-8 * jnp.eye(inflated.shape[-1])

    @jax.jit
    def update_ensemble_point(point, key):
        point_measurement = measurement_system(point, key)
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
    updated_ensemble = vmap(update_ensemble_point)(inflated, keys)
    return updated_ensemble


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

    (final_carry, errors_over_time) = lax.scan(
        scan_step, (initial_ensemble, initial_true_state), keys
    )
    errors_past_burn_in = errors_over_time[burn_in_time:]
    rmse = jnp.sqrt(jnp.mean(errors_past_burn_in**2))

    return rmse


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

    # Calculate likelihood for each ensemble member
    likelihoods = jax.vmap(compute_likelihood)(ensemble, keys)

    # Normalize weights
    weights = likelihoods / jnp.sum(likelihoods)
    return weights


key = jax.random.key(100)
key, subkey = jax.random.split(key)
measurement_system = RangeSensor(jnp.array([[0.25]]))
dynamical_system = Ikeda(u=0.9)
true_state = dynamical_system.initial_state
ensemble_size = 10
ensemble = jax.random.multivariate_normal(
    subkey,
    shape=(ensemble_size,),
    mean=true_state,
    cov=(1 / 4) * jnp.eye(2),
)

weights = compute_weights(
    ensemble, measurement_system(true_state, subkey), measurement_system, subkey
)


@eqx.filter_jit
def solve_optimal_transport(ensemble, weights, epsilon=0.0001):
    ensemble_size = ensemble.shape[0]

    # Source distribution: uniform weights
    a = jnp.ones(ensemble_size) / ensemble_size

    # Target distribution: importance weights
    b = weights

    # Create geometry for the pointcloud - no need to specify cost_fn
    from ott.geometry import pointcloud

    geom = pointcloud.PointCloud(ensemble, ensemble, epsilon=epsilon)

    # Create the linear OT problem
    from ott.problems.linear import linear_problem

    ot_problem = linear_problem.LinearProblem(geom, a=a, b=b)

    # Solve using Sinkhorn
    from ott.solvers.linear import sinkhorn

    solver = sinkhorn.Sinkhorn(inner_iterations=200)
    ot_result = solver(ot_problem)

    # Get the transport matrix
    transport_matrix = ot_result.matrix

    return transport_matrix


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


evaluate_filter(
    ensemble,
    dynamical_system,
    measurement_system,
    etpf_update,
    key,
)
