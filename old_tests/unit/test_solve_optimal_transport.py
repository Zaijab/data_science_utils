import jax.numpy as jnp
from data_science_utils.filtering import solve_optimal_transport


def test_solve_optimal_transport() -> None:
    """Test the optimal transport solver with a simple case."""

    # Simple 3-point ensemble
    ensemble = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    # Case 1: Uniform weights (should give approximately identity matrix)
    weights = jnp.array([1 / 3, 1 / 3, 1 / 3])
    transport_matrix = solve_optimal_transport(ensemble, weights)

    # Check row and column sums
    assert jnp.allclose(jnp.sum(transport_matrix, axis=1), jnp.ones(3) / 3, atol=1e-5)
    assert jnp.allclose(jnp.sum(transport_matrix, axis=0), weights, atol=1e-5)
    print("Hai")


test_solve_optimal_transport_scenario_2()
