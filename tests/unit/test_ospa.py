import equinox as eqx
import jax.numpy as jnp
import optax
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from data_science_utils.filters.evaluate import ospa_metric



def test_ospa_metric():
    """Test OSPA metric with symbolically verifiable cases."""

    # Case 1: Both sets empty
    X_empty = jnp.empty((0, 2))
    Y_empty = jnp.empty((0, 2))
    ospa, loc, card = ospa_metric(X_empty, Y_empty)
    assert (
        jnp.allclose(ospa, 0.0) and jnp.allclose(loc, 0.0) and jnp.allclose(card, 0.0)
    )

    # Case 2: X empty, Y non-empty
    Y = jnp.array([[1.0, 2.0]])
    ospa, loc, card = ospa_metric(X_empty, Y, c=100.0)
    assert (
        jnp.allclose(ospa, 100.0)
        and jnp.allclose(loc, 0.0)
        and jnp.allclose(card, 100.0)
    )

    # Case 3: Perfect match
    X = jnp.array([[0.0, 0.0]])
    Y = jnp.array([[0.0, 0.0]])
    ospa, loc, card = ospa_metric(X, Y)
    assert (
        jnp.allclose(ospa, 0.0) and jnp.allclose(loc, 0.0) and jnp.allclose(card, 0.0)
    )

    # Case 4: Single mismatch within cutoff (3-4-5 triangle)
    X = jnp.array([[0.0, 0.0]])
    Y = jnp.array([[3.0, 4.0]])
    ospa, loc, card = ospa_metric(X, Y, c=100.0, p=2.0)
    assert (
        jnp.allclose(ospa, 5.0) and jnp.allclose(loc, 5.0) and jnp.allclose(card, 0.0)
    )

    # Case 5: Cardinality penalty (c=100, p=2)
    X = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    Y = jnp.array([[0.0, 0.0]])
    ospa, loc, card = ospa_metric(X, Y, c=100.0, p=2.0)
    expected_ospa = 100.0 / jnp.sqrt(2.0)  # c * (1/2)^(1/p)
    expected_card = 50.0  # c * |m-n| / max(m,n) = 100 * 1 / 2
    assert (
        jnp.allclose(ospa, expected_ospa)
        and jnp.allclose(loc, 0.0)
        and jnp.allclose(card, expected_card)
    )


def test_ospa_metric_extended():
    """Additional symbolically computed OSPA test cases."""

    # Case 1: Two matched pairs, distance = 2√2 each
    X = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    Y = jnp.array([[2.0, 2.0], [3.0, 3.0]])
    ospa, loc, card = ospa_metric(X, Y, c=100.0, p=2.0)
    expected_ospa = 2.0 * jnp.sqrt(2.0)  # √((8+8)/2) = √8 = 2√2
    expected_loc = 2.0 * jnp.sqrt(2.0)  # √((8+8)/2) = 2√2
    assert (
        jnp.allclose(ospa, expected_ospa)
        and jnp.allclose(loc, expected_loc)
        and jnp.allclose(card, 0.0)
    )

    # Case 2: Cutoff truncation
    X = jnp.array([[0.0, 0.0]])
    Y = jnp.array([[200.0, 0.0]])
    ospa, loc, card = ospa_metric(X, Y, c=100.0, p=2.0)
    assert (
        jnp.allclose(ospa, 100.0)
        and jnp.allclose(loc, 100.0)
        and jnp.allclose(card, 0.0)
    )

    # Case 3: p=1 (Manhattan-style)
    X = jnp.array([[0.0, 0.0]])
    Y = jnp.array([[3.0, 4.0]])
    ospa, loc, card = ospa_metric(X, Y, c=100.0, p=1.0)
    assert (
        jnp.allclose(ospa, 5.0) and jnp.allclose(loc, 5.0) and jnp.allclose(card, 0.0)
    )

    # Case 4: Mixed localization + cardinality
    X = jnp.array([[0.0, 0.0], [10.0, 0.0]])
    Y = jnp.array([[1.0, 0.0]])
    ospa, loc, card = ospa_metric(X, Y, c=100.0, p=2.0)
    expected_ospa = jnp.sqrt(5000.5)  # √((1 + 10000)/2)
    expected_loc = 1.0  # √(1/1)
    expected_card = 50.0  # 100 * 1 / 2
    assert (
        jnp.allclose(ospa, expected_ospa)
        and jnp.allclose(loc, expected_loc)
        and jnp.allclose(card, expected_card)
    )
