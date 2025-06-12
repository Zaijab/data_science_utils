import os
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from data_science_utils.statistics import GMM
from data_science_utils.statistics import GMM as GMM


def create_sample_gmms() -> List[GMM]:
    """Create sample GMMs for testing."""
    # GMM 1: 2D, 2 components
    means1 = jnp.array([[0.0, 0.0], [3.0, 3.0]])
    covs1 = jnp.array([[[1.0, 0.2], [0.2, 1.0]], [[0.5, -0.1], [-0.1, 0.5]]])
    weights1 = jnp.array([0.6, 0.4])

    # GMM 2: 2D, 3 components
    means2 = jnp.array([[-2.0, 1.0], [1.0, -2.0], [4.0, 0.0]])
    covs2 = jnp.array(
        [[[0.8, 0.0], [0.0, 0.8]], [[1.2, 0.3], [0.3, 1.2]], [[0.3, 0.0], [0.0, 1.5]]]
    )
    weights2 = jnp.array([0.3, 0.5, 0.2])

    # GMM 3: 2D, 1 component
    means3 = jnp.array([[2.0, 2.0]])
    covs3 = jnp.array([[[2.0, 0.5], [0.5, 1.0]]])
    weights3 = jnp.array([1.0])

    return [
        GMM(means1, covs1, weights1),
        GMM(means2, covs2, weights2, max_components=125),
        GMM(means3, covs3, weights3),
    ]


def test_gmm_initialization():
    """Test GMM initialization."""
    gmms = create_sample_gmms()
    gmm = gmms[0]

    assert gmm.means.shape == (2, 2)
    assert gmm.covs.shape == (2, 2, 2)
    assert gmm.weights.shape == (2,)
    assert jnp.allclose(jnp.sum(gmm.weights), 1.0, atol=1e-6)


def test_gmm_sampling():
    """Test GMM sampling produces correct shapes."""
    key = jax.random.key(42)
    gmms = create_sample_gmms()
    gmm = gmms[0]

    sample = gmm.sample(key)
    assert sample.shape == (2,)
    assert jnp.all(jnp.isfinite(sample))


def test_gmm_sampling():
    """Test GMM sampling produces correct shapes."""
    key = jax.random.key(42)
    gmms = create_sample_gmms()
    gmm = gmms[1]

    print(gmm)
    print(gmm.weights)

    sample = gmm.sample(key)
    assert sample.shape == (2,)
    assert jnp.all(jnp.isfinite(sample))


def test_gmm_jaxtyping() -> None:
    key = jax.random.key(42)
    gmms = create_sample_gmms()
    gmm = gmms[1]

    @jaxtyped(typechecker=typechecker)
    def interior_function(gmm: GMM) -> GMM:
        return gmm

    print(gmm)
    interior_function(gmm)
