"""
In this file we provide multiple classes related to Gaussian Mixture Models.

Namely, if the user needs a simple GMM, they may use the GMM class wherein you specify the means, covariances, and weights.
It comes with a sample method once the user specifies all the necessary weights.

If a user wishes to sample from multiple GMMs, then there are two approaches.
Either they form a list of the GMMs and we can VMAP over the index.

gmm_funcs = [jax.tree_util.Partial(gmm.sample)] * 50 + [
    jax.tree_util.Partial(gmm_1.sample)
] * 50


index = jnp.arange(len(gmm_funcs))
subkeys = jax.random.split(subkey, len(gmm_funcs))


@eqx.filter_jit
@eqx.filter_vmap
def sample_parallel(i, key):
    return jax.lax.switch(i, gmm_funcs, key)


sample_parallel(index, subkeys).shape

The method we choose is to take the largest number of components (suppose gmm_1 has 3 components and gmm_2 has 5)
Then zero pad everything to be of the same size. The padding is done for you after initialization.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped, Int, Bool
from typing import List
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key, jaxtyped
from beartype import beartype as typechecker
from typing import List


@jaxtyped(typechecker=typechecker)
class GMM(eqx.Module):
    means: Float[Array, "num_components state_dim"]
    covs: Float[Array, "num_components state_dim state_dim"]
    weights: Float[Array, "num_components"]

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def sample(self, key: Key[Array, ""]) -> Float[Array, "state_dim"]:
        gaussian_index_key, gaussian_state_key = jax.random.split(key)
        gaussian_index = jax.random.choice(
            key=gaussian_index_key, a=jnp.arange(self.weights.shape[0]), p=self.weights
        )
        sample = jax.random.multivariate_normal(
            gaussian_state_key,
            mean=self.means[gaussian_index],
            cov=self.covs[gaussian_index],
        )
        return sample


@jaxtyped(typechecker=typechecker)
class MultiGMM(eqx.Module):
    """Vectorized multiple GMMs with zero-padding for uniform shapes."""

    means: Float[Array, "num_gmm max_components state_dim"]
    covs: Float[Array, "num_gmm max_components state_dim state_dim"]
    weights: Float[Array, "num_gmm max_components"]

    def __init__(self, gmms: List[GMM]):
        assert len(gmms) > 0, "Need at least one GMM"

        state_dim = gmms[0].means.shape[1]
        max_components = max(gmm.means.shape[0] for gmm in gmms)

        # Pad each GMM to max_components and collect in single pass
        def pad_gmm(gmm: GMM) -> GMM:
            pad_width = max_components - gmm.means.shape[0]

            padded_means = jnp.pad(gmm.means, ((0, pad_width), (0, 0)), mode="constant")
            padded_weights = jnp.pad(gmm.weights, (0, pad_width), mode="constant")

            identity_padding = jnp.tile(jnp.eye(state_dim), (pad_width, 1, 1))
            padded_covs = jnp.concatenate([gmm.covs, identity_padding], axis=0)

            return GMM(padded_means, padded_covs, padded_weights)

        # Pad all GMMs and use tree operations to stack
        padded_gmms = [pad_gmm(gmm) for gmm in gmms]
        stacked = jax.tree_map(lambda *arrays: jnp.stack(arrays), *padded_gmms)

        # Verify shapes are correct for JIT compilation
        assert stacked.means.shape == (len(gmms), max_components, state_dim)
        assert stacked.covs.shape == (len(gmms), max_components, state_dim, state_dim)
        assert stacked.weights.shape == (len(gmms), max_components)

        self.means = stacked.means
        self.covs = stacked.covs
        self.weights = stacked.weights

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def sample(
        self, key: Key[Array, ""], gmm_idx: int | Int[Array, "..."]
    ) -> Float[Array, "state_dim"]:
        """Sample from the gmm_idx-th GMM."""
        gaussian_index_key, gaussian_state_key = jax.random.split(key)

        # Weights are already zero-padded, just renormalize
        weights = self.weights[gmm_idx]
        # weights = weights / jnp.sum(weights)

        gaussian_index = jax.random.choice(
            key=gaussian_index_key, a=jnp.arange(self.weights.shape[1]), p=weights
        )

        sample = jax.random.multivariate_normal(
            gaussian_state_key,
            mean=self.means[gmm_idx, gaussian_index],
            cov=self.covs[gmm_idx, gaussian_index],
        )
        return sample

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def sample_all(self, key: Key[Array, ""]) -> Float[Array, "num_gmm state_dim"]:
        """Sample one point from each GMM."""
        keys = jax.random.split(key, self.means.shape[0])
        samples = eqx.filter_vmap(lambda k, i: self.sample(k, i))(
            keys, jnp.arange(self.means.shape[0])
        )
        return samples


@jaxtyped(typechecker=typechecker)
class LMBBirthModel(eqx.Module):
    """Labeled Multi-Bernoulli birth model from Section II.B of the paper."""

    birth_probs: Float[Array, "num_birth_locations"]  # r_B(ℓ)
    birth_gmms: MultiGMM  # p_B(·, ℓ) for each birth location

    def __init__(
        self, birth_probs: Float[Array, "num_birth_locations"], gmms: List[GMM]
    ):
        assert birth_probs.shape[0] == len(gmms), "Mismatch in birth locations"
        assert jnp.all(birth_probs >= 0) and jnp.all(
            birth_probs <= 1
        ), "Invalid probabilities"

        self.birth_probs = birth_probs
        self.birth_gmms = MultiGMM(gmms)

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def sample_birth_indicators(
        self, key: Key[Array, ""]
    ) -> Float[Array, "num_birth_locations"]:
        """Sample Bernoulli indicators for each birth location."""
        keys = jax.random.split(key, self.birth_probs.shape[0])
        indicators = eqx.filter_vmap(
            lambda k, p: jax.random.bernoulli(k, p).astype(jnp.float32)
        )(keys, self.birth_probs)
        return indicators

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def sample_birth_states(
        self, key: Key[Array, ""]
    ) -> Float[Array, "num_birth_locations state_dim"]:
        """Sample birth states from each GMM (regardless of birth indicators)."""
        return self.birth_gmms.sample_all(key)

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def sample_births(
        self, key: Key[Array, ""]
    ) -> tuple[Float[Array, "num_births state_dim"], Float[Array, "num_births"]]:
        """
        Complete birth sampling: indicators + states for born objects.

        Returns:
            born_states: States of actually born objects
            born_labels: Labels (indices) of born objects
        """
        indicator_key, state_key = jax.random.split(key)

        # Sample birth indicators and states
        indicators = self.sample_birth_indicators(indicator_key)
        all_states = self.sample_birth_states(state_key)

        # Extract only born objects
        born_mask = indicators.astype(bool)
        born_states = all_states[born_mask]
        born_labels = jnp.where(born_mask)[0].astype(jnp.float32)

        return born_states, born_labels


# Test implementation
if __name__ == "__main__":
    key = jax.random.key(42)

    # Create test GMMs with different numbers of components
    gmm_1 = GMM(
        jnp.array([[0.0, 0.0], [5.0, 3.0], [2.0, 1.0]]),
        jnp.array([jnp.eye(2), 2 * jnp.eye(2), 0.5 * jnp.eye(2)]),
        jnp.array([0.3, 0.5, 0.2]),
    )

    gmm_2 = GMM(
        jnp.array([[10.0, 10.0], [15.0, 13.0]]),
        jnp.array([3 * jnp.eye(2), jnp.eye(2)]),
        jnp.array([0.6, 0.4]),
    )

    gmm_3 = GMM(
        jnp.array([[20.0, 20.0]]), jnp.array([4 * jnp.eye(2)]), jnp.array([1.0])
    )

    gmms = MultiGMM([gmm_1, gmm_2, gmm_3])
    print(gmms)
    key, subkey = jax.random.split(key)
    print(gmms.sample(key, jnp.asarray(0)))
    print(gmms.sample_all(key))

    # Create LMB birth model
    # birth_probs = jnp.array([0.1, 0.05, 0.08])
    # lmb_birth = LMBBirthModel(birth_probs, [gmm_1, gmm_2, gmm_3])

    # Test sampling
    # key, subkey = jax.random.split(key)
    # born_states, born_labels = lmb_birth.sample_births(subkey)

    # print(f"Born states shape: {born_states.shape}")
    # print(f"Born labels: {born_labels}")
    # print(f"Number of births: {len(born_labels)}")
