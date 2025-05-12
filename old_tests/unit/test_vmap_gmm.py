import jax
import jax.numpy as jnp
import equinox as eqx

"""
This module tests sampling capabilities from a number of Gaussian mixture models.
By default, jax.vmap(sampling_func)(list_of_gmms) will not work.
This test file will try to use a workaround and see what happens.
"""


class GMM(eqx.Module):
    means: jnp.array
    covs: jnp.array
    weights: jnp.array

    @jax.jit
    def sample(self, key):
        gaussian_index_key, gaussian_state_key = jax.random.split(key)
        gaussian_index = jax.random.choice(
            key=gaussian_index_key, a=jnp.arange(self.weights.shape[0]), p=self.weights
        )
        my_sample = jax.random.multivariate_normal(
            gaussian_state_key,
            mean=self.means[gaussian_index, :],
            cov=self.covs[gaussian_index, :],
        )
        return my_sample


# TESTING
key = jax.random.key(32)
key, subkey = jax.random.split(key)

gmm_1 = GMM(
    jnp.array([[0, 0], [5, 3], [5, 3], [5, 3]]),
    jnp.array([jnp.eye(2), 2 * jnp.eye(2), 2 * jnp.eye(2), 3 * jnp.eye(2)]),
    jnp.array([0.3, 0.5, 0.1, 0.1]),
)
gmm_2 = GMM(
    jnp.array([[0, 0], [5, 3]]),
    jnp.array([jnp.eye(2), 2 * jnp.eye(2)]),
    jnp.array([0.3, 0.7]),
)

# This is possible to VMAP over.
# Concat all means, covs, weights together
# Use an ownership array and vmap over that
# In vmapped function, use concatenated arrays as static and index from it

subkeys = jax.random.split(key, 2)


import jax
import jax.numpy as jnp
import equinox as eqx


class GMM(eqx.Module):
    means: jnp.ndarray
    covs: jnp.ndarray
    weights: jnp.ndarray

    @eqx.filter_jit
    def sample(self, key):
        gaussian_index_key, gaussian_state_key = jax.random.split(key)
        gaussian_index = jax.random.choice(
            key=gaussian_index_key, a=jnp.arange(self.weights.shape[0]), p=self.weights
        )
        my_sample = jax.random.multivariate_normal(
            gaussian_state_key,
            mean=self.means[gaussian_index],
            cov=self.covs[gaussian_index],
        )
        return my_sample


class LMBBirthModel(eqx.Module):
    birth_probs: jnp.ndarray

    def born(self, key):
        subkeys = jax.random.split(key, self.birth_probs.shape[0])
        return jax.vmap(lambda k, p: jax.random.bernoulli(k, p))(
            subkeys, self.birth_probs
        )


# Testing with JAX arrays and eqx.Module
key = jax.random.key(32)
key, subkey = jax.random.split(key)

# Create GMMs with the same structure for this example
# (in real use, they could have different internal structures)
gmm = GMM(
    jnp.array([[0, 0], [5, 3], [5, 3], [5, 3]]),
    jnp.array([jnp.eye(2), 2 * jnp.eye(2), 2 * jnp.eye(2), 3 * jnp.eye(2)]),
    jnp.array([0.3, 0.5, 0.1, 0.1]),
)

# Create multiple GMMs (in practice, these would have different structures)
gmms = [gmm] * 4

# Create LMB birth model
lmb = LMBBirthModel(jnp.array([0.6, 0.8, 0.6, 0.1]))

# Generate a birth mask
key, subkey = jax.random.split(key)
mask = lmb.born(subkey)

# Get indices where mask is True
born_indices = jnp.where(mask)[0]

# Sample from the born GMMs
key, subkey = jax.random.split(key)


gmm_funcs = [jax.tree_util.Partial(gmm.sample)] * 1_000

index = jnp.arange(len(gmm_funcs))
subkeys = jax.random.split(subkey, len(gmm_funcs))


@jax.jit
@jax.vmap
def sample_parallel(i, key):
    return jax.lax.switch(i, gmm_funcs, key)


sample_parallel(index, subkeys)
