import jax
import jax.numpy as jnp
import equinox as eqx


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


class LMBBirthModel(eqx.Module):
    birth_probs: jnp.array

    def born(self, key):
        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, (self.birth_probs.shape[0],))
        return jax.vmap(lambda key, p: jax.random.bernoulli(key, p))(
            subkeys, lmb.birth_probs
        )


# TESTING
key = jax.random.key(32)
key, subkey = jax.random.split(key)

gmm = GMM(
    jnp.array([[0, 0], [5, 3], [5, 3], [5, 3]]),
    jnp.array([jnp.eye(2), 2 * jnp.eye(2), 2 * jnp.eye(2), 3 * jnp.eye(2)]),
    jnp.array([0.3, 0.5, 0.1, 0.1]),
)


lmb = LMBBirthModel(jnp.array([0.6, 0.8, 0.6, 0.1]))
key, subkey = jax.random.split(key)
mask = lmb.born(subkey)  # [T,F,T,F]
jnp.where(mask)[0]  # [0, 1]
gmms = [gmm] * 4

key, subkey = jax.random.split(key)
subkeys = list(jax.random.split(subkey, len(gmms)))


jax.tree_util.tree_map(lambda gmm, key: gmm.sample(key), gmms, subkeys)
