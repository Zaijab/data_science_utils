import jax
import jax.numpy as jnp


def test_log_sum_exp_equivalence() -> None:
    key = jax.random.key(0)
    batch_size = 10_000
    weights = jax.random.normal(key, shape=(batch_size,))

    def exp(logposterior_weights):
        """
        Computes: w -> exp(w) / sum(exp(w)) in a numerically stable way
        """
        m = jnp.max(logposterior_weights)
        g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
        posterior_weights = jnp.exp(logposterior_weights - g)
        posterior_weights = posterior_weights / jnp.sum(posterior_weights)
        return posterior_weights

    lse_original = exp(weights)
    lse_unstable = jnp.exp(weights) / jnp.sum(jnp.exp(weights))
    lse_scipy = jnp.exp(weights - jax.scipy.special.logsumexp(weights))

    print(lse_original - lse_scipy)
    assert jnp.allclose(lse_original, lse_scipy)


def test_log_sum_exp_equivalence() -> None:
    key = jax.random.key(0)
    batch_size = 10_000
    measurements = 42
    weights = jax.random.normal(key, shape=(measurements, batch_size))

    def exp(logposterior_weights):
        """
        Computes: w -> exp(w) / sum(exp(w)) in a numerically stable way
        """
        m = jnp.max(logposterior_weights, axis=1)
        g = m + jnp.log(jnp.sum(jnp.exp(logposterior_weights - m)))
        posterior_weights = jnp.exp(logposterior_weights - g)
        posterior_weights = posterior_weights / jnp.sum(posterior_weights)
        return posterior_weights

    lse_original = exp(weights)
    lse_unstable = jnp.exp(weights, axis=1) / jnp.sum(jnp.exp(weights, axis=1), axis=1)
    lse_scipy = jnp.exp(weights - jax.scipy.special.logsumexp(weights, axis=1))

    print(lse_original - lse_scipy)
    assert jnp.allclose(lse_original, lse_scipy)
