import jax.numpy as jnp


def is_positive_definite(M):
    try:
        jnp.linalg.cholesky(M)
        return True
    except e:
        assert False, "COVARIANCE MATRIX IS NOT POSITIVE DEFINITE"


def has_nan(M):
    assert not jnp.any(jnp.isnan(M))
