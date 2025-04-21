import jax
import jax.numpy as jnp
import jax.scipy as jsp

key = jax.random.key(0)
dimension = 10
initial_matrix = jax.random.normal(key, shape=(dimension, dimension))
P, L, U = jsp.linalg.lu(initial_matrix)

with jnp.printoptions(precision=3):
    print(L)
    print(jnp.tril(initial_matrix, -1))
