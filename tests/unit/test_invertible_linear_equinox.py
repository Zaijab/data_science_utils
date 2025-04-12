import jax
import jax.numpy as jnp
import equinox as eqx


class InvertibleLinear(eqx.Module):
    dimension: int = 10
    initial_matrix: object
    P: object
    L: object
    U: object

    def __init__(self, key):
        self.initial_matrix = jax.random.normal(
            key, shape=(self.dimension, self.dimension)
        )
        self.P, self.L, self.U = jax.scipy.linalg.lu(self.initial_matrix)


key = jax.random.key(0)
my_linear = InvertibleLinear(key)

with jnp.printoptions(precision=3):
    print(my_linear.U)
