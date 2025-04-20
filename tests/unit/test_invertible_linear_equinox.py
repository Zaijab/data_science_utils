import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import jaxtyped, Float, Key, Array


class InvertibleLinear(eqx.Module):
    initial_matrix: Float[Array, "..."]
    P: Float[Array, "..."]
    L: Float[Array, "..."]
    U: Float[Array, "..."]
    dimension: int = 10

    def __init__(self, key: Key[Array, "..."]):
        self.initial_matrix = jax.random.normal(
            key, shape=(self.dimension, self.dimension)
        )
        self.P, self.L, self.U = jax.scipy.linalg.lu(self.initial_matrix)


key: Key[Array, "..."] = jax.random.key(0)
my_linear = InvertibleLinear(key)

with jnp.printoptions(precision=3):
    print(my_linear.U)
