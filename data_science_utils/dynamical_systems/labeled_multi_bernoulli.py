import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array, Key, jaxtyped


class LMB(eqx.Module):

    T_birth: int
    L_birth: namedtuple
