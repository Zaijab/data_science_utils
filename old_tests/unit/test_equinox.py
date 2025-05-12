import jax
from functools import partial
import equinox as eqx

from jaxtyping import Float, Array
import datetime


class test_equinox(eqx.Module):
    n_things: int = 4
    array_or_smthin: Float[Array, "2"] = jax.numpy.array([1, 2])

    # @partial(jax.jit, static_argnames=["self"]) <- Breaks with array as argument
    @jax.jit
    def do_a_thing(self, x):
        return x


test_equinox_instance = test_equinox()
print(datetime.datetime.now())
print(test_equinox_instance.n_things)
print(test_equinox_instance.do_a_thing(5))


def func(
    thing1,
    thing2,
    thing3,
    thing4,
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaah,
    bruuuaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
):
    pass
