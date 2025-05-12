import jax
import equinox as eqx


@eqx.filter_jit
def f():
    print("Hello World!")
    jax.debug.print("Hello World! {}", jax.numpy.array([2.0]))
    return 0


f()
f()
