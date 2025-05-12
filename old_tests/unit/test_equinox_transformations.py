import jax
import equinox as eqx
import time


@eqx.filter_jit
def long_f_1(debug):

    if debug:
        time.sleep(3)

    return 0


long_f_1(True)
