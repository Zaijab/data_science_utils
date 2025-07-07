import jax
import time
import pdb

@jax.jit
def f():
    time.sleep(3)
    pdb.set_trace()
    return 0

print("Hoodly")
f()

print("Hoodly")
f()

print("Hoodly")
f()
