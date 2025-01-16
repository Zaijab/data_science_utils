from data_science_utils.dynamical_systems.ikeda import IkedaSystem
import jax
import jax.numpy as jnp

key = jax.random.key(0)

key, subkey = jax.random.split(key)
ikeda = IkedaSystem(state=jnp.array([[1.0, 2.0]]), u=0.9)
with jax.checking_leaks():
    ikeda.flow(ikeda.state)
    batch = ikeda.generate(subkey)
    batch = ikeda.flow(batch)
ikeda.iterate()
print(ikeda.state.shape)
