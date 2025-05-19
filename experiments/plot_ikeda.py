import jax
import matplotlib.pyplot as plt

from data_science_utils.dynamical_systems import Ikeda

u = 0.9
batch_size = 100
system = Ikeda(u=u, batch_size=batch_size)
attractor = system.generate(jax.random.key(0))
plt.scatter(attractor[:, 0], attractor[:, 1])
