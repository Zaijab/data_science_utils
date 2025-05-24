import uuid
from tqdm import tqdm
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from beartype import beartype as typechecker
from data_science_utils.dynamical_systems import Ikeda, Lorenz63, Lorenz96
from data_science_utils.losses import make_step
from data_science_utils.models import InvertibleNN
from data_science_utils.statistics import sample_epanechnikov
from jaxtyping import Array, Float, Key, jaxtyped

print(uuid.uuid4())


debug = False
key = jax.random.key(0)

# system = Ikeda(batch_size=25)
system = Lorenz63()
batch = system.generate(jax.random.key(0), batch_size=10)

key, subkey = jax.random.split(key)

model = InvertibleNN(
    input_dim=system.dimension,
    hidden_dim=64,
    num_coupling_layers=6,
    num_hidden_layers=2,
    key=subkey,
)


optim = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(
        learning_rate=1e-5,
        eps=1e-8,
    ),
)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
batch = eqx.filter_vmap(system.flow)(0.0, 1.0, batch)

from data_science_utils.losses import kl_divergence

print("Doin a thing")
kl_divergence(model, batch)
print("Doin a thing")

# loss, model, opt_state = make_step(model, batch, optim, opt_state)


# for i in range(10):
#     print(i)

#     batch = eqx.filter_vmap(system.flow)(0.0, 1.0, batch)
#     loss, model, opt_state = make_step(model, batch, optim, opt_state)

#     if (i % 500) == 0:
#         print(loss)
#         # plot_learning(model)
