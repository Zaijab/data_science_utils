"""
The purpose of this test is to run an arbitrary Equinox model through a training loop.
"""

import uuid

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from beartype import beartype as typechecker
from data_science_utils.dynamical_systems import Ikeda
from data_science_utils.losses import make_step
from data_science_utils.models import InvertibleNN
from data_science_utils.statistics import sample_epanechnikov
from jaxtyping import Array, Float, Key, jaxtyped

print(uuid.uuid4())


debug = False
key = jax.random.key(0)
key, subkey = jax.random.split(key)

model = InvertibleNN(key=subkey)

system = Ikeda()
batch = system.generate(jax.random.key(0))

optim = optax.chain(
    optax.clip_by_global_norm(10.0),
    optax.adam(
        learning_rate=1e-5,
        b1=0.9,
        b2=0.99999,
        eps=jnp.nextafter(0, 1),
    ),
)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
loss, model, opt_state = make_step(model, batch, optim, opt_state)

x = jax.random.multivariate_normal(subkey, mean=jnp.zeros(2), cov=jnp.eye(2))


def plot_learning(model: InvertibleNN) -> None:
    samples = sample_epanechnikov(
        jax.random.key(0), jnp.zeros(2), jnp.eye(2), batch.shape[0]
    )

    generated_data = eqx.filter_vmap(model)(samples)[0]

    plt.scatter(generated_data[:, 0], generated_data[:, 1], c="red", alpha=0.15)
    plt.xlim(-1, 2)
    plt.ylim(-3, 1.5)
    plt.show()


for i in range(100_000):
    # print(loss)
    # print(
    #     x,
    #     model.inverse(model.forward(x)[0])[0],
    #     jnp.allclose(x, model.inverse(model.forward(x)[0])[0]),
    # )
    batch = system.forward(batch)
    loss, model, opt_state = make_step(model, batch, optim, opt_state)

    if (i % 100) == 0:
        plot_learning(model)
