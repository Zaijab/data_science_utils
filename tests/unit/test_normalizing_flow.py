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
batch = system.generate(jax.random.key(0), batch_size=500)

key, subkey = jax.random.split(key)

model = InvertibleNN(
    input_dim=system.dimension,
    hidden_dim=32,
    num_coupling_layers=4,
    num_hidden_layers=1,
    key=subkey,
)


optim = optax.chain(
    optax.clip_by_global_norm(10.0),
    optax.adam(
        learning_rate=1e-4,
        eps=1e-8,
    ),
)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
batch = eqx.filter_vmap(system.flow)(0.0, 1.0, batch)

key, subkey = jax.random.split(key)
normal_key, state_key = jax.random.split(subkey)
x = system.initial_state(state_key)


def plot_learning(model, system, key, num_samples=1000, savepath=None):
    import matplotlib.pyplot as plt

    # Sample from system
    key, subkey1, subkey2 = jax.random.split(key, 3)
    system_samples = system.generate(subkey1, batch_size=num_samples)

    # Sample from model (push forward standard normal through flow)
    dim = system.dimension
    z_samples = jax.random.normal(subkey2, shape=(num_samples, dim))
    model_samples = jax.vmap(model.forward)(z_samples)[0]

    # Plot (assume 3D or 2D system)
    fig = plt.figure(figsize=(10, 5))
    if dim == 3:
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        ax1.scatter(*system_samples.T, s=1, alpha=0.5, label="True")
        ax1.set_xlim(left=-20, right=20)
        ax1.set_ylim(bottom=-25, top=25)
        ax1.set_zlim(bottom=0, top=45)
        ax2.scatter(*model_samples.T, s=1, alpha=0.5, label="Flow")
        ax2.set_xlim(left=-20, right=20)
        ax2.set_ylim(bottom=-25, top=25)
        ax2.set_zlim(bottom=0, top=45)
    else:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.scatter(
            system_samples[:, 0], system_samples[:, 1], s=1, alpha=0.5, label="True"
        )
        ax2.scatter(
            model_samples[:, 0], model_samples[:, 1], s=1, alpha=0.5, label="Flow"
        )

    ax1.set_title("True System")
    ax2.set_title("Learned Flow")
    for ax in [ax1, ax2]:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if dim == 3:
            ax.set_zlabel("z")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    else:
        plt.show()


pbar = tqdm(range(10_000_000))
for i in pbar:

    batch = eqx.filter_vmap(system.flow)(0.0, 1.0, batch)
    loss, model, opt_state = make_step(model, batch, optim, opt_state)

    pbar.set_description(f"Loss {loss}")

    if (i % 500) == 0:
        print(i, loss, x, model.forward(model.inverse(x)[0])[0])
        plot_learning(model, system, key)
