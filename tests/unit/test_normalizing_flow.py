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

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Key, jaxtyped
from beartype import beartype as typechecker
from typing import Tuple


DEFAULT_MIN_BIN_WIDTH: float = 1e-3
DEFAULT_MIN_BIN_HEIGHT: float = 1e-3
DEFAULT_MIN_DERIVATIVE: float = 1e-3


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def searchsorted(
    bin_locations: Float[Array, "num_bins"], inputs: Float[Array, "..."]
) -> Array:
    """Find bin indices for inputs in sorted bin_locations."""
    # Add small epsilon to rightmost bin edge for numerical stability
    bin_locations = bin_locations.at[-1].add(1e-6)
    return jnp.sum(inputs[..., None] >= bin_locations, axis=-1) - 1


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def rational_quadratic_spline(
    inputs: Float[Array, "..."],
    unnormalized_widths: Float[Array, "... num_bins"],
    unnormalized_heights: Float[Array, "... num_bins"],
    unnormalized_derivatives: Float[Array, "... num_bins_plus_1"],
    inverse: bool = False,
    left: float = 0.0,
    right: float = 1.0,
    bottom: float = 0.0,
    top: float = 1.0,
    min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
    min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
    min_derivative: float = DEFAULT_MIN_DERIVATIVE,
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Rational quadratic spline transformation on interval [left, right].

    Args:
        inputs: Input values to transform
        unnormalized_widths: Raw bin width parameters
        unnormalized_heights: Raw bin height parameters
        unnormalized_derivatives: Raw derivative parameters at bin edges
        inverse: Whether to compute inverse transformation
        left, right: Input domain bounds
        bottom, top: Output range bounds

    Returns:
        Tuple of (transformed_values, log_abs_det_jacobian)
    """
    # Input validation
    assert jnp.all(inputs >= left) and jnp.all(inputs <= right), "Input outside domain"

    num_bins = unnormalized_widths.shape[-1]

    # Normalize widths to sum to 1, then scale to interval
    widths = jax.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    # Cumulative widths define bin edges
    cumwidths = jnp.cumsum(widths, axis=-1)
    cumwidths = jnp.concatenate(
        [jnp.zeros_like(cumwidths[..., :1]), cumwidths], axis=-1
    )
    cumwidths = (right - left) * cumwidths + left
    cumwidths = cumwidths.at[..., 0].set(left)
    cumwidths = cumwidths.at[..., -1].set(right)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # Derivatives at bin edges (monotonically increasing)
    derivatives = min_derivative + jax.nn.softplus(unnormalized_derivatives)

    # Normalize heights similar to widths
    heights = jax.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights

    cumheights = jnp.cumsum(heights, axis=-1)
    cumheights = jnp.concatenate(
        [jnp.zeros_like(cumheights[..., :1]), cumheights], axis=-1
    )
    cumheights = (top - bottom) * cumheights + bottom
    cumheights = cumheights.at[..., 0].set(bottom)
    cumheights = cumheights.at[..., -1].set(top)
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # Find which bin each input falls into
    if inverse:
        bin_idx = searchsorted(cumheights, inputs)
    else:
        bin_idx = searchsorted(cumwidths, inputs)

    # Gather relevant bin parameters
    input_cumwidths = cumwidths[..., bin_idx]
    input_bin_widths = widths[..., bin_idx]
    input_cumheights = cumheights[..., bin_idx]
    input_heights = heights[..., bin_idx]

    delta = input_heights / input_bin_widths

    input_derivatives = derivatives[..., bin_idx]
    input_derivatives_plus_one = derivatives[..., bin_idx + 1]

    if inverse:
        # Solve quadratic equation for theta
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * delta
        ) + input_heights * (delta - input_derivatives)

        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * delta
        )

        c = -delta * (inputs - input_cumheights)

        discriminant = b**2 - 4 * a * c
        assert jnp.all(discriminant >= 0), "Negative discriminant in spline inverse"

        root = (2 * c) / (-b - jnp.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * delta)
            * theta_one_minus_theta
        )

        derivative_numerator = delta**2 * (
            input_derivatives_plus_one * root**2
            + 2 * delta * theta_one_minus_theta
            + input_derivatives * (1 - root) ** 2
        )

        logabsdet = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)
        return outputs, -logabsdet

    else:
        # Forward transformation
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            delta * theta**2 + input_derivatives * theta_one_minus_theta
        )

        denominator = delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * delta)
            * theta_one_minus_theta
        )

        outputs = input_cumheights + numerator / denominator

        derivative_numerator = delta**2 * (
            input_derivatives_plus_one * theta**2
            + 2 * delta * theta_one_minus_theta
            + input_derivatives * (1 - theta) ** 2
        )

        logabsdet = jnp.log(derivative_numerator) - 2 * jnp.log(denominator)
        return outputs, logabsdet


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def unconstrained_rational_quadratic_spline(
    inputs: Float[Array, "..."],
    unnormalized_widths: Float[Array, "... num_bins"],
    unnormalized_heights: Float[Array, "... num_bins"],
    unnormalized_derivatives: Float[Array, "... num_bins_plus_1"],
    inverse: bool = False,
    tail_bound: float = 1.0,
    **kwargs,
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """
    Unconstrained RQS with linear tails outside [-tail_bound, tail_bound].
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = jnp.where(outside_interval_mask, inputs, 0.0)
    logabsdet = jnp.where(outside_interval_mask, 0.0, 0.0)

    # Add boundary derivatives
    constant = jnp.log(jnp.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1)
    unnormalized_derivatives = jnp.concatenate(
        [
            jnp.full_like(unnormalized_derivatives[..., :1], constant),
            unnormalized_derivatives,
            jnp.full_like(unnormalized_derivatives[..., :1], constant),
        ],
        axis=-1,
    )

    # Apply spline only to inputs inside the interval
    if jnp.any(inside_interval_mask):
        inside_inputs = jnp.where(inside_interval_mask, inputs, 0.0)
        inside_widths = unnormalized_widths
        inside_heights = unnormalized_heights
        inside_derivatives = unnormalized_derivatives

        inside_outputs, inside_logabsdet = rational_quadratic_spline(
            inside_inputs,
            inside_widths,
            inside_heights,
            inside_derivatives,
            inverse=inverse,
            left=-tail_bound,
            right=tail_bound,
            bottom=-tail_bound,
            top=tail_bound,
            **kwargs,
        )

        outputs = jnp.where(inside_interval_mask, inside_outputs, outputs)
        logabsdet = jnp.where(inside_interval_mask, inside_logabsdet, logabsdet)

    return outputs, logabsdet


@jaxtyped(typechecker=typechecker)
class RQSCouplingLayer(eqx.Module):
    """Rational Quadratic Spline Coupling Layer."""

    input_dim: int
    num_bins: int
    swap: bool
    tail_bound: float

    s_net: eqx.nn.MLP
    t_net: eqx.nn.MLP
    w_net: eqx.nn.MLP

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        num_bins: int = 8,
        swap: bool = False,
        tail_bound: float = 3.0,
        *,
        key: Key[Array, "..."],
    ):
        self.input_dim = input_dim
        self.num_bins = num_bins
        self.swap = swap
        self.tail_bound = tail_bound

        split_dim1 = input_dim // 2
        split_dim2 = input_dim - split_dim1

        if swap:
            condition_dim = split_dim2
            output_dim = split_dim1
        else:
            condition_dim = split_dim1
            output_dim = split_dim2

        s_key, t_key, w_key = jax.random.split(key, 3)

        # Networks output parameters for RQS
        self.s_net = eqx.nn.MLP(
            in_size=condition_dim,
            out_size=output_dim * num_bins,  # heights
            width_size=hidden_dim,
            depth=num_hidden_layers,
            activation=jax.nn.tanh,
            key=s_key,
        )

        self.t_net = eqx.nn.MLP(
            in_size=condition_dim,
            out_size=output_dim * num_bins,  # widths
            width_size=hidden_dim,
            depth=num_hidden_layers,
            activation=jax.nn.tanh,
            key=t_key,
        )

        self.w_net = eqx.nn.MLP(
            in_size=condition_dim,
            out_size=output_dim * (num_bins + 1),  # derivatives
            width_size=hidden_dim,
            depth=num_hidden_layers,
            activation=jax.nn.tanh,
            key=w_key,
        )

    def _safe_split(
        self, x: Float[Array, "input_dim"]
    ) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
        """Split input handling odd dimensions."""
        split_point = self.input_dim // 2

        if self.swap:
            split_point = self.input_dim - split_point
            x1 = x[:split_point]
            x2 = x[split_point:]
            return x2, x1
        else:
            x1 = x[:split_point]
            x2 = x[split_point:]
            return x1, x2

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def forward(
        self, x: Float[Array, "input_dim"]
    ) -> Tuple[Float[Array, "input_dim"], Float[Array, ""]]:
        """Forward transformation through RQS coupling layer."""
        x1, x2 = self._safe_split(x)

        # Get spline parameters from conditioning variable x1
        heights = self.s_net(x1).reshape(-1, self.num_bins)
        widths = self.t_net(x1).reshape(-1, self.num_bins)
        derivatives = self.w_net(x1).reshape(-1, self.num_bins + 1)

        # Transform x2 using RQS
        y2, log_det = unconstrained_rational_quadratic_spline(
            x2, widths, heights, derivatives, inverse=False, tail_bound=self.tail_bound
        )

        # Sum log determinants across output dimensions
        log_det_total = jnp.sum(log_det)

        y = jnp.concatenate([x1, y2])
        return y, log_det_total

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def inverse(
        self, y: Float[Array, "input_dim"]
    ) -> Tuple[Float[Array, "input_dim"], Float[Array, ""]]:
        """Inverse transformation through RQS coupling layer."""
        y1, y2 = self._safe_split(y)

        # Get spline parameters from conditioning variable y1
        heights = self.s_net(y1).reshape(-1, self.num_bins)
        widths = self.t_net(y1).reshape(-1, self.num_bins)
        derivatives = self.w_net(y1).reshape(-1, self.num_bins + 1)

        # Inverse transform y2 using RQS
        x2, log_det = unconstrained_rational_quadratic_spline(
            y2, widths, heights, derivatives, inverse=True, tail_bound=self.tail_bound
        )

        # Sum log determinants across output dimensions
        log_det_total = jnp.sum(log_det)

        x = jnp.concatenate([y1, x2])
        return x, log_det_total

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, x: Float[Array, "input_dim"], inverse: bool = False
    ) -> Tuple[Float[Array, "input_dim"], Float[Array, ""]]:
        """Apply forward or inverse transformation."""
        return self.inverse(x) if inverse else self.forward(x)


layer = RQSCouplingLayer(
    input_dim=2,
    hidden_dim=64,
    num_hidden_layers=2,
    num_bins=8,  # Expressivity parameter
    tail_bound=3.0,  # Domain bound
    key=key,
)

# model = InvertibleNN(
#     input_dim=system.dimension,
#     hidden_dim=32,
#     num_coupling_layers=4,
#     num_hidden_layers=1,
#     key=subkey,
# )

model = layer

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
