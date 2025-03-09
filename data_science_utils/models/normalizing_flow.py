import jax
import jax.numpy as jnp
from flax import nnx
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from data_science_utils.statistics import logpdf_epanechnikov
from data_science_utils.dynamical_systems import ikeda_forward, ikeda_generate

samples = ikeda_generate(jax.random.key(0), batch_size = 10)


class DenseNetwork(nnx.Module):
    input_dim: int
    hidden_dim: int
    num_hidden_layers: int
    output_dim: int

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, output_dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim

        self.layers = [nnx.Linear(self.input_dim, self.hidden_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64)]
        for _ in range(self.num_hidden_layers):
            self.layers.append(nnx.Linear(self.hidden_dim, self.hidden_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64))
            self.output_layer = nnx.Linear(self.hidden_dim, self.output_dim, kernel_init=nnx.initializers.glorot_uniform(), rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nnx.gelu(x)
        x = self.output_layer(x)
        return x


rngs = nnx.Rngs(0)
dense_test = DenseNetwork(input_dim=2, hidden_dim=10, num_hidden_layers=3, output_dim=10, rngs=rngs)
dense_output = dense_test(samples)
print("Dense Output: ", dense_output.shape)


class CouplingLayer(nnx.Module):
    input_dim: int
    hidden_dim: int
    num_hidden_layers: int
    swap: bool = False

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, swap: bool, rngs: nnx.Rngs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.swap = swap

        self.s_net = DenseNetwork(
            input_dim=self.input_dim // 2,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            output_dim=self.input_dim // 2,
            rngs=rngs
        )
        self.t_net = DenseNetwork(
            input_dim=self.input_dim // 2,
            hidden_dim=self.hidden_dim,
            num_hidden_layers=self.num_hidden_layers,
            output_dim=self.input_dim // 2,
            rngs=rngs
        )

    def __call__(self, x, reverse=False):
        if self.swap:
            x1, x2 = jnp.split(x, 2, axis=-1)
            x1, x2 = x2, x1
        else:
            x1, x2 = jnp.split(x, 2, axis=-1)

        s = self.s_net(x1)
        t = self.t_net(x1)

        if reverse:
            y2 = (x2 - t) * jnp.exp(-s)
            log_det_jacobian = -jnp.sum(s, axis=-1)
        else:
            y2 = x2 * jnp.exp(s) + t
            log_det_jacobian = jnp.sum(s, axis=-1)

        if self.swap:
            y = jnp.concatenate([y2, x1], axis=-1)
        else:
            y = jnp.concatenate([x1, y2], axis=-1)

        return y, log_det_jacobian


coupling_test = CouplingLayer(input_dim=10, hidden_dim=4, num_hidden_layers=2, swap = False, rngs=rngs)
coupling_output, ldj = coupling_test(dense_output)
print("Coupling Output: ", coupling_output.shape)


class InvertibleLinear(nnx.Module):
    """
    Invertible linear transformation parameterized using LU decomposition.
    Provides an invertible linear layer that can be used in normalizing flows.
    """
    def __init__(self, dim, rngs):
        super().__init__()
        key = rngs.params()

        # Initialize a random matrix
        W_init = jax.random.normal(key, (dim, dim), dtype=jnp.float64)

        # LU decomposition with pivoting
        P, L, U = jax.scipy.linalg.lu(W_init)

        # Store P as a float to avoid integer gradient issues
        self.P = nnx.Param(P.astype(jnp.float64))

        # L without diagonal (diagonal is implicitly 1.0)
        self.L_coeffs = nnx.Param(jnp.tril(L, -1))

        # U including diagonal
        self.U = nnx.Param(U)

        # Store dimension for convenience
        self.dim = dim

    def __call__(self, x, reverse=False):
        # Reconstruct L with ones on diagonal
        eye = jnp.eye(self.dim, dtype=jnp.float64)
        L = eye + self.L_coeffs
        U = self.U
        P = self.P
        
        if not reverse:
            # Forward: y = x @ W where W = P @ L @ U
            W = P @ L @ U
            y = x @ W
            
            # Log determinant is sum of log of diagonal of U
            logdet = jnp.sum(jnp.log(jnp.abs(jnp.diag(U))))
            
            return y, logdet
        else:
            # Define a function to solve the system for a single example
            def solve_single(y_single):
                # Step 1: Apply P.T (inverse of P)
                w = P.T @ y_single
                
                # Step 2: Solve L @ z = w (lower triangular system)
                z = jax.scipy.linalg.solve_triangular(
                    L, w, lower=True, unit_diagonal=True
                )
                
                # Step 3: Solve U @ x = z (upper triangular system)
                x_single = jax.scipy.linalg.solve_triangular(
                    U, z, lower=False
                )
                
                return x_single
            
            # Apply to each sample if batched
            if x.ndim > 1:
                x_inv = jax.vmap(solve_single)(x)
            else:
                x_inv = solve_single(x)
            
            # Log determinant for inverse (negative of forward)
            logdet = -jnp.sum(jnp.log(jnp.abs(jnp.diag(U))))
            
            return x_inv, logdet

il = InvertibleLinear(dim=10, rngs=rngs)
il_output, ldj = il(coupling_output)

print(jnp.allclose(il(il_output, reverse=True)[0], coupling_output))

# class InvertibleNN(nnx.Module):
#     input_dim: int = 2
#     num_coupling_layers: int = 5
#     hidden_dim: int = 128
#     num_hidden_layers: int = 4

#     def __init__(self, input_dim=2, num_coupling_layers=2, hidden_dim=32, num_hidden_layers=2, rngs=nnx.Rngs()):
#         super().__init__()
#         self.input_dim = input_dim
#         self.num_coupling_layers = num_coupling_layers
#         self.hidden_dim = hidden_dim
#         self.num_hidden_layers = num_hidden_layers

#         self.coupling_layers = []
#         for i in range(self.num_coupling_layers):
#             swap = i % 2 == 1
#             layer = CouplingLayer(
#                 input_dim=self.input_dim,
#                 hidden_dim=self.hidden_dim,
#                 num_hidden_layers=self.num_hidden_layers,
#                 swap=swap,
#                 rngs=rngs
#             )
#             self.coupling_layers.append(layer)

#     def __call__(self, x, reverse=False):
#         log_det_jacobian = 0
#         if not reverse:
#             for layer in self.coupling_layers:
#                 x, ldj = layer(x)
#                 log_det_jacobian += ldj
#         else:
#             for layer in reversed(self.coupling_layers):
#                 x, ldj = layer(x, reverse=True)
#                 log_det_jacobian += ldj
#         return x, log_det_jacobian



@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        z, log_det_jacobian = model(batch, reverse=True)
        base_log_prob = jax.vmap(logpdf_epanechnikov, in_axes=(0, None, None))(z, jnp.zeros(2), jnp.eye(2))
        total_log_prob = base_log_prob + log_det_jacobian
        total_log_prob = jnp.where(jnp.isfinite(total_log_prob), total_log_prob, -1e6)
        return -jnp.mean(total_log_prob)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

debug = False
key = jax.random.key(0)
key, subkey = jax.random.split(key)
rngs = nnx.Rngs(0)

model = InvertibleNN(input_dim=2,
                     num_coupling_layers=2, hidden_dim=10, num_hidden_layers=2,
                     rngs=rngs)

schedule = optax.cosine_decay_schedule(
    init_value=3*1e-4,
    decay_steps=1e4,
    alpha=0.01,
    exponent=1.0,
)

optimizer = nnx.Optimizer(model, optax.chain(
    optax.clip_by_global_norm(10.0),        
    optax.adam(
        learning_rate = 1e-6,
        b1 = 0.9,
        b2 = 0.99999,
        eps = jnp.nextafter(0,1),
    ),
))


iters = 25*10**4
pbar = tqdm(range(1, iters + 1))
losses = []
batch = ikeda_generate(subkey, batch_size=100)
for i in pbar:
    batch = ikeda_forward(batch)
    key, subkey = jax.random.split(subkey)    
    loss = train_step(model, optimizer, batch)

    pbar.set_postfix({'loss': float(loss)})
    losses.append(loss)

    if i % 10**4 == 0:
        print(loss)
        plt.scatter(batch[:5000, 0], batch[:5000, 1], c='blue', alpha=0.9)
        generated_data = model(sample_standard_epanechnikov(key, num_samples=batch.shape[0]), reverse=False)[0]
        plt.scatter(generated_data[:, 0], generated_data[:,1], c='red', alpha=0.15)
        plt.show()
        plt.plot(losses)
        plt.show()
