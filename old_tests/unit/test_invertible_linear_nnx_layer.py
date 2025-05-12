"""
This module tests the invertible linear layer for use in normalizing flows.
We test the numerical stability of invertibility and how the layer maintains invertibility throughout training.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class InvertibleLinear(nnx.Module):
    def __init__(self, dim, rngs):
        super().__init__()
        key = rngs.params()
        # Initialize a random matrix
        W_init = jax.random.normal(key, (dim, dim))
        # LU decomposition with pivoting
        P, L, U = jax.scipy.linalg.lu(W_init)
        # Store P as a float to avoid integer gradient issues
        self.P = nnx.Param(P)
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
            # For the inverse, we want to solve for x in y = x @ W
            # We have x = y @ W^(-1) = y @ U^(-1) @ L^(-1) @ P.T

            # We'll use the transposition trick to compute this:
            # (y @ U^(-1) @ L^(-1) @ P.T).T = P @ (L^(-1)).T @ (U^(-1)).T @ y.T

            # Step 1: Compute (U^(-1)).T @ y.T (solve U.T @ z.T = y.T for z.T)
            y_T = jnp.atleast_2d(x).T  # Convert input to column vector
            U_T = jnp.transpose(U)
            z_T = jax.scipy.linalg.solve_triangular(U_T, y_T, lower=True)

            # Step 2: Compute (L^(-1)).T @ z.T (solve L.T @ w.T = z.T for w.T)
            L_T = jnp.transpose(eye + self.L_coeffs)
            w_T = jax.scipy.linalg.solve_triangular(
                L_T, z_T, lower=False, unit_diagonal=True
            )

            # Step 3: Compute P @ w.T
            x_T = P @ w_T

            # Step 4: Convert back to row vector
            x_inv = jnp.squeeze(x_T.T)

            # Log determinant for inverse (negative of forward)
            logdet = -jnp.sum(jnp.log(jnp.abs(jnp.diag(U))))
            return x_inv, logdet


# Main optimization loop with stability checks
def run_optimization_test(num_iter=1000, dim=10, batch_size=128, lr=1e-3, seed=0):
    # Set up RNGs
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    rngs = nnx.Rngs(params=subkey)

    # Initialize model
    model = SimpleFlow(dim=dim, hidden_dim=64, rngs=rngs)

    # Set up optimizer
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(nnx.extract(model).params)

    # Metrics to track
    losses = []
    invert_errors = []
    condition_numbers = []

    # JIT-compile the update function for efficiency
    @jax.jit
    def update_step(model, opt_state, batch):
        def loss_fn(params):
            model_with_params = model.replace(params=params)
            loss = jnp.mean(jax.vmap(model_with_params.nll)(batch))
            return loss

        params = nnx.extract(model).params
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_model = model.replace(params=new_params)
        return new_model, new_opt_state, loss

    # Train the model
    for i in range(num_iter):
        # Generate random batch
        key, subkey = jax.random.split(key)
        batch = jax.random.normal(subkey, (batch_size, dim))

        # Update model
        model, opt_state, loss = update_step(model, opt_state, batch)
        losses.append(loss.item())

        # Check invertibility (on a smaller batch for efficiency)
        if i % 10 == 0:
            key, subkey = jax.random.split(key)
            test_batch = jax.random.normal(subkey, (10, dim))
            invert_error = jnp.mean(
                jax.vmap(model.check_invertibility)(test_batch)
            ).item()
            invert_errors.append(invert_error)

            # Check condition number of the weight matrix (numerical stability)
            # Reconstruct W from P, L, U
            eye = jnp.eye(dim, dtype=jnp.float64)
            L = eye + model.invertible_linear.L_coeffs
            U = model.invertible_linear.U
            P = model.invertible_linear.P
            W = P @ L @ U

            # Compute condition number
            sing_vals = jnp.linalg.svd(W, compute_uv=False)
            cond_num = jnp.max(sing_vals) / jnp.min(sing_vals)
            condition_numbers.append(cond_num.item())

            print(
                f"Iteration {i}: Loss = {loss:.4f}, Invert Error = {invert_error:.8f}, Condition Number = {cond_num:.4f}"
            )

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Iteration")
    plt.ylabel("NLL")

    plt.subplot(1, 3, 2)
    plt.plot(range(0, num_iter, 10), invert_errors)
    plt.title("Invertibility Error")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.yscale("log")

    plt.subplot(1, 3, 3)
    plt.plot(range(0, num_iter, 10), condition_numbers)
    plt.title("Condition Number")
    plt.xlabel("Iteration")
    plt.ylabel("Cond(W)")
    plt.yscale("log")

    plt.tight_layout()
    plt.show()

    return model, losses, invert_errors, condition_numbers


# Run the test (comment this out if you want to import this code elsewhere)
if __name__ == "__main__":
    model, losses, invert_errors, condition_numbers = run_optimization_test(
        num_iter=1000, dim=10
    )

    # Print final statistics
    print(f"Final invertibility error: {invert_errors[-1]:.8f}")
    print(f"Final condition number: {condition_numbers[-1]:.4f}")

    # Test invertibility on a larger batch
    key = jax.random.key(1)
    test_batch = jax.random.normal(key, (1000, 10))
    invert_error = jnp.mean(jax.vmap(model.check_invertibility)(test_batch)).item()
    print(f"Invertibility error on large test batch: {invert_error:.8f}")


key = jax.random.key(0)
rngs = nnx.Rngs(0)
il = InvertibleLinear(dim=10, rngs=rngs)
test_input = jax.random.normal(key=key, shape=(10,))
il_output, ldj = il(test_input)

print(il_output.shape, "Hai")
print(jnp.allclose(il(il_output, reverse=True)[0], test_input))
