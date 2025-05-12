import jax
import jax.numpy as jnp
from flax import nnx
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_science_utils.statistics import logpdf_epanechnikov, sample_epanechnikov
from data_science_utils.dynamical_systems import ikeda_generate, ikeda_forward

class TrainableInvertibleLinear(nnx.Module):
    """
    Trainable Invertible Linear Layer with Guaranteed Invertibility
    """
    def __init__(self, dim: int, rngs: nnx.Rngs, eps: float = 1e-8):
        super().__init__()
        
        # Householder Reflections parameters
        self.v_vectors = nnx.Param(
            jax.random.normal(rngs.params(), (dim, dim), dtype=jnp.float64)
        )
        
        # Scaling parameters
        self.scale = nnx.Param(
            jnp.ones(1, dtype=jnp.float64)
        )
        
        self.diag_scale = nnx.Param(
            jnp.ones(dim, dtype=jnp.float64)
        )
        
        self.dim = dim
        self.eps = eps

    def _householder_reflection(self, v):
        """
        Compute Householder reflection matrix
        H = I - 2 * (v @ v.T) / (v.T @ v)
        """
        v = v / (jnp.linalg.norm(v) + self.eps)
        H = jnp.eye(self.dim) - 2 * jnp.outer(v, v)
        return H

    def __call__(self, x, reverse=False):
        # Compute orthogonal matrix using Householder reflections
        Q = jnp.eye(self.dim)
        for v in self.v_vectors.value.T:
            Q = Q @ self._householder_reflection(v)
        
        # Diagonal scaling with learned parameters
        D = jnp.diag(self.diag_scale.value)
        scale = self.scale.value[0]
        
        if not reverse:
            # Forward transformation
            y = x @ (Q @ D * scale)
            
            # Log determinant
            logdet = jnp.sum(jnp.log(jnp.abs(self.diag_scale.value * scale) + self.eps))
            
            return y, logdet
        else:
            # Inverse transformation
            def solve_single(y_single):
                # Inverse solve with transposed Q and inverse scaling
                z = y_single @ jnp.linalg.inv(Q @ D * scale)
                return z
            
            # Apply to each sample if batched
            if x.ndim > 1:
                x_inv = jax.vmap(solve_single)(x)
            else:
                x_inv = solve_single(x)
            
            # Log determinant for inverse
            logdet = -jnp.sum(jnp.log(jnp.abs(self.diag_scale.value * scale) + self.eps))
            
            return x_inv, logdet

    def verify_properties(self):
        """
        Verify invertibility and orthogonality properties
        """
        # Compute orthogonal matrix using Householder reflections
        Q = jnp.eye(self.dim)
        for v in self.v_vectors.value.T:
            Q = Q @ self._householder_reflection(v)
        
        # Diagonal scaling
        D = jnp.diag(self.diag_scale.value)
        scale = self.scale.value[0]
        W = Q @ D * scale
        
        # Compute inverse
        W_inv = jnp.linalg.inv(W)
        
        # Prepare results dictionary
        results = {
            'orthogonality_check': jnp.allclose(Q.T @ Q, jnp.eye(self.dim), atol=1e-6),
            'inverse_consistency': jnp.allclose(W @ W_inv, jnp.eye(self.dim), atol=1e-6),
            'condition_number': jnp.linalg.cond(W),
            'determinant': jnp.linalg.det(W)
        }
        
        return results

# Training Step
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

# Main Training Loop
def run_training():
    # Set random seeds for reproducibility
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    
    # Initialize model and optimizer
    rngs = nnx.Rngs(0)
    model = TrainableInvertibleLinear(dim=2, rngs=rngs)
    
    # Learning rate schedule
    schedule = optax.cosine_decay_schedule(
        init_value=3*1e-4,
        decay_steps=1e4,
        alpha=0.01,
        exponent=1.0,
    )
    
    # Optimizer with gradient clipping and Adam
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(10.0),        
        optax.adam(
            learning_rate=1e-6,
            b1=0.9,
            b2=0.99999,
            eps=jnp.nextafter(0,1),
        ),
    ))
    
    # Training parameters
    iters = 25*10**4
    pbar = tqdm(range(1, iters + 1))
    losses = []
    
    # Property tracking
    property_logs = {
        'orthogonality_check': [],
        'inverse_consistency': [],
        'condition_number': [],
        'determinant': []
    }
    
    # Initial batch generation
    batch = ikeda_generate(subkey, batch_size=100)
    
    # Training loop
    for i in pbar:
        # Ikeda dynamical system transformation of batch
        batch = ikeda_forward(batch)
        
        # Generate new random key
        key, subkey = jax.random.split(subkey)    
        
        # Perform training step
        loss = train_step(model, optimizer, batch)
        
        # Update progress bar
        pbar.set_postfix({'loss': float(loss)})
        losses.append(loss)
        
        # Verify properties every 100 steps
        if i % 100 == 0:
            properties = model.verify_properties()
            
            # Log properties
            for prop_name, prop_value in properties.items():
                # Convert boolean to float for plotting
                if isinstance(prop_value, bool):
                    prop_value = float(prop_value)
                property_logs[prop_name].append(prop_value)
            
            # Print properties periodically
            if i % 1000 == 0:
                print(f"\nIteration {i} Properties:")
                for prop_name, prop_values in property_logs.items():
                    print(f"{prop_name}: {prop_values[-1]}")
        
        # Visualization and logging every 10,000 iterations
        if i % 10**4 == 0:
            print(f"Iteration {i}, Loss: {loss}")
            
            # Scatter plot of original and generated data
            plt.figure(figsize=(15,5))
            
            # Original vs Generated Data
            plt.subplot(1,3,1)
            plt.title('Original vs Generated Data')
            plt.scatter(batch[:5000, 0], batch[:5000, 1], c='blue', alpha=0.9, label='Original')
            
            # Generate samples from Epanechnikov distribution
            generated_data = model(sample_epanechnikov(key, mu=jnp.zeros(2), sigma=jnp.eye(2), num_samples=batch.shape[0]), reverse=False)[0]
            plt.scatter(generated_data[:, 0], generated_data[:,1], c='red', alpha=0.15, label='Generated')
            plt.legend()
            
            # Loss curve
            plt.subplot(1,3,2)
            plt.title('Training Loss')
            plt.plot(losses)
            plt.yscale('log')
            
            # Property tracking
            plt.subplot(1,3,3)
            plt.title('Transformation Properties')
            for prop_name, prop_values in property_logs.items():
                plt.plot(prop_values, label=prop_name)
            plt.legend()
            plt.yscale('log')
            
            plt.tight_layout()
            plt.show()
    
    return model, losses, property_logs

# Run the training
model, losses, property_logs = run_training()
