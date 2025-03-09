import jax
import jax.numpy as jnp
from flax import nnx
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

# Use static shape to avoid JAX tracer issues
def create_toy_data(key, n_samples=1000, n_dim=2):
    """Generate synthetic data for testing."""
    key1, key2 = jax.random.split(key)
    # Simple 2D distribution: a scaled and rotated Gaussian
    mean = jnp.array([2.0, -1.0], dtype=jnp.float64)
    samples = jax.random.normal(key1, (n_samples, n_dim), dtype=jnp.float64)
    
    # Create a correlation between dimensions
    rotation = jnp.array([[0.8, -0.6], [0.6, 0.8]], dtype=jnp.float64)
    samples = samples @ rotation + mean
    
    return samples

# Simple dense network with consistent implementation
class SimpleNet(nnx.Module):
    def __init__(self, output_dim, rngs, name=None):
        super().__init__(name=name)
        # Deliberately small network to avoid complex nesting
        self.layer1 = nnx.Linear(2, 32, rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64)
        self.layer2 = nnx.Linear(32, output_dim, rngs=rngs, dtype=jnp.float64, param_dtype=jnp.float64)
        
    def __call__(self, x):
        x = self.layer1(x)
        x = jnp.tanh(x)
        x = self.layer2(x)
        return x

# Affine coupling layer with simplified implementation
class AffineCoupling(nnx.Module):
    def __init__(self, rngs, name=None):
        super().__init__(name=name)
        # Scale and translation networks
        self.scale_net = SimpleNet(1, rngs, name="scale_net")
        self.translate_net = SimpleNet(1, rngs, name="translate_net")
        
    def forward(self, x):
        # Split input
        x1, x2 = x[:, 0:1], x[:, 1:2]
        
        # Calculate scale and translation based on x1
        log_scale = self.scale_net(x1)  # shape: (batch, 1)
        translation = self.translate_net(x1)  # shape: (batch, 1)
        
        # Apply transformation to x2
        y2 = x2 * jnp.exp(log_scale) + translation
        
        # Recombine and return
        y = jnp.concatenate([x1, y2], axis=1)
        ldj = jnp.sum(log_scale, axis=1)  # Log determinant Jacobian
        
        return y, ldj
    
    def backward(self, y):
        # Split input
        y1, y2 = y[:, 0:1], y[:, 1:2]
        
        # Calculate scale and translation based on y1 (which is same as x1)
        log_scale = self.scale_net(y1)
        translation = self.translate_net(y1)
        
        # Apply inverse transformation
        x2 = (y2 - translation) * jnp.exp(-log_scale)
        
        # Recombine and return
        x = jnp.concatenate([y1, x2], axis=1)
        ldj = -jnp.sum(log_scale, axis=1)  # Negative log determinant
        
        return x, ldj
    
    def __call__(self, x, reverse=False):
        if not reverse:
            return self.forward(x)
        else:
            return self.backward(x)

# Normalizing flow with simplified design
class SimpleFlow(nnx.Module):
    def __init__(self, num_layers=3, rngs=None, name=None):
        super().__init__(name=name)
        
        if rngs is None:
            rngs = nnx.Rngs(0)
            
        # Create coupling layers
        self.layers = []
        for i in range(num_layers):
            # Alternate which variable is transformed
            layer = AffineCoupling(rngs=rngs, name=f"coupling_{i}")
            self.layers.append(layer)
            
            # Add permutation between layers (swap dimensions)
            # This is implemented directly rather than as a separate layer

    def forward(self, x):
        # Initialize log determinant
        ldj_total = jnp.zeros(x.shape[0], dtype=jnp.float64)
        
        # Pass through each layer
        for i, layer in enumerate(self.layers):
            # Apply layer
            x, ldj = layer(x, reverse=False)
            ldj_total += ldj
            
            # Apply permutation (swap dimensions) between layers except after the last
            if i < len(self.layers) - 1:
                x = x[:, ::-1]  # Simple reverse of features
                
        return x, ldj_total
    
    def backward(self, z):
        # Initialize log determinant
        ldj_total = jnp.zeros(z.shape[0], dtype=jnp.float64)
        
        # Pass through each layer in reverse
        for i in reversed(range(len(self.layers))):
            # Undo permutation (swap dimensions) before layer except for first layer
            if i < len(self.layers) - 1:
                z = z[:, ::-1]  # Simple reverse of features
                
            # Apply layer in reverse
            z, ldj = self.layers[i](z, reverse=True)
            ldj_total += ldj
            
        return z, ldj_total
    
    def __call__(self, x, reverse=False):
        if not reverse:
            return self.forward(x)
        else:
            return self.backward(x)

# Standard normal log probability (scalar output)
def standard_normal_logprob(z):
    return -0.5 * jnp.sum(z**2, axis=1) - z.shape[1] * 0.5 * jnp.log(2 * jnp.pi)

# Define training step without JIT initially to debug
def train_step(model, optimizer, batch):
    def loss_fn(model):
        # Transform data to latent space
        z, ldj = model(batch, reverse=True)  # go from data to latent
        
        # Calculate negative log likelihood
        log_prob = standard_normal_logprob(z) + ldj
        loss = -jnp.mean(log_prob)
        
        return loss
    
    # Get loss and gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    
    # Update model parameters
    optimizer.update(grads)
    
    return loss

# Once verified working, add JIT
train_step_jit = train_step  # start without JIT for debugging

# Test with toy data
key = jax.random.key(0)
n_samples = 1000
batch_size = 100
data = create_toy_data(key, n_samples)

# Create model with explicit RNG
key, subkey = jax.random.split(key)
rngs = nnx.Rngs(params=subkey)
model = SimpleFlow(num_layers=3, rngs=rngs)

# Create optimizer
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=1e-3))

# Run a few training steps
n_steps = 5
for i in range(n_steps):
    batch_idx = i % (n_samples // batch_size)
    batch = data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    loss = train_step(model, optimizer, batch)
    print(f"Step {i+1}, Loss: {loss:.4f}")

# After confirming it works, we can try the JIT version
try:
    # Apply JIT to training step
    train_step_jit = nnx.jit(train_step)
    
    # Test the JIT version
    batch = data[:batch_size]
    loss = train_step_jit(model, optimizer, batch)
    print(f"JIT step successful, Loss: {loss:.4f}")
except Exception as e:
    print(f"JIT step failed with error: {str(e)}")
    print("Continuing with non-JIT version...")
    train_step_jit = train_step

# Run a short training loop
epochs = 10
steps_per_epoch = n_samples // batch_size
losses = []

for epoch in range(epochs):
    epoch_losses = []
    for step in range(steps_per_epoch):
        # Get batch
        batch_idx = step % (n_samples // batch_size)
        batch = data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        
        # Train
        loss = train_step_jit(model, optimizer, batch)
        epoch_losses.append(loss)
    
    # Track losses
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")

# Generate samples from the model
key, subkey = jax.random.split(key)
z = jax.random.normal(subkey, (n_samples, 2), dtype=jnp.float64)
generated, _ = model(z, reverse=False)

# Visualize results
plt.figure(figsize=(12, 4))

# Original data
plt.subplot(1, 3, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
plt.title("Original Data")
plt.grid(True)

# Generated samples
plt.subplot(1, 3, 2)
plt.scatter(generated[:, 0], generated[:, 1], alpha=0.5, s=10, color='red')
plt.title("Generated Samples")
plt.grid(True)

# Loss curve
plt.subplot(1, 3, 3)
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Neg. Log Likelihood")
plt.grid(True)

plt.tight_layout()
plt.show()

# Test invertibility
test_data = data[:5]
print("\nTesting invertibility:")
print("Original data:")
print(test_data)

# Forward direction (data to latent)
latent, ldj_forward = model(test_data, reverse=True)
print("\nTransformed to latent space:")
print(latent)

# Backward direction (latent to data)
reconstructed, ldj_backward = model(latent, reverse=False)
print("\nReconstructed data:")
print(reconstructed)

# Error metrics
recon_error = jnp.mean(jnp.abs(test_data - reconstructed))
print(f"\nReconstruction error: {recon_error:.8f}")
print(f"Sum of log determinants: {jnp.mean(ldj_forward + ldj_backward):.8f} (should be close to 0)")
