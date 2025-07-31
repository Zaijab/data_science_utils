import jax
import jax.numpy as jnp
import equinox as eqx
from flax import nnx

def initialize_ensemble(key, batch_size):
    return jax.random.normal(key, (batch_size,))

def create_variable_measurement(key):
    n = jax.random.poisson(key, lam=10.0)
    n = jnp.minimum(n, 40)
    x = jax.random.normal(key, shape=(40,))
    return jax.lax.slice(x, (0,), (n,)) 

# @nnx.jit
@eqx.filter_jit
def create_ensemble_process_measurement(key):
    ensemble = initialize_ensemble(key, 10)
    measurements = create_variable_measurement(key)
    return measurements

print(create_ensemble_process_measurement(jax.random.key(0)))


###

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from jaxtyping import Array, Float
from beartype import beartype as typechecker

@jax.jit
def expensive_elementwise_op(x: Float[Array, "n"]) -> Float[Array, "n"]:
    result = x
    for _ in range(20):
        result = jnp.exp(jnp.sin(result)) + jnp.cos(jnp.log(jnp.abs(result) + 1e-8))
    return result

@jax.jit
def wrong_approach(x: Float[Array, "n"], mask: Float[Array, "n"]) -> Float[Array, "n"]:
    return jnp.where(mask, expensive_elementwise_op(x), jnp.nan)

@jax.jit  
def correct_approach(x: Float[Array, "n"], mask: Float[Array, "n"]) -> Float[Array, "n"]:
    safe_x = jnp.where(mask, x, 1.0)
    result = expensive_elementwise_op(safe_x)
    return jnp.where(mask, result, jnp.nan)

@jax.jit
def truly_optimized_approach(x: Float[Array, "n"], mask: Float[Array, "n"]) -> Float[Array, "n"]:
    valid_indices = jnp.where(mask, size=mask.sum(), fill_value=0)[0]
    valid_x = x[valid_indices]
    valid_result = expensive_elementwise_op(valid_x)
    result = jnp.full_like(x, jnp.nan)
    result = result.at[valid_indices].set(valid_result)
    return result

valid_elements = 64
total_sizes = [64, 128, 256, 512, 1024, 2048]
times_wrong, times_correct, times_optimized = [], [], []

for total_size in total_sizes:
    key = jax.random.key(42)
    x = jax.random.normal(key, (total_size,))
    mask = jnp.arange(total_size) < valid_elements
    
    # Warmup
    wrong_approach(x, mask).block_until_ready()
    correct_approach(x, mask).block_until_ready()
    truly_optimized_approach(x, mask).block_until_ready()
    
    # Time wrong approach
    start = time.perf_counter()
    for _ in range(10):
        wrong_approach(x, mask).block_until_ready()
    times_wrong.append((time.perf_counter() - start) / 10)
    
    # Time correct approach  
    start = time.perf_counter()
    for _ in range(10):
        correct_approach(x, mask).block_until_ready()
    times_correct.append((time.perf_counter() - start) / 10)
    
    # Time optimized approach
    start = time.perf_counter()
    for _ in range(10):
        truly_optimized_approach(x, mask).block_until_ready()
    times_optimized.append((time.perf_counter() - start) / 10)

plt.figure(figsize=(10, 6))
plt.loglog(total_sizes, times_wrong, 'ro-', label='Wrong: Post-mask expensive op', linewidth=2)
plt.loglog(total_sizes, times_correct, 'go-', label='Correct: Pre-mask inputs', linewidth=2)
plt.loglog(total_sizes, times_optimized, 'bo-', label='Truly optimized: Compute only valid', linewidth=2)
plt.xlabel('Total Array Size')
plt.ylabel('Time (seconds)')
plt.legend()
plt.title(f'Masked Computation Performance (Fixed {valid_elements} valid elements)')
plt.grid(True, alpha=0.3)
plt.show()

for i, size in enumerate(total_sizes):
    valid_fraction = valid_elements / size
    print(f"Size {size:4d} ({valid_fraction:5.1%} valid): "
          f"Wrong={times_wrong[i]:.4f}s, Correct={times_correct[i]:.4f}s, "
          f"Optimized={times_optimized[i]:.4f}s")

print(f"\nSpeedup at largest size:")
print(f"Correct vs Wrong: {times_wrong[-1]/times_correct[-1]:.2f}x")
print(f"Optimized vs Wrong: {times_wrong[-1]/times_optimized[-1]:.2f}x")

###
