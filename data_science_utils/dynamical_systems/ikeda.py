from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax import lax, random
from jaxtyping import Array, Bool, Float, jaxtyped


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['u'])
def ikeda_forward(x: Float[Array, "*batch 2"], u: float = 0.9) -> Float[Array, "*batch 2"]:
    x1, x2 = x[..., 0], x[..., 1]
    t = 0.4 - (6 / (1 + x1**2 + x2**2))
    sin_t, cos_t = jnp.sin(t), jnp.cos(t)
    x1_new = 1 + u * (x1 * cos_t - x2 * sin_t)
    x2_new = u * (x1 * sin_t + x2 * cos_t)
    return jnp.stack((x1_new, x2_new), axis=-1)


@partial(jax.jit, static_argnames=["u"])
def ikeda_backward(
    x: Float[Array, "*batch 2"],
    u: float = 0.9
) -> Float[Array, "*batch 2"]:

    x1_unscaled = (x[..., 0] - 1) / u
    x2_unscaled = x[..., 1] / u
    t = 0.4 - (6 / (1 + x1_unscaled**2 + x2_unscaled**2))

    sin_t, cos_t = jnp.sin(t), jnp.cos(t)
    x1_prev = x1_unscaled * cos_t + x2_unscaled * sin_t
    x2_prev = -x1_unscaled * sin_t + x2_unscaled * cos_t

    return jnp.stack((x1_prev, x2_prev), axis=-1)


@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['batch_size', 'u'])
def ikeda_generate(key, batch_size: int = 10**5, u: float = 0.9) -> Array:
    def body_fn(i, val):
        return ikeda_forward(val, u)
    initial_state = random.uniform(key, shape=(batch_size, 2), minval=-0.25, maxval=0.25)
    return lax.fori_loop(0, 15, body_fn, initial_state)


@partial(jax.jit, static_argnames=["u", "ninverses"])
def ikeda_attractor_discriminator(
    x: Float[Array, "*batch 2"],
    ninverses: int = 10,
    u: float = 0.9
) -> Bool[Array, "*batch"]:
    # Pre-compute threshold
    threshold_squared = 1.0 / (1.0 - u)

    def scan_fn(state, _):
        x_curr, is_outside = state
        x_inv = ikeda_backward(x_curr, u)
        norm_squared = jnp.sum(x_inv**2, axis=-1)
        new_is_outside = is_outside | (norm_squared > threshold_squared)
        return (x_inv, new_is_outside), None

    # Initialize with False for "is_outside"
    init_state = (x, jnp.zeros(x.shape[:-1], dtype=bool))

    # Use scan instead of fori_loop
    (x_final, is_outside), _ = lax.scan(
        scan_fn,
        init_state,
        None,  # We don't need any carry values
        length=ninverses
    )

    # Return inverse: if ANY iteration went outside the threshold, it's not on attractor
    return ~is_outside

