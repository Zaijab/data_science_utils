from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax import lax, random
from jaxtyping import Array, Bool, Float, Key, jaxtyped
import equinox as eqx
from diffrax import SaveAt
from data_science_utils.dynamical_systems import AbstractInvertibleDiscreteSystem


class Ikeda(AbstractInvertibleDiscreteSystem, strict=True):
    u: float = 0.9
    batch_size: int = 10**3

    @property
    def dimension(self):
        return 2

    @eqx.filter_jit
    def initial_state(
        self,
        key: Key[Array, "..."] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        state = jnp.array([1.25, 0])
        noise = (
            jax.random.multivariate_normal(key, mean=state, cov=jnp.eye(self.dimension))
            if key
            else 0
        )
        return state + noise

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def forward(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        x1, x2 = x[0], x[1]
        t = 0.4 - (6 / (1 + x1**2 + x2**2))
        sin_t, cos_t = jnp.sin(t), jnp.cos(t)
        x1_new = 1 + self.u * (x1 * cos_t - x2 * sin_t)
        x2_new = self.u * (x1 * sin_t + x2 * cos_t)
        return jnp.stack((x1_new, x2_new), axis=-1)

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def backward(self, x: Float[Array, "2"]) -> Float[Array, "2"]:
        x1_unscaled = (x[0] - 1) / self.u
        x2_unscaled = x[1] / self.u
        t = 0.4 - (6 / (1 + x1_unscaled**2 + x2_unscaled**2))
        sin_t, cos_t = jnp.sin(t), jnp.cos(t)
        x1_prev = x1_unscaled * cos_t + x2_unscaled * sin_t
        x2_prev = -x1_unscaled * sin_t + x2_unscaled * cos_t
        return jnp.stack((x1_prev, x2_prev), axis=-1)

    @eqx.filter_jit
    def trajectory(
        self,
        initial_time: float,
        final_time: float,
        state: Float[Array, "state_dim"],
        saveat: SaveAt,
    ):
        safe_initial_time = (
            jnp.atleast_1d(initial_time) if saveat.subs.t0 else jnp.array([])
        )
        safe_final_time = (
            jnp.atleast_1d(final_time) if saveat.subs.t1 else jnp.array([])
        )
        safe_array = jnp.array([]) if saveat.subs.ts is None else saveat.subs.ts
        xs = jnp.concatenate([safe_initial_time, safe_array, safe_final_time])

        def body_fn(carry, x):
            """
            state = carry
            time = x
            """
            current_state, current_time = carry

            def sub_while_cond_fun(sub_carry):
                sub_state, sub_time = sub_carry
                return sub_time < x

            def sub_while_body_fun(sub_carry):
                sub_state, sub_time = sub_carry
                return (ikeda_forward(sub_state), sub_time + 1)

            final_state, final_time = jax.lax.while_loop(
                sub_while_cond_fun, sub_while_body_fun, carry
            )

            return (final_state, final_time), final_state

        initial_carry = (state, 0)
        (final_state, final_time), states = jax.lax.scan(body_fn, initial_carry, xs)

        return xs, states

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def generate(
        self, key: Key[Array, "..."], batch_size: int = 1000, spin_up_steps: int = 100
    ) -> Float[Array, "{batch_size} 2"]:
        keys = jax.random.split(key, batch_size)
        initial_states = eqx.filter_vmap(self.initial_state)(keys)
        final_states = eqx.filter_vmap(self.flow)(0, 30, initial_states)
        return final_states

    @jaxtyped(typechecker=typechecker)
    @partial(jax.jit, static_argnames=["batch_size", "u"])
    def generate(key, batch_size: int = 10**5, u: float = 0.9) -> Array:
        def body_fn(i, val):
            return ikeda_forward(val, u)

        initial_state = random.uniform(
            key, shape=(batch_size, 2), minval=-0.25, maxval=0.25
        )
        return lax.fori_loop(0, 15, body_fn, initial_state)

    @partial(jax.jit, static_argnames=["u", "ninverses"])
    def ikeda_attractor_discriminator(
        self, x: Float[Array, "*batch 2"], ninverses: int = 10, u: float = 0.9
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
            None,
            length=ninverses,  # We don't need any carry values
        )

        # Return inverse: if ANY iteration went outside the threshold, it's not on attractor
        return ~is_outside


from functools import partial

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jax import lax, random
from jaxtyping import Array, Bool, Float, Key, jaxtyped
import equinox as eqx
from data_science_utils.dynamical_systems import AbstractDynamicalSystem
from diffrax import SaveAt


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def ikeda_forward(x: Float[Array, "2"], u: float = 0.9) -> Float[Array, "2"]:
    x1, x2 = x[0], x[1]
    t = 0.4 - (6 / (1 + x1**2 + x2**2))
    sin_t, cos_t = jnp.sin(t), jnp.cos(t)
    x1_new = 1 + u * (x1 * cos_t - x2 * sin_t)
    x2_new = u * (x1 * sin_t + x2 * cos_t)
    return jnp.stack((x1_new, x2_new), axis=-1)


points = jax.random.normal(jax.random.key(0), shape=(100, 2))
eqx.filter_vmap(ikeda_forward)(points, 0.7)


@eqx.filter_jit
def ikeda_trajectory_jit(
    initial_time: float,
    final_time: float,
    state: Float[Array, "state_dim"],
    saveat: SaveAt,
):
    safe_initial_time = (
        jnp.atleast_1d(initial_time) if saveat.subs.t0 else jnp.array([])
    )
    safe_final_time = jnp.atleast_1d(final_time) if saveat.subs.t1 else jnp.array([])
    safe_array = jnp.array([]) if saveat.subs.ts is None else saveat.subs.ts
    xs = jnp.concatenate([safe_initial_time, safe_array, safe_final_time])

    def body_fn(carry, x):
        """
        state = carry
        time = x
        """
        current_state, current_time = carry

        def sub_while_cond_fun(sub_carry):
            sub_state, sub_time = sub_carry
            return sub_time < x

        def sub_while_body_fun(sub_carry):
            sub_state, sub_time = sub_carry
            return (ikeda_forward(sub_state), sub_time + 1)

        final_state, final_time = jax.lax.while_loop(
            sub_while_cond_fun, sub_while_body_fun, carry
        )

        return (final_state, final_time), final_state

    initial_carry = (state, 0)
    (final_state, final_time), states = jax.lax.scan(body_fn, initial_carry, xs)

    return xs, states


# data = jnp.array([1.25, 0])

# test1 = ikeda_trajectory_jit(
#     initial_time=0.0,
#     final_time=1.0,
#     state=data,
#     saveat=SaveAt(t1=True),
# )

# print(test1)
# print()

# test1 = ikeda_trajectory_jit(
#     initial_time=0.0,
#     final_time=1.0,
#     state=data,
#     saveat=SaveAt(t0=True),
# )

# print(test1)
# print()

# test1 = ikeda_trajectory_jit(
#     initial_time=0.0,
#     final_time=1.0,
#     state=data,
#     saveat=SaveAt(t0=True, t1=True),
# )

# print(test1)
# print()


# test1 = ikeda_trajectory_jit(
#     initial_time=0.0,
#     final_time=6.0,
#     state=data,
#     saveat=SaveAt(ts=jnp.array([1.0, 2.0, 3.0, 5.0])),
# )

# print(test1)
# print()

# test1 = ikeda_trajectory_jit(
#     initial_time=0.0,
#     final_time=100,
#     state=data,
#     saveat=SaveAt(ts=jnp.array([1.0, 2.0, 3.0, 87.0, 90]), t0=True),
# )

# print(test1)
# print()
