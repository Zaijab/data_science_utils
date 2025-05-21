import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from diffrax import ConstantStepSize, ODETerm, SaveAt, Tsit5, diffeqsolve
from data_science_utils.dynamical_systems import AbstractDynamicalSystem
from jaxtyping import Array, Float, Key, jaxtyped


class Lorenz63(AbstractDynamicalSystem, strict=True):
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01

    @property
    def dimension(self):
        return 3

    @eqx.filter_jit
    def initial_state(
        self,
        key: Key[Array, "..."] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        state = jnp.array([8.0, 0.0, 0.0])

        noise = (
            0
            if key is None
            else jax.random.multivariate_normal(
                key, shape=(1,), mean=state, cov=jnp.eye(self.dimension)
            )
        )

        return state + noise

    @eqx.filter_jit
    def vector_field(self, t, y, args):
        x, y, z = y
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return jnp.array([dx, dy, dz])

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def trajectory(
        self,
        initial_time: float,
        final_time: float,
        state: Float[Array, "3"],
        saveat: SaveAt,
    ) -> tuple[Float[Array, "..."], Float[Array, "... 3"]]:
        """Integrate a single point forward in time."""
        term = ODETerm(self.vector_field)
        solver = Tsit5()

        sol = diffeqsolve(
            term,
            solver,
            t0=initial_time,
            t1=final_time,
            dt0=self.dt,
            y0=state,
            stepsize_controller=ConstantStepSize(),
            saveat=saveat,
            max_steps=100_000,
        )
        return sol.ts, sol.ys

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def generate(
        self, key: Key[Array, "..."], batch_size: int = 1000, spin_up_steps: int = 100
    ) -> Float[Array, "{batch_size} 3"]:
        keys = jax.random.split(key, batch_size)
        initial_states = eqx.filter_vmap(self.initial_state)(keys)
        final_states = eqx.filter_vmap(self.flow)(0, 30, initial_states)
        return final_states
