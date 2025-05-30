import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from diffrax import (
    ConstantStepSize,
    ODETerm,
    SaveAt,
    Tsit5,
    diffeqsolve,
    AbstractSolver,
    AbstractStepSizeController,
)

from data_science_utils.dynamical_systems import AbstractContinuousDynamicalSystem
from jaxtyping import Array, Float, Key, jaxtyped


@jaxtyped(typechecker=typechecker)
class Lorenz63(AbstractContinuousDynamicalSystem, strict=True):
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 0.01
    solver: AbstractSolver = Tsit5()
    stepsize_contoller: AbstractStepSizeController = ConstantStepSize()

    @property
    def dimension(self):
        return 3

    @jaxtyped(typechecker=typechecker)
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
                key, mean=jnp.zeros(self.dimension), cov=jnp.eye(self.dimension)
            )
        )

        return state + noise

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def vector_field(self, t, y, args):
        x, y, z = y
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return jnp.array([dx, dy, dz])
