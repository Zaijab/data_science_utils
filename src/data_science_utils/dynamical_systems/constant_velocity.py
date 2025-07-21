import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Key, jaxtyped
from beartype import beartype as typechecker
from data_science_utils.dynamical_systems import AbstractDiscreteDynamicalSystem
from typing import Literal


@jaxtyped(typechecker=typechecker)
class CVModel(AbstractDiscreteDynamicalSystem, strict=True):
    """
    Constant Velocity motion model.
    Describes a dynamical system in which a particle (x, v_x) updates according to:
    (x, v_x) -> (x + v_x * dt, v_x)
    (x, v_x, y, v_y) -> (x + v_x * dt, y + v_y * dt)
    """

    position_dimension: int
    process_noise_std: float
    transition_matrix: Float[
        Array, "2*{self.position_dimension} 2*{self.position_dimension}"
    ]
    process_noise_matrix: Float[Array, "2*{self.position_dimension} 1"]
    ordering: Literal["vo", "durant"]

    @property
    def dimension(self):
        return 2 * self.position_dimension

    def initial_state(self, key=None):
        return jnp.zeros(self.dimension)

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        position_dimension: int = 3,
        sampling_period: float = 1.0,
        process_noise_std: float = 5.0,
        ordering: Literal["vo", "durant"] = "vo",
        debug: bool = False,
    ):
        self.position_dimension = position_dimension
        self.process_noise_std = process_noise_std
        self.ordering = ordering

        if ordering == "vo":
            single_dimension_transition_matrix: Float[Array, "2 2"] = jnp.array(
                [[1.0, sampling_period], [0.0, 1.0]]
            )

            self.transition_matrix: Float[
                Array, "2*{position_dimension} 2*{position_dimension}"
            ] = jnp.kron(
                jnp.eye(position_dimension), single_dimension_transition_matrix
            )

            single_dimensional_process_noise_matrix: Float[Array, "2 1"] = jnp.array(
                [[(sampling_period**2) / 2], [sampling_period]]
            )

            self.process_noise_matrix: Float[Array, "2*{position_dimension} 1"] = (
                jnp.tile(
                    single_dimensional_process_noise_matrix, (position_dimension, 1)
                )
            )
        elif ordering == "durant":
            # State: (x, y, z, v_x, v_y, v_z)
            # Upper-right block has dt * I, others are identity or zero
            state_dim = 2 * position_dimension
            F = jnp.zeros((state_dim, state_dim))

            # Position equations: x_{k+1} = x_k + v_k * dt
            F = F.at[:position_dimension, :position_dimension].set(
                jnp.eye(position_dimension)
            )
            F = F.at[:position_dimension, position_dimension:].set(
                sampling_period * jnp.eye(position_dimension)
            )

            # Velocity equations: v_{k+1} = v_k
            F = F.at[position_dimension:, position_dimension:].set(
                jnp.eye(position_dimension)
            )

            self.transition_matrix = F

            # Process noise matrix: acceleration affects position and velocity
            B = jnp.zeros((state_dim, position_dimension))

            # Position affected by (dt^2)/2 * acceleration
            B = B.at[:position_dimension, :].set(
                ((sampling_period**2) / 2) * jnp.eye(position_dimension)
            )

            # Velocity affected by dt * acceleration
            B = B.at[position_dimension:, :].set(
                sampling_period * jnp.eye(position_dimension)
            )

            self.process_noise_matrix = B

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def forward(
        self,
        state: Float[Array, "2*{self.position_dimension}"],
        key: None | Key[Array, "..."] = None,
        debug: bool = False,
    ) -> Float[Array, "2*{self.position_dimension}"]:
        next_state = self.transition_matrix @ state
        if debug:
            assert isinstance(next_state, Float[Array, "2*{self.position_dimension}"])

        if key is not None:
            noise = (
                self.process_noise_matrix
                * self.process_noise_std
                * jax.random.normal(key)
            )
            if debug:
                assert isinstance(noise, Float[Array, "2*{self.position_dimension} 1"])

            next_state = next_state + noise

        return next_state
