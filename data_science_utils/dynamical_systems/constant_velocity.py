import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Key, jaxtyped
from beartype import beartype as typechecker


# @jaxtyped(typechecker=typechecker)
# class CVModel(eqx.Module):
#     """
#     Constant Velocity motion model.
#     Describes a dynamical system in which a particle (x, v_x) updates according to:
#     (x, v_x) -> (x + v_x * dt)
#     (x, v_x, y, v_y) -> (x + v_x * dt, y + v_y * dt)
#     """

#     position_dimension: int
#     process_noise_std: float
#     transition_matrix: Float[
#         Array, "2*{self.position_dimension} 2*{self.position_dimension}"
#     ]
#     process_noise_matrix: Float[Array, "2*{self.position_dimension} 1"]

#     def __init__(
#         self,
#         position_dimension: int = 3,
#         sampling_period: float = 1.0,
#         process_noise_std: float = 5.0,
#         debug: bool = False,
#     ):
#         self.position_dimension = position_dimension
#         self.process_noise_std = process_noise_std

#         single_dimension_transition_matrix: Float[Array, "2 2"] = jnp.array(
#             [[1.0, sampling_period], [0.0, 1.0]]
#         )

#         self.transition_matrix: Float[
#             Array, "2*{position_dimension} 2*{position_dimension}"
#         ] = jnp.kron(jnp.eye(position_dimension), single_dimension_transition_matrix)

#         single_dimensional_process_noise_matrix: Float[Array, "2 1"] = jnp.array(
#             [[(sampling_period**2) / 2], [sampling_period]]
#         )

#         self.process_noise_matrix: Float[Array, "2*{position_dimension} 1"] = jnp.tile(
#             single_dimensional_process_noise_matrix, (position_dimension, 1)
#         )

#     @jaxtyped(typechecker=typechecker)
#     @partial(jax.jit, static_argnames=["debug"])
#     def forward(
#         self,
#         state: Float[Array, "2*{self.position_dimension} 1"],
#         key: None | Key[Array, "..."] = None,
#         debug: bool = False,
#     ) -> Float[Array, "2*{self.position_dimension} 1"]:
#         next_state = self.transition_matrix @ state
#         if debug:
#             assert isinstance(next_state, Float[Array, "2*{self.position_dimension} 1"])

#         if key is not None:
#             noise = (
#                 self.process_noise_matrix
#                 * self.process_noise_std
#                 * jax.random.normal(key)
#             )
#             if debug:
#                 assert isinstance(noise, Float[Array, "2*{self.position_dimension} 1"])

#             next_state = next_state + noise

#         return next_state


@jaxtyped(typechecker=typechecker)
class MaskedCVModel(eqx.Module):
    """
    Constant Velocity model for multiple particles with birth/death processes.

    State: [max_particles, state_dim] where state_dim = 2 * position_dimension
    Mask: [max_particles] boolean array indicating active particles
    """

    position_dimension: int
    max_particles: int
    process_noise_std: float
    transition_matrix: Float[Array, "state_dim state_dim"]
    process_noise_matrix: Float[Array, "state_dim position_dimension"]
    survival_probability: float

    def __init__(
        self,
        position_dimension: int = 3,
        max_particles: int = 100,
        sampling_period: float = 1.0,
        process_noise_std: float = 5.0,
        survival_probability: float =  0.99
    ):
        self.position_dimension = position_dimension
        self.max_particles = max_particles
        self.process_noise_std = process_noise_std

        # Build transition matrix for single particle
        single_dim_F = jnp.array([[1.0, sampling_period], [0.0, 1.0]])
        self.transition_matrix = jnp.kron(jnp.eye(position_dimension), single_dim_F)

        # Process noise matrix: maps position_dimension noise to state_dim
        single_dim_G = jnp.array([[(sampling_period**2) / 2], [sampling_period]])
        self.process_noise_matrix = jnp.kron(jnp.eye(position_dimension), single_dim_G)

        self.survival_probability = survival_probability

    def initial_states(self):
        return jnp.zeros((50, 4)), jnp.zeros(50, dtype=bool)


    def survival_process(self, states, mask, key):
        # Find out how many states will survive
        survived = jax.random.bernoulli(key, shape=(self.max_particles,), p=self.survival_probability)
        mask = mask & (survived)
        return states, mask

    def birth_process(self, states, mask, key):
        # Find out how many to birth
        # Place the birthed states in the proper place
        return states, mask
    
    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def forward(
        self,
        states: Float[Array, "max_particles state_dim"],
        mask: Bool[Array, "max_particles"],
        key: Key[Array, ""] | None = None,
    ) -> Float[Array, "max_particles state_dim"]:
        """Apply CV dynamics only to active particles."""

        # Find out how many states survived
        key, survival_key = jax.random.split(key)
        states, mask = self.survival_process(states, mask, survival_key)

        # Of those states, apply the CV
        states = self.transition_matrix @ states

        # For every state, add process noise
        # This noise is nothing actually, no process noise

        # No extra births from the dynamical system

        # Just let garbage data sit in states, use mask for anything important
        return states, mask



if __name__ == "__main__":
    dynamical_system = MaskedCVModel(position_dimension=2, max_particles=50)
    states, masks = dynamical_system.initial_states()

    dynamical_system.

    birth_states = jnp.array([[1.0, 0.1, 2.0, -0.1], [3.0, 0.2, 1.0, 0.3]])
    states, mask = model.birth_process(states, mask, birth_states, key)

    # Forward dynamics
    states = model.forward_masked(states, mask, key)

    # Death process
    states, mask = model.death_process(states, mask, 0.95, key)
    print(states.at[mask].get())
