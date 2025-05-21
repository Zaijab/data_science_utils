import equinox as eqx
import abc
from jaxtyping import Array, Float, Key
from beartype import beartype as typechecker
from diffrax import (
    SaveAt,
    ODETerm,
    diffeqsolve,
    AbstractSolver,
    AbstractStepSizeController,
)


class AbstractDynamicalSystem(eqx.Module, strict=True):
    """Abstract base class for dynamical systems in stochastic filtering."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the state space."""
        raise NotImplementedError

    @abc.abstractmethod
    def initial_state(
        self,
        key: Key[Array, "..."] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        """
        Return a default initial state.
        Many dynamical systems have a cannonical / useful state that they start from.
        We have `None` act at this singular state and a `jax.random.key` will initialize the point in a random manner.
        This will be useful for generating points in an attractor if need be.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def trajectory(
        self,
        initial_time: float,
        final_time: float,
        state: Float[Array, "state_dim"],
        saveat: SaveAt,
    ) -> tuple[Float[Array, "..."], Float[Array, "... state_dim"]]:
        """
        Solve for the trajectory given boundary times (and how many points to save).
        Return the times and corresponding solutions.
        """
        raise NotImplementedError

    def flow(
        self,
        initial_time: float,
        final_time: float,
        state: Float[Array, "state_dim"],
    ) -> Float[Array, "state_dim"]:
        """
        Trajectory with SaveAt = t1.
        Returns the y value at t1
        """
        _, states = self.trajectory(
            initial_time=initial_time,
            final_time=final_time,
            state=state,
            saveat=SaveAt(t1=True),
        )
        return states[-1]

    def orbit(
        self,
        initial_time: float,
        final_time: float,
        state: Float[Array, "state_dim"],
        saveat: SaveAt,
    ) -> Float[Array, "state_dim"]:
        """
        Trajectory but just return ys.
        """
        _, states = self.trajectory(initial_time, final_time, state, saveat)
        return states


class AbstractContinuousSystem(AbstractDynamicalSystem, strict=True):
    solver: AbstractSolver
    stepsize_contoller: AbstractStepSizeController

    @abc.abstractmethod
    def vector_field():
        raise NotImplementedError

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def trajectory(
        self,
        initial_time: float,
        final_time: float,
        state: Float[Array, "{self.dimension}"],
        saveat: SaveAt,
    ) -> tuple[Float[Array, "..."], Float[Array, "... {self.dimension}"]]:
        """Integrate a single point forward in time."""

        sol = diffeqsolve(
            term=ODETerm(self.vector_field),
            solver=self.solver,
            t0=initial_time,
            t1=final_time,
            dt0=self.dt,
            y0=state,
            stepsize_controller=self.stepsize_contoller,
            saveat=saveat,
            max_steps=100_000,
        )
        return sol.ts, sol.ys


class AbstractDiscreteSystem(AbstractDynamicalSystem, strict=True):

    @abc.abstractmethod
    def forward():
        raise NotImplementedError

    @eqx.filter_jit
    def trajectory(
        self,
        initial_time: float,
        final_time: float,
        state: Float[Array, "state_dim"],
        saveat: SaveAt,
        discrete_update: callable,
    ):
        """
        This function computes the trajectory for a discrete system.
        It returns the tuple of the times and
        """
        assert initial_time <= final_time, "This is a discrete system without inverse."

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
                return (self.forward(sub_state), sub_time + 1)

            final_state, final_time = jax.lax.while_loop(
                sub_while_cond_fun, sub_while_body_fun, carry
            )

            return (final_state, final_time), final_state

        initial_carry = (state, 0)
        (final_state, final_time), states = jax.lax.scan(body_fn, initial_carry, xs)

        return xs, states


class AbstractInvertibleDiscreteSystem(AbstractDynamicalSystem, strict=True):

    @abc.abstractmethod
    def forward():
        raise NotImplementedError

    @abc.abstractmethod
    def backward():
        raise NotImplementedError

    @eqx.filter_jit
    def trajectory(
        initial_time: float,
        final_time: float,
        state: Float[Array, "state_dim"],
        saveat: SaveAt,
        discrete_update: callable,
    ):
        """
        This function computes the trajectory for a discrete system.
        It returns the tuple of the times and
        """
        step_function = self.forward if final_time >= initial_time else self.backward

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
                return (self.forward(sub_state), sub_time + 1)

            final_state, final_time = jax.lax.while_loop(
                sub_while_cond_fun, sub_while_body_fun, carry
            )

            return (final_state, final_time), final_state

        initial_carry = (state, 0)
        (final_state, final_time), states = jax.lax.scan(body_fn, initial_carry, xs)

        return xs, states
