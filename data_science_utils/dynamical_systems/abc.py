import equinox as eqx
import abc
from jaxtyping import Array, Float, Key
from diffrax import SaveAt


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
        key: Key[Array, ...] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        """
        Return a default initial state.
        Many dynamical systems have a cannonical / useful state that they start from.
        We have `None` act at this singular state and a `jax.random.key` will initialize the point in a random manner.
        This will be useful for generating points in an attractor if need be.
        """
        raise NotImplementedError

    def trajectory(
        self,
        initial_time: float,
        final_time: float,
        state: Float[Array, "state_dim"],
        saveat: SaveAt,
    ) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
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
