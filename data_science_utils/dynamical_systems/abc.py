import equinox as eqx
import abc
from jaxtyping import Array, Float, Key


class AbstractDynamicalSystem(eqx.Module, strict=True):
    """Abstract base class for dynamical systems in stochastic filtering."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the state space."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def initial_state(self) -> Float[Array, "state_dim"]:
        """Return a default initial state."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        x: Float[Array, "*batch state_dim"],
    ) -> Float[Array, "*batch state_dim"]:
        """Propagate the state forward in time."""
        raise NotImplementedError

    @abc.abstractmethod
    def backward(
        self,
        x: Float[Array, "*batch state_dim"],
    ) -> Float[Array, "*batch state_dim"]:
        """Propagate the state backward in time."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate(
        self,
        key: Key[Array, "..."],
        batch_size: int,
    ) -> Float[Array, "batch_size state_dim"]:
        """Generate random samples from the system."""
        raise NotImplementedError
