from abc import ABC, abstractmethod
from typing import Any
from jaxtyping import Array


class DynamicalSystem(ABC):
    state: Any

    @abstractmethod
    def flow(self, x: Array) -> Array:
        """
        Push given points forward along the flow of the dynamical system.
        """
        pass

    @abstractmethod
    def iterate(self) -> None:
        """
        Iterate the internal state of the dynamical system.
        """
        pass

    @abstractmethod
    def generate(self, key, batch_size):
        """
        Generate Points on the chaotic attractor of the set
        """
        pass
