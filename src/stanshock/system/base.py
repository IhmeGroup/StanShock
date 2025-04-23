from __future__ import annotations

from abc import ABC, abstractmethod

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array


class RightHandSide(ABC):
    @abstractmethod
    def __call__(self, time: float, state: FluidState, physics: FluidPhysics) -> Array:
        """Compute the temporal gradient of the current state of the system."""
