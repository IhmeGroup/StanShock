from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from stanshock.physics.fluid_base import FluidPhysics, FluidState

# Define the Array type for use in type hints to make future changes easier
Array = np.ndarray


class RightHandSide(ABC):
    @abstractmethod
    def __call__(self, time: float, state: FluidState, physics: FluidPhysics) -> Array:
        """Compute the temporal gradient of the current state of the system."""
