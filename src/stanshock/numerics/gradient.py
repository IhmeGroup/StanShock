from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from stanshock.physics.fluid_base import FluidState
from stanshock.system.geometry import Geometry


class Gradient(ABC):
    @abstractmethod
    def face_gradients(
        self, cell_states: FluidState, geometry: Geometry, only: list[str] | None = None
    ) -> FluidState:
        """Compute property gradients across the interior cell faces."""


class CentralDifference:
    def __init__(self, mt: int = 1) -> None:
        self.mt = mt

    def face_gradients(
        self, cell_states: FluidState, geometry: Geometry, only: list[str] | None = None
    ) -> FluidState:
        """Compute property gradients across faces using central difference on a uniform mesh."""
        if only is None:
            only = ["velocity", "temperature", "composition"]

        mt = self.mt

        index_face_left = np.s_[mt - 1 : -mt]
        right = cell_states.shape[0] if mt == 1 else -mt + 1
        index_face_right = np.s_[mt:right]

        face_gradients = FluidState(shape=(cell_states.shape[0] + 1,))

        for var_name in only:
            var = getattr(cell_states, var_name)
            grad_var = (var[index_face_right] - var[index_face_left]) / geometry.dx
            setattr(face_gradients, var_name, grad_var)

        return face_gradients
