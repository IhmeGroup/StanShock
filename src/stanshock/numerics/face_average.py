from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array


class FaceAverage(ABC):
    @abstractmethod
    def __call__(self, face_states: FluidState) -> FluidState:
        """Average the left & right extrapolated states at the faces."""


class SimpleAverage(FaceAverage):
    def __init__(
        self, state_variables: Literal["T-P", "rho-P", "rho-T"] = "rho-P"
    ) -> None:
        """Simple averaging to the face.

        Uses the specified state variables for the thermodynamic state.
        """
        self.state_variables = state_variables

    def __call__(self, face_states: FluidState) -> FluidState:
        r = None
        if "rho" in self.state_variables:
            assert face_states.density is not None
            r = 0.5 * (face_states.density[0, :] + face_states.density[1, :])

        P = None
        if "P" in self.state_variables:
            assert face_states.pressure is not None
            P = 0.5 * (face_states.pressure[0, :] + face_states.pressure[1, :])

        T = None
        if "T" in self.state_variables:
            assert face_states.temperature is not None
            T = 0.5 * (face_states.temperature[0, :] + face_states.temperature[1, :])

        assert face_states.velocity is not None
        u = 0.5 * (face_states.velocity[0, :] + face_states.velocity[1, :])

        assert face_states.composition is not None
        Y = 0.5 * (face_states.composition[0, :] + face_states.composition[1, :])

        return FluidState(
            shape=u.shape,
            density=r,
            pressure=P,
            temperature=T,
            composition=Y,
            velocity=u,
        )


class RoeAverage(FaceAverage):
    def __init__(self, physics: FluidPhysics) -> None:
        """Apply Roe-averaging to face states."""
        self.physics = physics

    def roe_average(self, D: Array, phi: Array) -> Array:
        """Perform Roe-average to phi, given square root of the density ratio, D."""
        return (D * phi[1] + phi[0]) / (D + 1.0)

    def __call__(self, face_states: FluidState) -> FluidState:
        assert face_states.density is not None
        D = np.sqrt(face_states.density[1, :] / face_states.density[0, :])
        r = np.sqrt(face_states.density[0, :] * face_states.density[1, :])

        assert face_states.pressure is not None
        P = self.roe_average(D, face_states.pressure)

        assert face_states.velocity is not None
        u = self.roe_average(D, face_states.velocity)

        assert face_states.composition is not None
        Y = self.roe_average(D[:, None], face_states.composition)

        return FluidState(
            shape=u.shape,
            density=r,
            pressure=P,
            composition=Y,
            velocity=u,
        )
