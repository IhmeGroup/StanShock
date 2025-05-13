from __future__ import annotations

from stanshock.numerics.face_extrapolation import FaceExtrapolator
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide


class ViscousFlux(RightHandSide):
    def __init__(
        self, face_extrapolator: FaceExtrapolator, boundary_conditions, dx
    ) -> None:
        self.face_extrapolator = face_extrapolator
        self.boundary_conditions = boundary_conditions
        self.dx = dx
        self.F = 1.0

    def source(
        self, _time: float, state_array: Array, physics: FluidPhysics, gamma_star: Array
    ) -> Array:
        state = physics.conservative_to_primitive(state_array, gamma_star)
        state.gamma = gamma_star

        face_states = self.face_extrapolator(state)
        face_states = self.boundary_conditions(face_states)

        return self.source_from_primitives(_time, face_states, physics)

    def source_from_primitives(
        self, _time: float, face_states: FluidState, physics: FluidPhysics
    ) -> Array:
        # Compute temperature
        face_states.temperature = physics.get_temperature(face_states)

        # Compute gradients across the faces via central difference
        dudx = (face_states.velocity[1, :] - face_states.velocity[0, :]) / self.dx
        dTdx = (face_states.temperature[1, :] - face_states.temperature[0, :]) / self.dx
        dYdx = (
            face_states.composition[1, :, :] - face_states.composition[0, :, :]
        ) / self.dx

        # Use the physics model to compute the viscous fluxes
        # from the face states and gradients
        face_flux = physics.get_viscous_flux(face_states, dudx, dTdx, dYdx)

        # Flame thickening (applies to energy and species fluxes)
        face_flux[:, 2:] *= self.F

        # Apply central difference
        return (face_flux[1:, :] - face_flux[:-1, :]) / self.dx
