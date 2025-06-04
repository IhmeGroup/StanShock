from __future__ import annotations

import numpy as np

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
        # Compute properties at the extrapolated cell faces
        face_states.temperature = physics.get_temperature(face_states)
        viscosity = physics.get_mu(face_states)
        conductivity = physics.get_thermal_conductivity(face_states) * self.F
        diffusivities = physics.get_mass_diffusivity(face_states) * self.F

        # Average the properties from either side
        density = 0.5 * (face_states.density[0, :] + face_states.density[1, :])
        viscosity = 0.5 * (viscosity[0, :] + viscosity[1, :])
        conductivity = 0.5 * (conductivity[0, :] + conductivity[1, :])
        diffusivities = 0.5 * (diffusivities[0, :, :] + diffusivities[1, :, :])

        # Compute gradients across the faces via central difference
        dudx = (face_states.velocity[1, :] - face_states.velocity[0, :]) / self.dx
        dTdx = (face_states.temperature[1, :] - face_states.temperature[0, :]) / self.dx
        dYdx = (
            face_states.composition[1, :, :] - face_states.composition[0, :, :]
        ) / self.dx

        # Compute the fluxes
        face_flux = np.concatenate(
            (
                (4.0 / 3.0 * viscosity * dudx)[:, None],
                (conductivity * dTdx)[:, None],
                density[:, None] * diffusivities * dYdx,
            ),
            axis=1,
        )

        # Apply central difference
        return (face_flux[1:, :] - face_flux[:-1, :]) / self.dx
