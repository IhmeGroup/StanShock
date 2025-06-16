from __future__ import annotations

import numpy as np

from stanshock.numerics.boundary_conditions import BoundaryConditions
from stanshock.numerics.face_extrapolation import FaceExtrapolator
from stanshock.numerics.gradient import Gradient
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide
from stanshock.system.geometry import Geometry


class ViscousFlux(RightHandSide):
    def __init__(
        self,
        boundary_conditions: BoundaryConditions,
        face_extrapolator: FaceExtrapolator,
        geometry: Geometry,
        gradient: Gradient,
    ) -> None:
        self.face_extrapolator = face_extrapolator
        self.boundary_conditions = boundary_conditions
        self.geometry = geometry
        self.gradient = gradient
        self.F = 1.0

    def source(
        self,
        time: float,
        state_array: Array,
        physics: FluidPhysics,
        gamma_star: Array,
    ) -> Array:
        state_array = self.boundary_conditions.update_ghost_layers(time, state_array)

        state: FluidState = physics.conservative_to_primitive(state_array, gamma_star)
        state.gamma = gamma_star

        face_states: FluidState = self.face_extrapolator(state)
        face_states = self.boundary_conditions.update_face_states(time, face_states)

        state.temperature = physics.get_temperature(state)
        face_gradients = self.gradient.face_gradients(state, self.geometry)

        return self.source_implementation(physics, face_states, face_gradients)

    def source_implementation(
        self,
        physics: FluidPhysics,
        face_states: FluidState,
        face_gradients: FluidState,
    ) -> Array:
        # Compute properties at the extrapolated cell faces
        viscosity = physics.get_mu(face_states)
        conductivity = physics.get_thermal_conductivity(face_states) * self.F
        diffusivities = physics.get_mass_diffusivity(face_states) * self.F

        # Average the properties from either side
        density = 0.5 * (face_states.density[0, :] + face_states.density[1, :])
        viscosity = 0.5 * (viscosity[0, :] + viscosity[1, :])
        conductivity = 0.5 * (conductivity[0, :] + conductivity[1, :])
        diffusivities = 0.5 * (diffusivities[0, :, :] + diffusivities[1, :, :])

        # Get gradients across the faces
        dudx = face_gradients.velocity
        dTdx = face_gradients.temperature
        dYdx = face_gradients.composition

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
        return (face_flux[1:, :] - face_flux[:-1, :]) / self.geometry.dx
