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
        e0_star: Array,
    ) -> Array:
        state_array = self.boundary_conditions.update_ghost_layers(time, state_array)

        state: FluidState = physics.conservative_to_primitive(
            state_array, gamma_star, e0_star
        )

        face_states: FluidState = self.face_extrapolator(state)
        face_states = self.boundary_conditions.update_face_states(time, face_states)

        state.temperature = physics.get_temperature(state)
        face_gradients: FluidState = self.gradient.face_gradients(state, self.geometry)

        # Average the left and right face states - simple rho-P average for now
        r = 0.5 * (face_states.density[0, :] + face_states.density[1, :])
        P = 0.5 * (face_states.pressure[0, :] + face_states.pressure[1, :])
        Y = 0.5 * (face_states.composition[0, :] + face_states.composition[1, :])
        u = 0.5 * (face_states.velocity[0, :] + face_states.velocity[1, :])
        avg_face_states = FluidState(
            shape=r.shape,
            density=r,
            pressure=P,
            composition=Y,
            velocity=u,
        )

        return self.source_implementation(physics, avg_face_states, face_gradients)

    def source_implementation(
        self,
        physics: FluidPhysics,
        avg_face_states: FluidState,
        face_gradients: FluidState,
    ) -> Array:
        # Compute properties at the extrapolated cell faces
        viscosity = physics.get_mu(avg_face_states)
        conductivity = physics.get_thermal_conductivity(avg_face_states) * self.F
        diffusivities = physics.get_mass_diffusivity(avg_face_states) * self.F
        enthalpies = physics.get_species_enthalpies(avg_face_states)
        density = avg_face_states.density
        Y = avg_face_states.mass_fractions

        # Get gradients across the faces
        dudx = face_gradients.velocity
        dTdx = face_gradients.temperature
        dYdx = face_gradients.composition

        # Compute individual flux terms
        viscous_momentum_flux = 4.0 / 3.0 * viscosity * dudx
        viscous_heating = viscous_momentum_flux * avg_face_states.velocity
        diffusive_mass_flux = density[:, None] * diffusivities * dYdx

        # Apply correction-velocity to preserve mass continuity
        if physics.n_scalars_rho_sum > 1:
            correction = np.sum(
                diffusive_mass_flux[:, : physics.n_scalars_rho_sum],
                axis=-1,
                keepdims=True,
            )
            diffusive_mass_flux[:, : physics.n_scalars_rho_sum] -= (
                Y[:, : physics.n_scalars_rho_sum] * correction
            )

        # Compute energy flux due to mass diffusion
        diffusive_energy_flux = np.sum(diffusive_mass_flux * enthalpies, axis=-1)

        # Compute the fluxes
        face_flux = np.concatenate(
            (
                viscous_momentum_flux[:, None],
                (conductivity * dTdx - diffusive_energy_flux + viscous_heating)[
                    :, None
                ],
                diffusive_mass_flux,
            ),
            axis=1,
        )

        # Apply central difference
        return (face_flux[1:, :] - face_flux[:-1, :]) / self.geometry.dx
