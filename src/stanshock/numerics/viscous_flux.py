from __future__ import annotations

import numpy as np

from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Unpack
from stanshock.system.base import PrecomputeStepName, PrecomputeSteps, RightHandSide


class ViscousFlux(RightHandSide):
    PRECOMPUTE_STEPS: tuple[PrecomputeStepName, ...] = (
        "geometry",
        "physics",
        "boundary_conditions",
        "face_extrapolator",
        "face_average",
        "gradient",
    )

    def __init__(
        self,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        super().__init__(**precompute_steps)
        self.F = 1.0

    def source_implementation(
        self,
        time: float,
        state_array: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        _ = time, state_array, state, face_states
        assert self.physics is not None
        assert avg_face_states is not None
        assert face_gradients is not None

        # Compute properties at the extrapolated cell faces
        viscosity = self.physics.get_mu(avg_face_states)
        conductivity = self.physics.get_thermal_conductivity(avg_face_states) * self.F
        diffusivities = self.physics.get_mass_diffusivity(avg_face_states) * self.F
        enthalpies = self.physics.get_species_enthalpies(avg_face_states)
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
        if self.physics.n_scalars_rho_sum > 1:
            correction = np.sum(
                diffusive_mass_flux[:, : self.physics.n_scalars_rho_sum],
                axis=-1,
                keepdims=True,
            )
            diffusive_mass_flux[:, : self.physics.n_scalars_rho_sum] -= (
                Y[:, : self.physics.n_scalars_rho_sum] * correction
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
