from __future__ import annotations

import cantera as ct
import numpy as np

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array


class CanteraInterface(FluidPhysics):
    def __init__(self, gas: ct.Solution, ox_def=None, fuel_def=None, prog_def=None):
        super().__init__(gas, ox_def, fuel_def, prog_def)
        self._cached_solutions: dict[int, ct.SolutionArray] = {}

    def set_state(self, state: FluidState) -> FluidState:
        """Sets up a Cantera SolutionArray with the current fluid state.

        Caches SolutionArray objects of a given size for reuse, and checks if
        the state has been updated since the last call.
        """
        if state.shape not in self._cached_solutions:
            self._cached_solutions[state.shape] = ct.SolutionArray(
                self.gas, shape=state.shape
            )

        self.sol = self._cached_solutions[state.shape]

        # Update SolutionArray only if state has changed
        if not state._cache_valid:
            if self.gas.n_species > 1:
                state.mass_fractions = np.zeros(
                    (state.composition.shape)[:-1] + (self.gas.n_species,)
                )
                state.mass_fractions[..., :-1] = state.composition
                state.mass_fractions[..., -1] = 1.0 - np.sum(state.composition, axis=-1)
            else:
                state.mass_fractions = state.composition

            # Prioritize density-based updates
            if state.density is not None:
                if state.pressure is not None:
                    self.sol.DPY = state.density, state.pressure, state.mass_fractions
                else:
                    self.sol.TDY = (
                        state.temperature,
                        state.density,
                        state.mass_fractions,
                    )
                state.density = self.sol.density_mass
            else:
                self.sol.TPY = state.temperature, state.pressure, state.mass_fractions
                state.temperature = self.sol.T

        state._cache_valid = True

        return state

    def get_cp(self, state: FluidState):
        """Compute specific heat capacity at constant pressure."""
        self.set_state(state)
        state.cp = self.sol.cp_mass
        return state.cp

    def get_gamma(self, state: FluidState):
        """Compute specific heat ratio, gamma."""
        self.set_state(state)
        state.gamma = self.sol.cp / self.sol.cv
        return state.gamma

    def get_mu(self, state: FluidState):
        """Compute dynamic viscosity."""
        self.set_state(state)
        state.viscosity = self.sol.viscosity
        return state.viscosity

    def get_thermal_conductivity(self, state: FluidState):
        """Compute thermal conductivity."""
        self.set_state(state)
        state.thermal_conductivity = self.sol.thermal_conductivity
        return state.thermal_conductivity

    def get_temperature(self, state: FluidState):
        """Compute temperature of the gas."""
        self.set_state(state)
        state.temperature = self.sol.T
        return state.temperature

    def get_pressure(self, state: FluidState):
        """Compute pressure of the gas."""
        self.set_state(state)
        state.pressure = self.sol.P
        return state.pressure

    def get_sound_speed(self, state: FluidState):
        """Compute speed of sound of the gas."""
        gamma = self.get_gamma(state)
        state.sound_speed = np.sqrt(gamma * self.sol.P / self.sol.density)
        return state.sound_speed

    def get_mass_diffusivity(self, state: FluidState):
        """Compute mixture-averaged diffusion coefficients for use with mass fraction gradients."""
        self.set_state(state)
        return self.sol.mix_diff_coeffs_mass

    def get_enthalpies(self, state: FluidState):
        """Compute partial mass enthalpies of the gas."""
        self.set_state(state)
        return self.sol.partial_molar_enthalpies / self.sol.molecular_weights

    def get_source_terms(self, state: FluidState):
        """Compute reaction source terms corresponding to transported scalars."""
        self.set_state(state)
        return self.sol.net_production_rates

    def get_viscous_flux(
        self,
        face_states: FluidState,
        dudx: Array,
        dTdx: Array,
        dYdx: Array,
    ) -> Array:
        """Compute viscous fluxes."""
        viscosity = self.get_mu(face_states)
        conductivity = self.get_thermal_conductivity(face_states)
        diffusivities = self.get_mass_diffusivity(face_states)
        enthalpies = self.get_enthalpies(face_states)

        # Average the properties from either side
        if self.gas.n_species > 1:
            composition = np.zeros((face_states.shape[1], self.gas.n_species))
            composition[:, :-1] = 0.5 * (
                face_states.composition[0, :, :] + face_states.composition[1, :, :]
            )
            composition[:, -1] = 1.0 - np.sum(composition[:, :-1], axis=1)
        else:
            composition = 0.5 * (
                face_states.composition[0, :, :] + face_states.composition[1, :, :]
            )
        density = 0.5 * (face_states.density[0, :] + face_states.density[1, :])
        viscosity = 0.5 * (viscosity[0, :] + viscosity[1, :])
        conductivity = 0.5 * (conductivity[0, :] + conductivity[1, :])
        diffusivities = 0.5 * (diffusivities[0, :, :] + diffusivities[1, :, :])
        enthalpies = 0.5 * (enthalpies[0, :, :] + enthalpies[1, :, :])

        if self.gas.n_species > 1:
            rhoYV = np.zeros((face_states.shape[1], self.gas.n_species))
            rhoYV[:, :-1] = -density[:, None] * diffusivities[:, :-1] * dYdx
            rhoYV[:, -1] = -density * diffusivities[:, -1] * (-np.sum(dYdx, axis=1))
        else:
            rhoYV = -density[:, None] * diffusivities[:, None] * dYdx
        rhoYVc = -np.sum(rhoYV, axis=1)
        rhoYV += rhoYVc[:, None] * composition

        flux_rho = np.zeros((face_states.shape[1], 1))
        flux_rhou = (4.0 / 3.0 * viscosity * dudx)[:, None]
        flux_rhoY = rhoYV[:, :-1]
        flux_rhoE = (conductivity * dTdx)[:, None] + (rhoYV * enthalpies).sum(axis=-1)[
            :, None
        ]

        return np.concatenate(
            (
                flux_rho,
                flux_rhou,
                flux_rhoE,
                flux_rhoY,
            ),
            axis=1,
        )
