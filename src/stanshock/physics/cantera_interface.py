from __future__ import annotations

import cantera as ct
import numpy as np

from stanshock.physics.fluid_base import FluidPhysics, FluidState


class CanteraInterface(FluidPhysics):
    def __init__(self, gas: ct.Solution):
        super().__init__(gas)
        self._cached_solutions = {}

    @property
    def n_scalars(self):
        return self.gas.n_species

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

        # Update SolutionArray only if state has changed - crude check for now:
        if (
            state.pressure is not self.sol.P
            or state.density is not self.sol.density_mass
        ):
            state.mass_fractions = state.composition

            if state.density is not None:
                if state.pressure is not None:
                    self.sol.DPY = state.density, state.pressure, state.mass_fractions
                elif state.temperature is not None:
                    self.sol.TDY = (
                        state.temperature,
                        state.density,
                        state.mass_fractions,
                    )

            # Update the state variables to point directly to the SolutionArray properties
            state.pressure = self.sol.P
            state.temperature = self.sol.T
            state.density = self.sol.density_mass
            state.composition = state.mass_fractions = self.sol.Y

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

    def get_sound_speed(self, state: FluidState):
        """Compute speed of sound of the gas."""
        gamma = self.get_gamma(state)
        state.sound_speed = np.sqrt(gamma * self.sol.P / self.sol.density)
        return state.sound_speed

    def get_mass_diffusivity(self, state: FluidState):
        """Compute mixture-averaged diffusion coefficients."""
        self.set_state(state)
        return self.sol.mix_diff_coeffs

    def get_source_terms(self, state: FluidState):
        """Compute reaction source terms corresponding to transported scalars."""
        self.set_state(state)
        return self.sol.net_production_rates
