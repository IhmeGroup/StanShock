from __future__ import annotations

import cantera as ct
import numpy as np

from stanshock.physics.fluid_base import FluidPhysics, FluidState


class CanteraInterface(FluidPhysics):
    def __init__(self, gas: ct.Solution):
        super().__init__(gas)
        self._cached_solutions: dict[int, ct.SolutionArray] = {}
        self._cache_valid: dict[int, bool] = {}

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
        """Compute mixture-averaged diffusion coefficients."""
        self.set_state(state)
        return self.sol.mix_diff_coeffs

    def get_source_terms(self, state: FluidState):
        """Compute reaction source terms corresponding to transported scalars."""
        self.set_state(state)
        return self.sol.net_production_rates
