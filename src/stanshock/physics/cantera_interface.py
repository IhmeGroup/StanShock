from __future__ import annotations

import numpy as np
from cantera import Solution, SolutionArray

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array, Composition


class CanteraInterface(FluidPhysics):
    def __init__(
        self,
        gas: Solution,
        ox_def: Composition | None = None,
        fuel_def: Composition | None = None,
        prog_def: Composition | None = None,
    ) -> None:
        super().__init__(gas, ox_def, fuel_def, prog_def)
        self._cached_solutions: dict[tuple[int, ...], SolutionArray[Solution]] = {}
        self.sol: SolutionArray[Solution]

    def set_state(self, state: FluidState) -> FluidState:
        """Sets up a Cantera SolutionArray with the current fluid state.

        Caches SolutionArray objects of a given size for reuse, and checks if
        the state has been updated since the last call.

        Ensures the density, pressure, temperature, and internal energy are
        all initialized.
        """
        if state.shape not in self._cached_solutions:
            self._cached_solutions[state.shape] = SolutionArray(
                self.gas, shape=state.shape
            )

        self.sol = self._cached_solutions[state.shape]

        # Update SolutionArray only if state has changed
        if not state._cache_valid:
            state.mass_fractions = state.composition
            assert state.mass_fractions is not None

            # Prioritize density-based updates
            if state.density is not None:
                if state.pressure is not None:
                    self.sol.DPY = state.density, state.pressure, state.mass_fractions
                elif state.internal_energy is not None:
                    self.sol.UVY = (
                        state.internal_energy,
                        1 / state.density,
                        state.mass_fractions,
                    )
                elif state.temperature is not None:
                    self.sol.TDY = (
                        state.temperature,
                        state.density,
                        state.mass_fractions,
                    )
                else:
                    msg = "Cannot set state: need either pressure, internal energy, or temperature."
                    raise ValueError(msg)
                state.density = self.sol.density_mass
            elif (state.pressure is not None) and (state.temperature is not None):
                self.sol.TPY = state.temperature, state.pressure, state.mass_fractions
                state.temperature = self.sol.T
                state.density = self.sol.density_mass

        state._cache_valid = True

        return state

    def get_cp(self, state: FluidState) -> Array:
        """Compute specific heat capacity at constant pressure."""
        state = self.set_state(state)
        state.cp = self.sol.cp_mass
        return state.cp

    def get_gamma(self, state: FluidState) -> Array:
        """Compute specific heat ratio, gamma."""
        state = self.set_state(state)
        state.gamma = self.sol.cp / self.sol.cv
        return state.gamma

    def get_mu(self, state: FluidState) -> Array:
        """Compute dynamic viscosity."""
        state = self.set_state(state)
        state.viscosity = self.sol.viscosity
        return state.viscosity

    def get_thermal_conductivity(self, state: FluidState) -> Array:
        """Compute thermal conductivity."""
        state = self.set_state(state)
        state.thermal_conductivity = self.sol.thermal_conductivity
        return state.thermal_conductivity

    def get_temperature(self, state: FluidState) -> Array:
        """Compute temperature of the gas."""
        if state.temperature is None:
            state = self.set_state(state)
            state.temperature = self.sol.T
        return state.temperature

    def get_pressure(self, state: FluidState) -> Array:
        """Compute pressure of the gas."""
        if state.pressure is None:
            state = self.set_state(state)
            state.pressure = self.sol.P
        return state.pressure

    def get_internal_energy(self, state: FluidState) -> Array:
        """Compute internal energy of the gas."""
        if state.e0_star is not None:
            assert state.pressure is not None
            assert state.density is not None
            assert state.gamma_star is not None
            state.internal_energy = (
                state.pressure / (state.density * (state.gamma_star - 1.0))
                + state.e0_star
            )
        else:
            state = self.set_state(state)
            state.internal_energy = self.sol.int_energy_mass

        return state.internal_energy

    def get_species_enthalpies(self, state: FluidState) -> Array:
        state = self.set_state(state)
        return self.sol.partial_molar_enthalpies / self.gas.molecular_weights

    def get_sound_speed(self, state: FluidState) -> Array:
        """Compute speed of sound of the gas."""
        gamma = state.gamma
        if gamma is None:
            gamma = self.get_gamma(state)
        assert state.pressure is not None
        assert state.density is not None
        state.sound_speed = np.sqrt(gamma * state.pressure / state.density)
        return state.sound_speed

    def get_mass_diffusivity(self, state: FluidState) -> Array:
        """Compute mixture-averaged diffusion coefficients."""
        state = self.set_state(state)
        return self.sol.mix_diff_coeffs_mass

    def get_source_terms(self, state: FluidState) -> Array:
        """Compute reaction source terms corresponding to transported scalars."""
        state = self.set_state(state)
        return self.sol.net_production_rates
