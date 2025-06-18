from __future__ import annotations

import numpy as np
from cantera import Solution, SolutionArray, gas_constant

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array, Composition, Index, Unpack
from stanshock.system.base import PrecomputeStepName, PrecomputeSteps, RightHandSide


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


class ConstantVolumeChemistry(RightHandSide):
    PRECOMPUTE_STEPS: tuple[PrecomputeStepName, ...] = ("geometry", "physics")

    def __init__(
        self,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        super().__init__(**precompute_steps)
        assert self.physics is not None
        self.idx_source: Index = np.arange(1, self.physics.n_scalars + 2)

        # Define some parameters to be set during precompute step
        self.density0: Array = np.zeros((0,))
        self.shape_initial: tuple[int, ...] = (0, 0)
        self.shape_subset: tuple[int, ...] = (0, 0)

    def precompute(
        self,
        time: float,
        state_array: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> tuple[
        Array,
        FluidState | None,
        FluidState | None,
        FluidState | None,
        FluidState | None,
    ]:
        """Store temperature and mass fractions in the state array."""
        _ = time
        assert self.physics is not None
        n_species = self.physics.n_scalars
        state_array = state_array.reshape((-1, n_species + 2))
        self.shape_initial = state_array.shape

        state_array = state_array[self.idx_domain]
        self.shape_subset = state_array.shape

        state: FluidState = self.physics.conservative_to_primitive(
            state_array, gamma_star, e0_star
        )

        assert state.density is not None
        self.density0 = state.density

        state_array_new: Array = np.zeros((self.shape_subset[0], n_species + 1))
        state_array_new[:, :-1] = state.composition
        state_array_new[:, -1] = self.physics.get_temperature(state)

        return state_array_new, state, None, None, None

    def source(
        self,
        time: float,
        state_array: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        _ = time, gamma_star, e0_star
        assert self.physics is not None

        temperature = state_array[:, -1]

        state: FluidState = FluidState(
            shape=(self.shape_subset[0],),
            density=self.density0,
            temperature=temperature,
            composition=state_array[:, :-1],
        )
        wdot: Array = self.physics.get_source_terms(state)

        eRT = self.physics.sol.standard_int_energies_RT
        cv = self.physics.get_cp(state) / self.physics.get_gamma(state)

        dydt: Array = np.zeros(self.shape_subset)
        dydt[:, :-1] = (
            wdot * self.physics.gas.molecular_weights[None, :] / self.density0[:, None]
        )
        dydt[:, -1] = (
            -np.sum(eRT * wdot, axis=-1, keepdims=True)
            * (gas_constant * temperature / (self.density0 * cv))[:, None]
        )

        return np.ravel(dydt)

    def postcompute(self, state_array: Array, y: Array, state: FluidState) -> Array:
        # Compute state from initial density and updated mass fractions + temperature
        assert self.physics is not None
        state._cache_valid = False
        state.pressure = None
        state.composition = y[:, :-1]
        state.temperature = y[:, -1]
        state.pressure = self.physics.get_pressure(state)

        y = self.physics.primitive_to_conservative(state)

        # Update the state array
        state_array = state_array.reshape(self.shape_initial)
        state_array[self.idx_domain, self.idx_source] = y[:, self.idx_source]

        return np.ravel(state_array)
