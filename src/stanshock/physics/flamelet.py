from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from cantera import Solution
from scipy.interpolate import RegularGridInterpolator

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array, Composition


class TableVariable:
    """
    Class to represent a variable in an FPV table.
    """

    def __init__(self, name: str, data: Array, Z: Array, Q: Array, L: Array) -> None:
        """
        Initialize the TableVariable object with the name of the variable and the data.
        """
        self.name: str = name
        self.data: Array = data
        self.interp: RegularGridInterpolator[np.float64] = RegularGridInterpolator(
            (Z, Q, L), data, bounds_error=False, fill_value=None
        )

    def lookup(self, Z: Array | float, Q: Array | float, L: Array | float) -> Array:
        """
        Perform a lookup of the variable at the given Z, Q, and L values.
        """
        return self.interp((Z, Q, L))


class FPVTable(FluidPhysics):
    """
    Class to read an FPV table from an HDF5 file and perform lookups.
    """

    def __init__(
        self,
        filename: Path | str,
        gas: Solution,
        ox_def: Composition | None = None,
        fuel_def: Composition | None = None,
        prog_def: Composition | None = None,
        p_correction: bool = False,
        T_correction: bool = False,
    ) -> None:
        """
        Initialize the FPVTable object by reading the HDF5 file.
        """
        self.gas: Solution = gas
        self.n_scalars: int = 3
        self.n_scalars_rho_sum: int = 1
        self.scalar_names = ["density", "mixture fraction", "progress variable"]
        self.is_flamelet = True
        self.filename: str = filename.name if isinstance(filename, Path) else filename
        self.p_correction: bool = p_correction
        self.T_correction: bool = T_correction
        with h5py.File(filename, "r") as f:
            self.P: float = f["Header"]["Doubles"]["Double_0"].attrs["Value"][0]
            self.Z: Array = f["Coordinates"]["Coor_0"][:]
            self.Q: Array = f["Coordinates"]["Coor_1"][:]
            self.L: Array = f["Coordinates"]["Coor_2"][:]

            var_names: list[str] = [
                var.decode("utf-8") for var in f["Header"]["Variable Names"][:]
            ]
            data_raw: Array = f["Data"][:]
            n_tot = self.Z.size * self.Q.size * self.L.size
            self.variables: list[TableVariable] = []
            for i, var in enumerate(var_names):
                data = data_raw[i * n_tot : (i + 1) * n_tot].reshape(
                    self.Z.size, self.Q.size, self.L.size, order="C"
                )
                self.variables.append(TableVariable(var, data, self.Z, self.Q, self.L))

        self.ox_def: Composition | None = ox_def
        self.fuel_def: Composition | None = fuel_def
        if self.ox_def is None or self.fuel_def is None:
            self.ox_def, self.fuel_def = self.get_fuel_and_oxidizer_definitions()
        self.initialize_bilger_mixture_fraction()

        self.prog_def: Composition | None = prog_def
        if self.prog_def is not None:
            self.initialize_progress_variable(self.prog_def)

    def get_fuel_and_oxidizer_definitions(
        self, cutoff: float = 1e-6
    ) -> tuple[Composition, Composition]:
        """Get the fuel and oxidizer composition from the table."""
        self.ox_def = {}
        self.fuel_def = {}
        sum_ox = 0.0
        sum_fuel = 0.0
        Wk = self.gas.molecular_weights
        for isp, sp_name in enumerate(self.gas.species_names):
            Y = self.lookup_direct(sp_name, 0.0, 0.0, 0.0).item() / Wk[isp]
            if cutoff < Y:
                self.ox_def[sp_name] = Y
                sum_ox += Y

            Y = self.lookup_direct(sp_name, 1.0, 0.0, 0.0).item() / Wk[isp]
            if cutoff < Y:
                self.fuel_def[sp_name] = Y
                sum_fuel += Y

        # Convert the mass fractions to mole fractions
        self.ox_def = {sp_name: Y / sum_ox for sp_name, Y in self.ox_def.items()}
        self.fuel_def = {sp_name: Y / sum_fuel for sp_name, Y in self.fuel_def.items()}

        return self.ox_def, self.fuel_def

    def set_state(self, state: FluidState) -> FluidState:
        """Get the flamelet table coordinates from the composition."""
        assert state.composition is not None
        Z = state.mixture_fraction = state.composition[..., 1]
        C = state.progress_variable = state.composition[..., 2]
        state.normalized_progress_variable = self.get_normalized_progress_variable(Z, C)
        return state

    def get_composition(self, Y: Array) -> Array:
        """Converts mass fractions to mixture fraction and progress variable."""
        Z = self.get_bilger_mixture_fraction(Y)
        C = self.get_progress_variable(Y)

        return np.stack([np.ones_like(Z), Z, C], axis=1)

    def get_normalized_progress_variable(self, Z: Array, C: Array) -> Array:
        """
        Compute the normalized progress variable value at the given Z and C values.
        """
        C_min: Array = np.zeros_like(Z)
        C_max: Array = np.ones_like(Z)
        for v in self.variables:
            if v.name == "PROG":
                C_max = v.lookup(Z, 0, 1)
        L = (C - C_min) / (C_max - C_min)
        return np.clip(L, 0, 1)

    def lookup(self, var: str, state: FluidState) -> Array:
        """
        Perform a lookup of the variable with the given name at the given Z, Q, and L values.
        """
        if state.normalized_progress_variable is None:
            state = self.set_state(state)
        assert state.mixture_fraction is not None
        assert state.normalized_progress_variable is not None

        Z = state.mixture_fraction
        Q = np.zeros_like(Z)
        L = state.normalized_progress_variable

        return self.lookup_direct(var, Z, Q, L)

    def lookup_direct(
        self, var: str, Z: Array | float, Q: Array | float, L: Array | float
    ) -> Array:
        """
        Perform a lookup of the variable with the given name at the given Z, Q, and L values.
        """
        for v in self.variables:
            if v.name == var:
                return v.lookup(Z, Q, L)

        msg = f"Variable {var} not found in table {self.filename}."
        raise ValueError(msg)

    def lookup_all(self, state: FluidState) -> dict[str, Array]:
        """
        Perform a lookup of all variables at the given Z, Q, and L values.
        """
        if state.normalized_progress_variable is None:
            state = self.set_state(state)
        assert state.mixture_fraction is not None
        assert state.normalized_progress_variable is not None

        Z = state.mixture_fraction
        Q = np.zeros_like(Z)
        L = state.normalized_progress_variable

        return {v.name: v.lookup(Z, Q, L) for v in self.variables}

    def get_gamma(self, state: FluidState) -> Array:
        """
        Compute the specific heat ratio at the given Z, Q, L and T values.
        """
        if state.temperature is None:
            state.temperature = self.get_temperature(state)

        gamma0 = self.lookup("GAMMA0", state)
        ag = self.lookup("AGAMMA", state)
        T0 = self.lookup("T0", state)
        state.gamma = gamma0 + ag * (state.temperature - T0)
        return state.gamma

    def get_specific_gas_constant(self, state: FluidState) -> Array:
        """
        Compute the gas constant at the given Z, Q, and L values.
        """
        return self.lookup("ROM", state)

    def get_cp(self, state: FluidState) -> Array:
        """
        Compute the specific heat at the given Z, Q, L and T values.
        """
        R = self.get_specific_gas_constant(state)
        gamma = self.get_gamma(state)
        state.cp = R * gamma / (gamma - 1)
        return state.cp

    def get_cv(self, state: FluidState) -> Array:
        """
        Compute the specific heat at constant volume at the given Z, Q, and L values.
        """
        R = self.get_specific_gas_constant(state)
        gamma = self.get_gamma(state)
        return R / (gamma - 1)

    def get_mu(self, state: FluidState) -> Array:
        """
        Compute the dynamic viscosity at the given Z, Q, and L values.
        """
        if state.temperature is None:
            state.temperature = self.get_temperature(state)
        assert state.temperature is not None
        mu0 = self.lookup("MU0", state)
        T0 = self.lookup("T0", state)
        amu = self.lookup("AMU", state)
        state.viscosity = mu0 * (state.temperature / T0) ** amu
        return state.viscosity

    def get_thermal_conductivity(self, state: FluidState) -> Array:
        """
        Compute the thermal conductivity at the given Z, Q, and L values.
        """
        assert state.temperature is not None
        if state.cp is None:
            state.cp = self.get_cp(state)

        loc0 = self.lookup("LOC0", state)
        T0 = self.lookup("T0", state)
        aloc = self.lookup("ALOC", state)
        state.thermal_conductivity = state.cp * loc0 * (state.temperature / T0) ** aloc
        return state.thermal_conductivity

    def get_temperature(self, state: FluidState) -> Array:
        """
        Compute the temperature at the given Z, Q, L and e values.
        Note: Using sensible energy instead of internal energy because
        StanShock transports the total non-chemical energy.
        """
        R = self.get_specific_gas_constant(state)
        if state.pressure is not None:
            assert state.density is not None
            state.temperature = state.pressure / (state.density * R)
        else:
            assert state.internal_energy is not None
            T0 = self.lookup("T0", state)
            e0 = self.lookup("E0", state)
            gamma0 = self.lookup("GAMMA0", state)
            ag = self.lookup("AGAMMA", state)
            state.temperature = T0 + ((gamma0 - 1) / ag) * (
                np.exp(ag * (state.internal_energy - e0) / R) - 1
            )
        return state.temperature

    def get_pressure(self, state: FluidState) -> Array:
        if state.pressure is None:
            assert state.density is not None
            assert state.temperature is not None
            R = self.get_specific_gas_constant(state)
            state.pressure = state.temperature * R * state.density
        return state.pressure

    def get_internal_energy(self, state: FluidState) -> Array:
        if state.e0_star is not None:
            assert state.gamma_star is not None
            assert state.density is not None
            assert state.pressure is not None
            state.internal_energy = (
                state.pressure / (state.density * (state.gamma_star - 1.0))
                + state.e0_star
            )
        else:
            T = state.temperature
            if T is None:
                if state.pressure is None:
                    msg = (
                        "Fluid state not fully defined. "
                        "Temperature, pressure, and internal energy are not set."
                    )
                    raise ValueError(msg)
                T = self.get_temperature(state)

            R = self.get_specific_gas_constant(state)
            T0 = self.lookup("T0", state)
            e0 = self.lookup("E0", state)
            gamma0 = self.lookup("GAMMA0", state)
            ag = self.lookup("AGAMMA", state)

            state.internal_energy = e0 + R / ag * np.log(
                1.0 + ag * (T - T0) / (gamma0 - 1.0)
            )

        return state.internal_energy

    def get_species_enthalpies(self, state: FluidState) -> Array:
        state = self.set_state(state)
        return np.zeros(state.shape)

    def get_sound_speed(self, state: FluidState) -> Array:
        assert state.density is not None
        if state.gamma is None:
            state.gamma = self.get_gamma(state)
        if state.pressure is None:
            state.pressure = self.get_pressure(state)
        state.sound_speed = np.sqrt(state.gamma * state.pressure / state.density)
        return state.sound_speed

    def get_source_terms(self, state: FluidState) -> Array:
        return self.lookup("SRC_PROG", state)

    def get_source_progress_variable_compressibility_factor(
        self, state: FluidState
    ) -> Array:
        """
        Compute the scaling factor for the progress variable source term at the given Z, Q, L, p and T values.
        """
        factor = np.ones(state.shape)
        if self.p_correction:
            assert state.pressure is not None
            factor *= state.pressure / self.P
        if self.T_correction:
            if state.temperature is None:
                state.temperature = self.get_temperature(state)
            TA = self.lookup("TA", state)
            T0 = self.lookup("T0", state)
            factor *= np.exp(-TA * ((1 / state.temperature) - (1 / T0)))
        return factor
