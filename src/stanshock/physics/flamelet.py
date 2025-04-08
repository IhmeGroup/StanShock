from __future__ import annotations

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from stanshock.physics.fluid_base import FluidPhysics, FluidState


class TableVariable:
    """
    Class to represent a variable in an FPV table.
    """

    def __init__(self, name, data, Z, Q, L):
        """
        Initialize the TableVariable object with the name of the variable and the data.
        """
        self.name = name
        self.data = data
        self.interp = RegularGridInterpolator(
            (Z, Q, L), data, bounds_error=False, fill_value=None
        )

    def lookup(self, Z, Q, L):
        """
        Perform a lookup of the variable at the given Z, Q, and L values.
        """
        return self.interp((Z, Q, L))


class FPVTable(FluidPhysics):
    """
    Class to read an FPV table from an HDF5 file and perform lookups.
    """

    def __init__(self, filename, gas):
        """
        Initialize the FPVTable object by reading the HDF5 file.
        """
        super().__init__(gas)
        self.n_scalars = 2
        self.scalar_names = ["mixture fraction", "progress variable"]
        self.normalize_scalars = False
        self.is_flamelet = True
        self.filename = filename
        with h5py.File(filename, "r") as f:
            self.P = f["Header"]["Doubles"]["Double_0"].attrs["Value"][0]
            self.Z = f["Coordinates"]["Coor_0"][()]
            self.Q = f["Coordinates"]["Coor_1"][()]
            self.L = f["Coordinates"]["Coor_2"][()]

            var_names = [
                var.decode("utf-8") for var in f["Header"]["Variable Names"][()]
            ]
            data_raw = f["Data"][()]
            n_tot = self.Z.size * self.Q.size * self.L.size
            self.variables = []
            for i, var in enumerate(var_names):
                data = data_raw[i * n_tot : (i + 1) * n_tot].reshape(
                    self.Z.size, self.Q.size, self.L.size, order="C"
                )
                self.variables.append(TableVariable(var, data, self.Z, self.Q, self.L))

    def set_state(self, state: FluidState) -> FluidState:
        """Get the flamelet table coordinates from the composition."""
        Z = state.mixture_fraction = state.composition[:, 0]
        C = state.progress_variable = state.composition[:, 1]
        state.normalized_progress_variable = self.get_normalized_progress_variable(Z, C)
        return state

    def get_composition(self, Y):
        """Converts mass fractions to mixture fraction and progress variable."""
        Z = self.get_bilger_mixture_fraction(Y)
        C = self.get_progress_variable(Y)

        return np.stack([Z, C], axis=1)

    def get_normalized_progress_variable(self, Z, C):
        """
        Compute the normalized progress variable value at the given Z and C values.
        """
        C_min = np.zeros_like(Z)
        for v in self.variables:
            if v.name == "PROG":
                C_max = v.lookup(Z, 0, 1)
        L = (C - C_min) / (C_max - C_min)
        return np.clip(L, 0, 1)

    def lookup(self, var, state: FluidState):
        """
        Perform a lookup of the variable with the given name at the given Z, Q, and L values.
        """
        Z = state.mixture_fraction
        Q = np.zeros_like(Z)
        L = state.normalized_progress_variable

        for v in self.variables:
            if v.name == var:
                return v.lookup(Z, Q, L)

        msg = f"Variable {var} not found in table {self.filename}."
        raise ValueError(msg)

    def lookup_all(self, state: FluidState):
        """
        Perform a lookup of all variables at the given Z, Q, and L values.
        """
        Z = state.mixture_fraction
        Q = np.zeros_like(Z)
        L = state.normalized_progress_variable

        return {v.name: v.lookup(Z, Q, L) for v in self.variables}

    def get_gamma(self, state: FluidState):
        """
        Compute the specific heat ratio at the given Z, Q, L and T values.
        """
        gamma0 = self.lookup("GAMMA0", state)
        ag = self.lookup("AGAMMA", state)
        T0 = self.lookup("T0", state)
        return gamma0 + ag * (state.temperature - T0)

    def get_specific_gas_constant(self, state: FluidState):
        """
        Compute the gas constant at the given Z, Q, and L values.
        """
        return self.lookup("ROM", state)

    def get_cp(self, state: FluidState):
        """
        Compute the specific heat at the given Z, Q, L and T values.
        """
        R = self.get_specific_gas_constant(state)
        gamma = self.get_gamma(state)
        return R * gamma / (gamma - 1)

    def get_cv(self, state: FluidState):
        """
        Compute the specific heat at constant volume at the given Z, Q, and L values.
        """
        R = self.get_specific_gas_constant(state)
        gamma = self.get_gamma(state)
        return R / (gamma - 1)

    def get_mu(self, state: FluidState):
        """
        Compute the dynamic viscosity at the given Z, Q, and L values.
        """
        mu0 = self.lookup("MU0", state)
        T0 = self.lookup("T0", state)
        amu = self.lookup("AMU", state)
        return mu0 * (state.temperature / T0) ** amu

    def get_thermal_conductivity(self, state: FluidState):
        """
        Compute the thermal conductivity at the given Z, Q, and L values.
        """
        loc0 = self.lookup("LOC0", state)
        T0 = self.lookup("T0", state)
        aloc = self.lookup("ALOC", state)
        return loc0 * (state.temperature / T0) ** aloc

    def get_temperature(self, state: FluidState):
        R = self.get_specific_gas_constant(state)
        state.temperature = state.pressure / (R * state.density)
        return state.temperature
