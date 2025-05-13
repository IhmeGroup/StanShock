from __future__ import annotations

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide


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

    def __init__(self, filename, gas, ox_def=None, fuel_def=None, prog_def=None):
        """
        Initialize the FPVTable object by reading the HDF5 file.
        """
        self.gas = gas
        self.n_scalars = 2
        self.scalar_names = ["mixture fraction", "progress variable"]
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

        self.ox_def = ox_def
        self.fuel_def = fuel_def
        self.prog_def = prog_def
        if self.ox_def is None or self.fuel_def is None:
            self.get_fuel_and_oxidizer_definitions()
        self.initialize_bilger_mixture_fraction()

        if self.prog_def is not None:
            self.initialize_progress_variable(self.prog_def)

    def get_fuel_and_oxidizer_definitions(self, cutoff=1e-6):
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
        if state.normalized_progress_variable is None:
            self.set_state(state)

        Z = state.mixture_fraction
        Q = np.zeros_like(Z)
        L = state.normalized_progress_variable

        return self.lookup_direct(var, Z, Q, L)

    def lookup_direct(self, var, Z, Q, L):
        """
        Perform a lookup of the variable with the given name at the given Z, Q, and L values.
        """
        for v in self.variables:
            if v.name == var:
                return v.lookup(Z, Q, L)

        msg = f"Variable {var} not found in table {self.filename}."
        raise ValueError(msg)

    def lookup_all(self, state: FluidState):
        """
        Perform a lookup of all variables at the given Z, Q, and L values.
        """
        if state.normalized_progress_variable is None:
            self.set_state(state)

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
        state.gamma = gamma0 + ag * (state.temperature - T0)
        return state.gamma

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
        state.cp = R * gamma / (gamma - 1)
        return state.cp

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
        state.viscosity = mu0 * (state.temperature / T0) ** amu
        return state.viscosity

    def get_thermal_conductivity(self, state: FluidState):
        """
        Compute the thermal conductivity at the given Z, Q, and L values.
        """
        if state.cp is None:
            self.get_cp(state)

        loc0 = self.lookup("LOC0", state)
        T0 = self.lookup("T0", state)
        aloc = self.lookup("ALOC", state)
        state.thermal_conductivity = state.cp * loc0 * (state.temperature / T0) ** aloc
        return state.thermal_conductivity

    def get_temperature(self, state: FluidState):
        R = self.get_specific_gas_constant(state)
        state.temperature = state.pressure / (R * state.density)
        return state.temperature

    def get_pressure(self, state: FluidState):
        R = self.get_specific_gas_constant(state)
        state.pressure = state.temperature * R * state.density
        return state.pressure

    def get_sound_speed(self, state):
        if state.gamma is None:
            self.get_gamma(state)
        if state.pressure is None:
            self.get_pressure(state)
        state.sound_speed = np.sqrt(state.gamma * state.pressure / state.density)
        return state.sound_speed

    def get_source_terms(self, state):
        return self.lookup("SRC_PROG", state)

    def primitive_to_conservative(self, state: FluidState):
        """Transform primitive variables into vector of conservatives."""
        # Get conservative vector with sensible + kinetic energy
        y = super().primitive_to_conservative(state)

        # Add chemical energy from flamelet table
        y[..., 2] += state.density * self.lookup("E0_CHEM", state)

        return y

    def conservative_to_primitive(self, state_array: Array, gamma: Array) -> FluidState:
        """Transform conservative variables into primitives."""
        r = state_array[..., 0]
        ru = state_array[..., 1]
        E = state_array[..., 2]
        rY = state_array[..., 3:]

        # Get the composition first
        Y = rY / r[..., None]

        # Bound
        Y[Y > 1.0] = 1.0
        Y[Y < 0.0] = 0.0

        # Subtract chemical energy from flamelet table
        Z = Y[..., 0]
        C = Y[..., 1]
        Q = np.zeros_like(Z)
        L = self.get_normalized_progress_variable(Z, C)
        E -= r * self.lookup_direct("E0_CHEM", Z, Q, L)

        # Now get velocity and pressure
        u = ru / r
        p = (gamma - 1.0) * (E - 0.5 * r * u**2.0)
        # TODO - update this to use new energy equation

        return FluidState(
            shape=r.shape,
            density=r,
            velocity=u,
            pressure=p,
            composition=Y,
            mixture_fraction=Z,
            progress_variable=C,
            normalized_progress_variable=L,
        )

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

        # Average the properties from either side
        density = 0.5 * (face_states.density[0, :] + face_states.density[1, :])
        viscosity = 0.5 * (viscosity[0, :] + viscosity[1, :])
        conductivity = 0.5 * (conductivity[0, :] + conductivity[1, :])
        diffusivities = 0.5 * (diffusivities[0, :, :] + diffusivities[1, :, :])

        return np.concatenate(
            (
                np.zeros((face_states.shape[1], 1)),
                (4.0 / 3.0 * viscosity * dudx)[:, None],
                (conductivity * dTdx)[:, None],
                density[:, None] * diffusivities * dYdx,
            ),
            axis=1,
        )


class FPVSource(RightHandSide):
    def source(
        self, _time: float, state_array: Array, physics: FPVTable, gamma_star: Array
    ) -> Array:
        state = physics.conservative_to_primitive(state_array, gamma_star)

        return state.density * physics.get_source_terms(state)
