from __future__ import annotations

import functools
import warnings
from pathlib import Path

import h5py
import numpy as np
from cantera import Solution
from joblib import Parallel, delayed
from scipy import special, stats
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array, Composition
from stanshock.utils.h5 import h5_getarray


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

    def __new__(
        cls,
        filename: Path | str,
        gas: Solution,
        ox_def: Composition | None = None,
        fuel_def: Composition | None = None,
        prog_def: Composition | None = None,
        p_correction: bool = False,
        T_correction: bool = False,
    ) -> FPVTable:
        """Apply PDF convolution if the table lacks a Q dimension."""
        table = super().__new__(cls)
        cls.__init__(
            table,
            filename,
            gas,
            ox_def,
            fuel_def,
            prog_def,
            p_correction,
            T_correction,
        )
        if len(table.Q) <= 2:
            return apply_pdf_convolution(table)
        return table

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
        self.filename: Path = Path(filename)
        self.p_correction: bool = p_correction
        self.T_correction: bool = T_correction
        with h5py.File(str(self.filename), "r") as f:
            doubles = f["Header/Doubles/Double_0"]
            assert isinstance(doubles, h5py.Dataset)
            Pref = doubles.attrs.get("Value")
            assert isinstance(Pref, np.ndarray)
            self.P: float = float(Pref[0])
            self.Z: Array = h5_getarray(f, "Coordinates/Coor_0")
            self.Q: Array = h5_getarray(f, "Coordinates/Coor_1")
            self.L: Array = h5_getarray(f, "Coordinates/Coor_2")

            raw_var_names = f["Header/Variable Names"]
            assert isinstance(raw_var_names, h5py.Dataset)
            var_names: list[str] = [var.decode("utf-8") for var in raw_var_names[:]]
            data_raw: Array = h5_getarray(f, "Data")
            n_tot = self.Z.size * self.Q.size * self.L.size
            self.variables: dict[str, TableVariable] = {}
            for i, var in enumerate(var_names):
                data = data_raw[i * n_tot : (i + 1) * n_tot].reshape(
                    self.Z.size, self.Q.size, self.L.size, order="C"
                )
                self.variables[var] = TableVariable(var, data, self.Z, self.Q, self.L)

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
        if state.mixture_fraction_variance is None:
            state.mixture_fraction_variance = np.zeros_like(Z)
        return state

    def get_composition_from_mass_fractions(self, Y: Array) -> Array:
        """Converts mass fractions to mixture fraction and progress variable."""
        Z = self.get_bilger_mixture_fraction(Y)
        C = self.get_progress_variable(Y)

        return np.stack([np.ones_like(Z), Z, C], axis=1)

    def get_mass_fractions(self, state: FluidState) -> Array:
        """Returns mass fractions from transported scalars."""
        if state.normalized_progress_variable is None:
            state = self.set_state(state)
        assert state.mixture_fraction is not None
        assert state.mixture_fraction_variance is not None
        assert state.normalized_progress_variable is not None

        Z = state.mixture_fraction
        Q = state.mixture_fraction_variance
        L = state.normalized_progress_variable

        nsp = self.gas.n_species
        Y = np.zeros((*state.shape, nsp))
        for isp in range(nsp):
            sp = self.gas.species_names[isp]
            Y[..., isp] = self.lookup_direct(sp, Z, Q, L)

        return Y

    def get_normalized_progress_variable(self, Z: Array, C: Array) -> Array:
        """
        Compute the normalized progress variable value at the given Z and C values.
        """
        C_min: Array = np.zeros_like(Z)
        C_max: Array = np.ones_like(Z)
        C_max = self.variables["PROG"].lookup(Z, 0, 1)
        L = (C - C_min) / (C_max - C_min)
        return np.clip(L, 0, 1)

    def lookup(self, var: str, state: FluidState) -> Array:
        """
        Perform a lookup of the variable with the given name at the given Z, Q, and L values.
        """
        if state.normalized_progress_variable is None:
            state = self.set_state(state)
        assert state.mixture_fraction is not None
        assert state.mixture_fraction_variance is not None
        assert state.normalized_progress_variable is not None

        Z = state.mixture_fraction
        Q = state.mixture_fraction_variance
        L = state.normalized_progress_variable

        return self.lookup_direct(var, Z, Q, L)

    def lookup_direct(
        self, var: str, Z: Array | float, Q: Array | float, L: Array | float
    ) -> Array:
        """
        Perform a lookup of the variable with the given name at the given Z, Q, and L values.
        """
        if var not in self.variables:
            msg = f"Variable {var} not found in table {self.filename}."
            raise ValueError(msg)
        return self.variables[var].lookup(Z, Q, L)

    def lookup_all(self, state: FluidState) -> dict[str, Array]:
        """
        Perform a lookup of all variables at the given Z, Q, and L values.
        """
        if state.normalized_progress_variable is None:
            state = self.set_state(state)
        assert state.mixture_fraction is not None
        assert state.mixture_fraction_variance is not None
        assert state.normalized_progress_variable is not None

        Z = state.mixture_fraction
        Q = state.mixture_fraction_variance
        L = state.normalized_progress_variable

        return {var: v.lookup(Z, Q, L) for var, v in self.variables.items()}

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
        if state.temperature is not None:
            return state.temperature

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

    def get_density(self, state: FluidState) -> Array:
        if state.density is None:
            assert state.pressure is not None
            assert state.temperature is not None
            R = self.get_specific_gas_constant(state)
            state.density = state.pressure / (state.temperature * R)
        return state.density

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
        if state.density is None:
            state.density = self.get_density(state)
        if state.gamma is None:
            state.gamma = self.get_gamma(state)
        if state.pressure is None:
            state.pressure = self.get_pressure(state)
        state.sound_speed = np.sqrt(state.gamma * state.pressure / state.density)
        return state.sound_speed

    def get_source_terms(self, state: FluidState) -> Array:
        factor = self.get_source_progress_variable_compressibility_factor(state)
        return factor * self.lookup("SRC_PROG", state)

    def get_source_progress_variable_compressibility_factor(
        self, state: FluidState
    ) -> Array:
        """
        Compute the scaling factor for the progress variable source term at the given Z, Q, L, p and T values.
        """
        factor = np.ones(state.shape)
        if self.p_correction:
            P = self.get_pressure(state)
            factor *= P / self.P
        if self.T_correction:
            T = self.get_temperature(state)
            TA = self.lookup("TA", state)
            T0 = self.lookup("T0", state)
            factor *= np.exp(-TA * (1.0 / T - 1.0 / T0))
        return factor


def _convolve_point(
    i_Zbar: int,
    i_S: int,
    i_Lbar: int,
    Zbar_vec: Array,
    Qbar_vec: Array,
    Lbar_vec: Array,
    uv_vec: Array,
    W_vec: Array,
    variables: dict[str, TableVariable],
) -> tuple[int, int, int, Array]:
    Zbar = Zbar_vec[i_Zbar]
    sigma2 = Qbar_vec[i_S]
    Lbar = Lbar_vec[i_Lbar]

    # Bound valid values of sigma2
    Zvmax = Zbar * (1.0 - Zbar)
    sigma2 = min(Zvmax, sigma2)

    eps = 1.0e-6
    if (Zbar < eps) or (Zbar > 1 - eps) or (sigma2 < eps):
        result = np.array([v.lookup(Zbar, 0.0, Lbar) for v in variables.values()])
        return i_Zbar, i_S, i_Lbar, result

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Compute the shape parameters
        alpha_Z = ((Zbar * (1 - Zbar) / sigma2) - 1) * Zbar
        beta_Z = alpha_Z * (1 - Zbar) / Zbar
        alpha_L = ((Lbar * (1 - Lbar) / sigma2) - 1) * Lbar
        beta_L = alpha_L * (1 - Lbar) / Lbar

        # Compute the quadrature points and weights
        Z_int_vec = stats.beta.ppf(uv_vec, alpha_Z, beta_Z)
        L_int_vec = stats.beta.ppf(uv_vec, alpha_L, beta_L)

        # Evaluate the integrand at the quadrature points
        W_Z_mesh, W_L_mesh = np.meshgrid(W_vec, W_vec, indexing="ij")
        W_mesh = W_Z_mesh * W_L_mesh
        Z_int_mesh, L_int_mesh = np.meshgrid(Z_int_vec, L_int_vec, indexing="ij")
        integrand = np.stack(
            [v.lookup(Z_int_mesh, 0.0, L_int_mesh) for v in variables.values()]
        )

        # Perform the integration
        result = np.sum(integrand * W_mesh, axis=(1, 2))

    return i_Zbar, i_S, i_Lbar, result


def apply_pdf_convolution(table: FPVTable, keep_backup: bool = True) -> FPVTable:
    """Convolve a laminar flamelet table with a beta PDF to model turbulent mixing."""
    print("Convolving flamelet table with beta PDF...")

    # Grid in Zbar, Lbar, logsigma2 dimensions (to be tabulated over)
    n_tab = (table.Z.size, 31, table.L.size)
    Zbar_vec = np.linspace(0.0, 1.0, n_tab[0])
    Qbar_vec = np.linspace(0.0, 0.5, n_tab[1]) ** 2
    Lbar_vec = np.linspace(0.0, 1.0, n_tab[2])

    uv_vec, W_vec = special.roots_legendre(200)
    W_vec = 0.5 * W_vec
    uv_vec = 0.5 * (uv_vec + 1.0)

    compute_func = functools.partial(
        _convolve_point,
        Zbar_vec=Zbar_vec,
        Qbar_vec=Qbar_vec,
        Lbar_vec=Lbar_vec,
        uv_vec=uv_vec,
        W_vec=W_vec,
        variables=table.variables,
    )
    tasks = [
        (i_Zbar, i_S, i_Lbar)
        for i_Zbar in range(n_tab[0])
        for i_S in range(n_tab[1])
        for i_Lbar in range(n_tab[2])
    ]

    # Parallel version
    gen = Parallel(n_jobs=-1, return_as="generator")(
        delayed(compute_func)(i_Zbar, i_S, i_Lbar) for i_Zbar, i_S, i_Lbar in tasks
    )

    data = np.zeros((len(table.variables), *n_tab))
    for i_Zbar, i_S, i_Lbar, value in tqdm(gen, total=data[0].size):
        data[:, i_Zbar, i_S, i_Lbar] = value
    data[np.isnan(data)] = 0.0

    # Write data to the original file
    filename = table.filename
    if keep_backup:
        backup_file = filename.with_stem(filename.stem + "_orig")
        backup_file.unlink(missing_ok=True)
        filename.copy(backup_file)

    with h5py.File(str(filename), "r+") as f:
        coords = f["Coordinates"]
        assert isinstance(coords, h5py.Group)
        del coords["Coor_1"]
        coords.create_dataset("Coor_1", data=Qbar_vec)

        del f["Data"]
        f.create_dataset("Data", data=data.flatten())

    return FPVTable(
        filename,
        table.gas,
        table.ox_def,
        table.fuel_def,
        table.prog_def,
        table.p_correction,
        table.T_correction,
    )
