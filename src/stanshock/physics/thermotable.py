from __future__ import annotations

import cantera as ct
import numpy as np
from numba import double, njit

from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Composition, Index

# Type signatures for numba
double1D = double[:]
double2D = double[:, :]
double3D = double[:, :, :]


@njit(double1D(double2D, double1D))
def get_specific_gas_constant_compiled(Y: Array, molecular_weights: Array) -> Array:
    """
    Function used by the thermoTable class to find the gas constant. This
    function is compiled for speed-up.
        inputs:
            Y: scalar [nX,nSp]
            molecular_weights: species molecular weights [nSp]
        output:
            R: gas constants [nX]
    """
    # find dimensions
    nX = len(Y[:, 0])
    nSp = len(Y[0, :])
    # determine R
    R = np.zeros(nX)
    for iX in range(nX):
        molecularWeight: float = 0.0
        for iSp in range(nSp):
            molecularWeight += Y[iX, iSp] / molecular_weights[iSp]
        molecularWeight = 1.0 / molecularWeight
        R[iX] = ct.gas_constant / molecularWeight
    return R


@njit(double1D(double1D, double2D, double1D, double2D, double2D))
def get_cp_compiled(T: Array, Y: Array, TTable: Array, a: Array, b: Array) -> Array:
    """
    Function used by the thermoTable class to find the constant pressure
    specific heats. This function is compiled for speed-up.
        inputs:
            T: Temperatures [nX]
            Y: scalar [nX,nSp]
            TTable: table of temperatures [nT]
            a: first order coefficient for cp [nT]
            b: zeroth order coefficient for cp [nT]
        output:
            cp: constant pressure specific heat ratios [nX]
    """
    # find dimensions
    nX = len(Y[:, 0])
    nSp = len(Y[0, :])
    # find table extremes
    TMin = TTable[0]
    dT = TTable[1] - TTable[0]  # assume constant steps in table
    TMax = TTable[-1] + dT
    # determine the indices
    indices = np.zeros(nX, dtype=np.int64)
    for iX in range(nX):
        indices[iX] = int((T[iX] - TMin) / dT)
    # determine cp
    cp = np.zeros((nX,))
    for iX in range(nX):
        if (T[iX] < 0.99 * TMin) or (T[iX] > 1.01 * TMax):
            msg = f"Temperature out of bounds: {T[iX]} not in range [{TMin}, {TMax}]"
            raise ValueError(msg)
        index = indices[iX]
        bbar = 0.0
        for iSp in range(nSp):
            bbar += Y[iX, iSp] * (a[index, iSp] * T[iX] + b[index, iSp])
        cp[iX] = bbar
    return cp


class ThermoTable(CanteraInterface):
    """
    This is a class defined to encapsulate the temperature table with the
    relevant methods
    """

    def __init__(
        self,
        gas: ct.Solution,
        ox_def: Composition | None = None,
        fuel_def: Composition | None = None,
        prog_def: Composition | None = None,
    ) -> None:
        """
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat
        coefficients. The coefficients are selected to retain the exact
        enthalpies at the table points.
        """
        super().__init__(gas, ox_def, fuel_def, prog_def)
        nsp: int = gas.n_species
        self.TMin: float = gas.min_temp
        nT = 101
        self.TMax: float = gas.max_temp
        # vector of temperatures assuming thermal equilibrium between species
        self.T: Array = np.linspace(self.TMin, self.TMax, nT, dtype=np.float64)
        self.dT = (self.TMax - self.TMin) / (nT - 1)
        # matrix of species enthalpies per temperature
        self.h: Array = np.zeros((nT, nsp))
        # cpk = ak*T+bk for T in [Tk,Tk+1], k in {0,1,2,...,nT-1}
        # matrix of species first order coefficients
        self.a: Array = np.zeros((nT, nsp))
        # matrix of species zeroth order coefficients
        self.b: Array = np.zeros((nT, nsp))
        # transport property curve fit coefficients
        polynomial_order: int = 4
        self.viscosity_coeffs: Array = np.zeros((polynomial_order + 1, nsp))
        self.conductivity_coeffs: Array = np.zeros((polynomial_order + 1, nsp))
        self.binary_diff_coeffs: Array = np.zeros((polynomial_order + 1, nsp, nsp))
        self.polynomial_order = polynomial_order
        self.nsp = nsp

        # determine the coefficients
        for ksp, species in enumerate(gas.species()):
            # initialize with actual cp
            cpk = species.thermo.cp(self.T[0]) / self.molecular_weights[ksp]
            hk = species.thermo.h(self.T[0]) / self.molecular_weights[ksp]
            for kT, Tk in enumerate(self.T):
                # compute next
                Tkp1 = Tk + self.dT
                hkp1 = species.thermo.h(Tkp1) / self.molecular_weights[ksp]
                dh = hkp1 - hk
                # store
                self.h[kT, ksp] = hk
                self.a[kT, ksp] = 2.0 / self.dT * (dh / self.dT - cpk)
                self.b[kT, ksp] = cpk - self.a[kT, ksp] * Tk
                # update
                cpk = self.a[kT, ksp] * (Tkp1) + self.b[kT, ksp]
                hk = hkp1

            # Load transport property coefficients
            self.viscosity_coeffs[:, ksp] = gas.get_viscosity_polynomial(i=ksp)
            self.conductivity_coeffs[:, ksp] = gas.get_thermal_conductivity_polynomial(
                i=ksp
            )
            for jsp in range(nsp):
                self.binary_diff_coeffs[:, ksp, jsp] = (
                    gas.get_binary_diff_coeffs_polynomial(i=ksp, j=jsp)
                )

        # Compute the matching species internal energies
        self.e = (
            self.h - ct.gas_constant * self.T[:, None] / self.molecular_weights[None, :]
        )
        self.c = self.b - ct.gas_constant / self.molecular_weights[None, :]

        # Precompute some constants used by viscosity mixing rule.
        ratio = self.molecular_weights[:, None] / self.molecular_weights[None, :]
        self.W_ratio_4 = ratio**-0.25
        self.W_bottom_sqrt = (1.0 + ratio) ** -0.5

    def get_specific_gas_constant(self, state: FluidState) -> Array:
        """
        This method computes the mixture-specific gas constat
            inputs:
                Y: matrix of mass fractions [n,nSp]
            outputs:
                R: vector of mixture-specific gas constants [n]
        """
        Y = self.get_mass_fractions(state)
        return get_specific_gas_constant_compiled(
            Y.reshape((-1, self.n_scalars)), self.molecular_weights
        ).reshape(state.shape)

    def get_cp(self, state: FluidState) -> Array:
        """
        This method computes the constant pressure specific heat as determined
        by Billet and Abgrall (2003) for the double flux method.
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                cp: vector of constant pressure specific heats
        """
        if state.cp is None:
            Y = self.get_mass_fractions(state)
            T = self.get_temperature(state)
            state.cp = get_cp_compiled(
                T.flatten(), Y.reshape((-1, self.n_scalars)), self.T, self.a, self.b
            ).reshape(state.shape)
        return state.cp

    def get_gamma(self, state: FluidState) -> Array:
        """
        This method computes the specific heat ratio, gamma.
            inputs:
                state: FluidState object
            outputs:
                gamma: vector of specific heat ratios [n]
        """
        if state.gamma is None:
            cp = self.get_cp(state)
            R = self.get_specific_gas_constant(state)
            state.gamma = cp / (cp - R)
        return state.gamma

    def get_logT_poly(self, state: FluidState) -> Array:
        if state.logT_poly is None:
            logT = np.log(self.get_temperature(state))
            state.logT_poly = np.stack(
                [logT**i for i in range(self.polynomial_order + 1)], axis=-1
            )
        return state.logT_poly

    def get_mu(self, state: FluidState) -> Array:
        if state.viscosity is None:
            T = self.get_temperature(state)[:, None]
            logT_poly = self.get_logT_poly(state)
            X = self.get_mole_fractions(state)

            # Compute species viscosities
            sqrt_muk = T**0.25 * (logT_poly @ self.viscosity_coeffs)
            muk = sqrt_muk**2

            # Compute mixture viscosity using modified Wilke's rule
            top_sqrt = (
                1.0 + sqrt_muk[..., None] / sqrt_muk[..., None, :] * self.W_ratio_4
            )
            phi_kj = top_sqrt * top_sqrt * self.W_bottom_sqrt

            sumi = np.sum(X[..., None, :] * phi_kj, axis=-1)
            sumo = np.sum(X * muk / sumi, axis=-1)

            state.viscosity = sumo * np.sqrt(np.asarray(8.0))
        return state.viscosity

    def get_thermal_conductivity(self, state: FluidState) -> Array:
        if state.thermal_conductivity is None:
            T = self.get_temperature(state)[:, None]
            logT_poly = self.get_logT_poly(state)
            X = self.get_mole_fractions(state)

            # Compute species thermal conductivities
            kappak = np.sqrt(T) * (logT_poly @ self.conductivity_coeffs)

            # Compute mixture thermal conductivity
            state.thermal_conductivity = 0.5 * (
                np.sum(X * kappak, axis=-1) + 1.0 / np.sum(X / kappak, axis=-1)
            )
        return state.thermal_conductivity

    def get_mass_diffusivity(self, state: FluidState) -> Array:
        if state.mass_diffusivity is None:
            T = self.get_temperature(state)[:, None]
            logT_poly = self.get_logT_poly(state)
            P = self.get_pressure(state)[:, None]
            Y = self.get_mass_fractions(state)
            X = self.get_mole_fractions(state)

            # Compute binary diffusion coefficients
            diffkj = (T**1.5 / P)[..., None] * np.einsum(
                "...i,ijk->...jk", logT_poly, self.binary_diff_coeffs
            )

            # Compute mixture-averaged diffusion coefficients
            invdiffkj = 1.0 / diffkj

            idx_diag = np.arange(self.nsp, dtype=int)
            invdiffkk = invdiffkj[..., idx_diag, idx_diag]

            oneminusYk = 1.0 - Y
            sumXjoverDkj = oneminusYk * (
                np.sum(X[..., None, :] * invdiffkj, axis=-1) - X * invdiffkk
            )
            sumYjoverDkj = X * (
                np.sum(Y[..., None, :] * invdiffkj, axis=-1) - Y * invdiffkk
            )

            state.mass_diffusivity = oneminusYk / np.maximum(
                sumXjoverDkj + sumYjoverDkj, np.asarray(1e-30)
            )
        return state.mass_diffusivity

    def get_temperature(self, state: FluidState) -> Array:
        """
        This method applies the ideal gas law to compute the temperature
            inputs:
                state: FluidState object
            outputs:
                T: vector of temperatures
        """
        if state.temperature is not None:
            return state.temperature

        if state.pressure is not None and state.density is not None:
            # Compute temperature using ideal gas law
            R = self.get_specific_gas_constant(state)
            state.temperature = state.pressure / (state.density * R)
        else:
            # Compute temperature from the internal energy
            assert state.internal_energy is not None
            R = self.get_specific_gas_constant(state)
            Y = self.get_mass_fractions(state)
            e_int_ref = Y @ self.e.T

            if not np.all(e_int_ref[:, :-1] <= e_int_ref[:, 1:]):
                msg = "Tabulated internal energy for the mixture is not monotonic in temperature."
                raise ValueError(msg)

            N = state.shape[0]
            index = np.zeros(state.shape[0], dtype=int)
            for i in range(N):
                index[i] = max(
                    np.searchsorted(
                        e_int_ref[i], state.internal_energy[i], side="right"
                    )
                    - 1,
                    0,
                )
            de = state.internal_energy - e_int_ref[np.arange(N, dtype=int), index]

            # Solve quadratic formula - always pick larger real root
            a: Array = np.sum(Y * (0.5 * self.a[index]), axis=-1)
            b: Array = np.sum(Y * self.c[index], axis=-1)
            Tm: Array = self.T[index]
            c = -((a * Tm + b) * Tm + de)

            state.temperature = np.divide(
                np.sqrt(b**2 - 4.0 * a * c) - b,
                2.0 * a,
                out=-c / b,
                where=np.abs(a) > 1e-14,
            )

        return state.temperature

    def get_pressure(self, state: FluidState) -> Array:
        if state.pressure is not None:
            return state.pressure

        assert state.density is not None
        if state.gamma_star is not None and state.internal_energy is not None:
            assert state.e0_star is not None
            state.pressure = (
                (state.gamma_star - 1.0)
                * state.density
                * (state.internal_energy - state.e0_star)
            )
        else:
            assert state.temperature is not None
            R = self.get_specific_gas_constant(state)
            state.pressure = state.temperature * R * state.density
        return state.pressure

    def get_internal_energy(self, state: FluidState) -> Array:
        if state.internal_energy is None:
            T = self.get_temperature(state)
            R = self.get_specific_gas_constant(state)
            h = self.get_enthalpy(state)
            state.internal_energy = h - R * T
        return state.internal_energy

    def get_index(self, T: Array) -> Index:
        return np.asarray((T - self.TMin) / self.dT, dtype=int)

    def get_bbar(self, T: Array) -> tuple[Index, Array]:
        index = self.get_index(T)
        return index, 0.5 * self.a[index] * (T + self.T[index])[:, None] + self.b[index]

    def get_species_enthalpies(self, state: FluidState) -> Array:
        T = self.get_temperature(state)

        if any(np.logical_or(0.99 * self.TMin > T, 1.01 * self.TMax < T)):
            msg = f"Temperature not within table. {T.min() = }, {T.max() = }"
            raise ValueError(msg)

        index, bbar = self.get_bbar(T)
        return self.h[index, :] + bbar * (T - self.T[index])[:, None]

    def get_enthalpy(self, state: FluidState) -> Array:
        """
        This method computes the enthalpy according to Billet and Abgrall (2003).
        This is the enthalpy that is frozen over the time step
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                h0: vector of frozen enthalpies for the mixture [n]
        """
        if state.enthalpy is None:
            Y = self.get_mass_fractions(state)
            hk = self.get_species_enthalpies(state)
            state.enthalpy = np.sum(Y * hk, axis=-1)
        return state.enthalpy

    def get_sound_speed(self, state: FluidState) -> Array:
        if state.sound_speed is None:
            assert state.pressure is not None
            assert state.density is not None
            gamma = self.get_gamma(state)
            state.sound_speed = np.sqrt(gamma * state.pressure / state.density)
        return state.sound_speed
