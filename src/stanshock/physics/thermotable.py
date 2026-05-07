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
def get_specific_gas_constant_compiled(Y: Array, molecularWeights: Array) -> Array:
    """
    Function used by the thermoTable class to find the gas constant. This
    function is compiled for speed-up.
        inputs:
            Y: scalar [nX,nSp]
            molecularWeights: species molecular weights [nSp]
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
            molecularWeight += Y[iX, iSp] / molecularWeights[iSp]
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
        if (T[iX] < TMin) or (T[iX] > TMax):
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
        nSp: int = gas.n_species
        self.TMin: float = gas.min_temp
        nT = 21
        self.TMax: float = gas.max_temp
        # vector of temperatures assuming thermal equilibrium between species
        self.T: Array = np.linspace(self.TMin, self.TMax, nT, dtype=np.float64)
        self.dT = (self.TMax - self.TMin) / (nT - 1)
        # matrix of species enthalpies per temperature
        self.h: Array = np.zeros((nT, nSp))
        # cpk = ak*T+bk for T in [Tk,Tk+1], k in {0,1,2,...,nT-1}
        # matrix of species first order coefficients
        self.a: Array = np.zeros((nT, nSp))
        # matrix of species zeroth order coefficients
        self.b: Array = np.zeros((nT, nSp))
        self.molecularWeights: Array = gas.molecular_weights
        # determine the coefficients
        for kSp, species in enumerate(gas.species()):
            # initialize with actual cp
            cpk = species.thermo.cp(self.T[0]) / self.molecularWeights[kSp]
            hk = species.thermo.h(self.T[0]) / self.molecularWeights[kSp]
            for kT, Tk in enumerate(self.T):
                # compute next
                Tkp1 = Tk + self.dT
                hkp1 = species.thermo.h(Tkp1) / self.molecularWeights[kSp]
                dh = hkp1 - hk
                # store
                self.h[kT, kSp] = hk
                self.a[kT, kSp] = 2.0 / self.dT * (dh / self.dT - cpk)
                self.b[kT, kSp] = cpk - self.a[kT, kSp] * Tk
                # update
                cpk = self.a[kT, kSp] * (Tkp1) + self.b[kT, kSp]
                hk = hkp1
        # Compute the matching species internal energies
        self.e = (
            self.h - ct.gas_constant * self.T[:, None] / self.molecularWeights[None, :]
        )
        self.c = self.b - ct.gas_constant / self.molecularWeights[None, :]

    def get_specific_gas_constant(self, state: FluidState) -> Array:
        """
        This method computes the mixture-specific gas constat
            inputs:
                Y: matrix of mass fractions [n,nSp]
            outputs:
                R: vector of mixture-specific gas constants [n]
        """
        assert state.composition is not None
        return get_specific_gas_constant_compiled(
            state.composition.reshape((-1, self.n_scalars)), self.molecularWeights
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
        assert state.composition is not None
        if state.temperature is None:
            state.temperature = self.get_temperature(state)
        return get_cp_compiled(
            state.temperature.flatten(),
            state.composition.reshape((-1, self.n_scalars)),
            self.T,
            self.a,
            self.b,
        ).reshape(state.shape)

    def get_gamma(self, state: FluidState) -> Array:
        """
        This method computes the specific heat ratio, gamma.
            inputs:
                state: FluidState object
            outputs:
                gamma: vector of specific heat ratios [n]
        """
        cp = self.get_cp(state)
        R = self.get_specific_gas_constant(state)
        return cp / (cp - R)

    def get_temperature(self, state: FluidState) -> Array:
        """
        This method applies the ideal gas law to compute the temperature
            inputs:
                state: FluidState object
            outputs:
                T: vector of temperatures
        """
        if state.pressure is not None and state.density is not None:
            # Compute temperature using ideal gas law
            R = self.get_specific_gas_constant(state)
            state.temperature = state.pressure / (state.density * R)
        else:
            # Compute temperature from the internal energy
            assert state.composition is not None
            assert state.internal_energy is not None
            R = self.get_specific_gas_constant(state)
            Y = state.composition
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
        if state.internal_energy is not None:
            return state.internal_energy

        assert state.temperature is not None
        T = state.temperature
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
        if state.temperature is None:
            T = self.get_temperature(state)
        else:
            T = state.temperature

        if any(np.logical_or(self.TMin > T, self.TMax < T)):
            msg = "Temperature not within table"
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
        assert state.composition is not None
        hk = self.get_species_enthalpies(state)
        return np.sum(state.composition * hk, axis=-1)

    def get_sound_speed(self, state: FluidState) -> Array:
        assert state.pressure is not None
        assert state.density is not None
        gamma = state.gamma
        if gamma is None:
            gamma = self.get_gamma(state)
        return np.sqrt(gamma * state.pressure / state.density)
