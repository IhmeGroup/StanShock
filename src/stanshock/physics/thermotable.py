from __future__ import annotations

import cantera as ct
import numpy as np
from numba import double, njit

from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.physics.fluid_base import FluidState

# Type signatures for numba
double1D = double[:]
double2D = double[:, :]
double3D = double[:, :, :]


@njit(double1D(double2D, double1D))
def get_specific_gas_constant_compiled(Y, molecularWeights):
    """
    Function used by the thermoTable class to find the gas constant. This
    function is compiled for speed-up.
        inputs:
            Y: scalar [nX,nSp-1]
            molecularWeights: species molecular weights [nSp]
        output:
            R: gas constants [nX]
    """
    # find dimensions
    nX = len(Y[:, 0])
    nSp = molecularWeights.size
    if nSp > 1:
        Y_full = np.zeros((nX, nSp))
        Y_full[:, :-1] = Y
        Y_full[:, -1] = 1.0 - np.sum(Y, axis=1)
    else:
        Y_full = Y
    # determine R
    R = np.zeros(nX)
    for iX in range(nX):
        molecularWeight = 0.0
        for iSp in range(nSp):
            molecularWeight += Y_full[iX, iSp] / molecularWeights[iSp]
        molecularWeight = 1.0 / molecularWeight
        R[iX] = ct.gas_constant / molecularWeight
    return R


# @njit(double1D(double1D, double2D, double1D, double2D, double2D))
def get_cp_compiled(T, Y, TTable, a, b):
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
    nSp = a.shape[1]
    if nSp > 1:
        Y_full = np.zeros((nX, nSp))
        Y_full[:, :-1] = Y
        Y_full[:, -1] = 1.0 - np.sum(Y, axis=1)
    else:
        Y_full = Y
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
            bbar += Y_full[iX, iSp] * (
                a[index, iSp] / 2.0 * (T[iX] + TTable[index]) + b[index, iSp]
            )
        cp[iX] = bbar
    return cp


class ThermoTable(CanteraInterface):
    """
    This is a class defined to encapsulate the temperature table with the
    relevant methods
    """

    def __init__(self, gas: ct.Solution, ox_def=None, fuel_def=None, prog_def=None):
        """
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat
        coefficients. The coefficients are selected to retain the exact
        enthalpies at the table points.
        """
        super().__init__(gas, ox_def, fuel_def, prog_def)
        nSp = gas.n_species
        self.TMin = 50.0
        self.dT = 100.0
        self.TMax = 9950.0
        self.T = np.arange(
            self.TMin, self.TMax, self.dT
        )  # vector of temperatures assuming thermal equilibrium between species
        nT = len(self.T)
        self.h = np.zeros((nT, nSp))  # matrix of species enthalpies per temperature
        # cpk = ak*T+bk for T in [Tk,Tk+1], k in {0,1,2,...,nT-1}
        self.a = np.zeros((nT, nSp))  # matrix of species first order coefficients
        self.b = np.zeros((nT, nSp))  # matrix of species zeroth order coefficients
        self.molecularWeights = gas.molecular_weights
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

    def get_specific_gas_constant(self, state: FluidState):
        """
        This method computes the mixture-specific gas constat
            inputs:
                Y: matrix of mass fractions [n,nSp]
            outputs:
                R: vector of mixture-specific gas constants [n]
        """
        return get_specific_gas_constant_compiled(
            state.composition.reshape((-1, self.n_scalars)), self.molecularWeights
        ).reshape(state.shape)

    def get_cp(self, state: FluidState):
        """
        This method computes the constant pressure specific heat as determined
        by Billet and Abgrall (2003) for the double flux method.
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                cp: vector of constant pressure specific heats
        """
        if state.temperature is None:
            state.temperature = self.get_temperature(state)
        return get_cp_compiled(
            state.temperature.flatten(),
            state.composition.reshape((-1, self.n_scalars)),
            self.T,
            self.a,
            self.b,
        ).reshape(state.shape)

    def get_frozen_enthalpy(self, T, Y):
        """
        This method computes the enthalpy according to Billet and Abgrall (2003).
        This is the enthalpy that is frozen over the time step
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                h0: vector of frozen enthalpies for the mixture [n]
        """
        if any(np.logical_or(self.TMin > T, self.TMax < T)):
            msg = "Temperature not within table"
            raise ValueError(msg)
        nT = len(T)
        indices = [int((Tk - self.TMin) / self.dT) for Tk in T]
        h0 = np.zeros(nT)
        for k, index in enumerate(indices):
            bbar = self.a[index, :] / 2.0 * (T[k] + self.T[index]) + self.b[index, :]
            h0[k] = np.dot(Y[k, :], self.h[index] - bbar * self.T[index])
        return h0

    def get_gamma(self, state: FluidState):
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

    def get_temperature(self, state: FluidState):
        """
        This method applies the ideal gas law to compute the temperature
            inputs:
                state: FluidState object
            outputs:
                T: vector of temperatures
        """
        R = self.get_specific_gas_constant(state)
        return state.pressure / (state.density * R)

    def get_pressure(self, state: FluidState):
        R = self.get_specific_gas_constant(state)
        state.pressure = state.temperature * R * state.density
        return state.pressure

    def get_sound_speed(self, state: FluidState):
        return np.sqrt(state.gamma * state.pressure / state.density)
