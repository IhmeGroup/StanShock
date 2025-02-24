"""
Copyright 2017 Kevin Grogan
Copyright 2024 Matthew Bonanni

This file is part of StanScram.

StanScram is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License.

StanScram is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with StanScram.  If not, see <https://www.gnu.org/licenses/>.
"""

# necessary modules
from __future__ import annotations

import cantera as ct
import numpy as np

from stanscram.numerics.face_extrapolation import WENO5
from stanscram.numerics.inviscid_flux import HLLC
from stanscram.numerics.viscous_flux import viscousFluxFunction
from stanscram.physics.skinfriction import skinFriction
from stanscram.physics.thermo.table import thermoTable
from stanscram.processing.initialize import (
    initializeConstant,
    initializeDiffuseInterface,
    initializeRiemannProblem,
)
from stanscram.processing.plot import plotState


class stanScram:
    """
    Class: stanScram
    --------------------------------------------------------------------------
    This is a class defined to encapsulate the data and methods used for the
    1D gasdynamics solver stanScram.
    """

    def __init__(self, gas, **kwargs):
        """
        Method: __init__
        ----------------------------------------------------------------------
        initialization of the object with default values. The keyword arguments
        allow the user to initialize the state
        """
        # initialize the class
        self.mt = 3  # number of ghost nodes
        self.mn = 3  # number of 1D Euler equations

        self.cfl = 1.0  # stability condition
        self.dx = 1.0  # grid spacing
        self.n = 10  # grid size
        self.boundaryConditions = ["outflow", "outflow"]
        self.x = np.linspace(0.0, self.dx * (self.n - 1), self.n)
        self.gas = gas  # cantera solution object for the gas
        self.r = np.ones(self.n) * gas.density  # density
        self.u = np.zeros(self.n)  # velocity
        self.p = np.ones(self.n) * gas.P  # pressure
        self.gamma = np.ones(self.n) * gas.cp / gas.cv  # specific heat ratio
        self.F = np.ones(self.n)  # thickening
        self.t = 0.0  # time
        self.verbose = True  # console output switch
        self.outputEvery = (
            1  # number of iterations of simulation advancement between logging updates
        )
        self.h = None  # height of the channel
        self.w = None  # width of the channel
        self.dlnAdt = (
            None  # area of the shock tube as a function of time (needed for quasi-1D)
        )
        self.dlnAdx = (
            None  # area of the shock tube as a function of x (needed for quasi-1D)
        )
        self.includeBoundaryLayerTerms = False  # flag to include boundary layer terms
        self.Tw = None  # wall temperature (needed for BL)
        self.sourceTerms = None  # source term function
        self.injector = None  # injector model
        self.ox_def = None  # oxidizer definition
        self.fuel_def = None  # fuel definition
        self.prog_def = None  # progress variable definition
        self.fluxFunction = HLLC
        self.initialization = None  # initialization options
        self.probes = []  # list of probe objects
        self.XTDiagrams = []  # list of XT diagram objects
        self.cf = None  # skin friction functor
        self.thermoTable = thermoTable(gas)  # thermodynamic table object
        self.optimizationIteration = 0  # counter to keep track of optimization
        self.physics = "FPV"  # flag to determine the physics model
        self.fpv_table = None  # table for FPV model
        self.reacting = False  # flag to solver about whether to solve source terms
        self.inReactingRegion = (
            lambda x, t: True
        )  # the reacting region of the shock tube.
        self.includeDiffusion = False  # exclude diffusion
        self.thickening = None  # thickening function
        self.plotStateInterval = -1  # plot the state every n iterations
        # overwrite the default data
        for key, item in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key] = item

        # set the number of scalars
        if self.physics == "FPV":
            if self.fpv_table is None:
                msg = "FPV table must be defined"
                raise Exception(msg)
            self.n_scalars = 2
            self.initZBilger()
            self.initProg()
        elif self.physics == "FRC":
            if self.injector is not None:
                msg = "JIC injector model not supported for FRC"
                raise Exception(msg)
            self.n_scalars = self.gas.n_species
        else:
            msg = "Invalid Physics Model"
            raise Exception(msg)
        self.Y = np.zeros((self.n, self.n_scalars))  # scalars

        # initialize the state
        if self.initialization is None:
            msg = "No initialization method selected"
            raise Exception(msg)
        if self.initialization[0].lower() == "constant":
            initializeConstant(self, *self.initialization[1:])
        elif self.initialization[0].lower() == "riemann":
            initializeRiemannProblem(self, *self.initialization[1:])
        elif self.initialization[0].lower() == "diffuse_interface":
            initializeDiffuseInterface(self, *self.initialization[1:])
        if (
            not self.n
            == len(self.x)
            == len(self.r)
            == len(self.u)
            == len(self.p)
            == len(self.gamma)
        ):
            msg = "Initialization Error"
            raise Exception(msg)

    def getCp(self, T, Y):
        """
        Method: getCp
        ----------------------------------------------------------------------
        This method computes the constant pressure specific heat as determined
        by Billet and Abgrall (2003) for the double flux method.
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSc]
            outputs:
                cp: vector of constant pressure specific heats
        """
        cp = np.zeros_like(T)
        if self.physics == "FPV":
            Z = Y[:, 0]
            C = Y[:, 1]
            Q = np.zeros_like(self.x)
            L = self.fpv_table.L_from_C(Z, C)
            cp = self.fpv_table.get_cp(Z, Q, L, T)
        elif self.physics == "FRC":
            cp = self.thermoTable.getCp(T, Y)
        return cp

    def getGamma(self, T, Y):
        """
        Method: getGamma
        ----------------------------------------------------------------------
        This method computes the specific heat ratio, gamma.
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSc]
            outputs:
                gamma: vector of specific heat ratios
        """
        gamma = np.zeros_like(T)
        if self.physics == "FPV":
            Z = Y[:, 0]
            C = Y[:, 1]
            Q = np.zeros_like(self.x)
            L = self.fpv_table.L_from_C(Z, C)
            gamma = self.fpv_table.get_gamma(Z, Q, L, T)
        elif self.physics == "FRC":
            gamma = self.thermoTable.getGamma(T, Y)
        return gamma

    def getMu(self, T, p, Y):
        """
        Method: getMu
        ----------------------------------------------------------------------
        This method computes the dynamic viscosity of the gas at the current state
            inputs:
                T: vector of temperatures [n]
                P: vector of pressures [n]
                Y: scalar matrix [n,nSc]
            outputs:
                mu: vector of dynamic viscosities
        """
        mu = np.zeros_like(T)
        if self.physics == "FPV":
            Z = Y[:, 0]
            C = Y[:, 1]
            Q = np.zeros_like(self.x)
            L = self.fpv_table.L_from_C(Z, C)
            mu = self.fpv_table.get_mu(Z, Q, L, T)
        elif self.physics == "FRC":
            for i, Ti in enumerate(T):
                self.gas.TP = Ti, p[i]
                if self.gas.n_species > 1:
                    self.gas.Y = Y[i, :]
                mu[i] = self.gas.viscosity
        return mu

    def getLoc(self, T, p, Y):
        """
        Method: getLoc
        ----------------------------------------------------------------------
        This method computes lambda / cv, where lambda is the thermal conductivity
        and cv is the specific heat at constant volume.
            inputs:
                T: vector of temperatures [n]
                Y: scalar matrix [n,nSc]
            outputs:
                loc: vector of lambda / cv
        """
        loc = np.zeros_like(T)
        if self.physics == "FPV":
            Z = Y[:, 0]
            C = Y[:, 1]
            Q = np.zeros_like(self.x)
            L = self.fpv_table.L_from_C(Z, C)
            loc = self.fpv_table.get_loc(Z, Q, L, T)
        elif self.physics == "FRC":
            for i, Ti in enumerate(T):
                self.gas.TP = Ti, p[i]
                if self.gas.n_species > 1:
                    self.gas.Y = Y[i, :]
                loc[i] = self.gas.thermal_conductivity / self.gas.cv
        return loc

    def getTemperature(self, r, p, Y):
        """
        Method: getTemperature
        ----------------------------------------------------------------------
        This method computes the temperature of the gas at the current state
            inputs:
                r=density
                p=pressure
                Y=scalar matrix [x,scalar]
            outputs:
                T=temperature
        """
        T = np.zeros_like(r)
        if self.physics == "FPV":
            Z = Y[:, 0]
            C = Y[:, 1]
            Q = np.zeros_like(self.x)
            L = self.fpv_table.L_from_C(Z, C)
            R = self.fpv_table.get_R(Z, Q, L)
            T = p / (r * R)
        elif self.physics == "FRC":
            T = self.thermoTable.getTemperature(r, p, Y)
        return T

    def soundSpeed(self, r, p, gamma):
        """
        Method: soundSpeed
        ----------------------------------------------------------------------
        This method returns the speed of sound for the gas at its current state
            outputs:
                speed of sound
        """
        return np.sqrt(gamma * p / r)

    def waveSpeed(self):
        """
        Method: waveSpeed
        ----------------------------------------------------------------------
        This method determines the absolute maximum of the wave speed
            outputs:
                speed of acoustic wave
        """
        return abs(self.u) + self.soundSpeed(self.r, self.p, self.gamma)

    def timeStep(self):
        """
        Method: timeStep
        ----------------------------------------------------------------------
        This method determines the maximal timestep in accord with the CFL
        condition
            outputs:
                timestep
        """
        localDts = self.dx / self.waveSpeed()
        if self.includeDiffusion:
            T = self.getTemperature(self.r, self.p, self.Y)
            cp = self.getCp(T, self.Y)
            # cv = cp / self.gamma
            mu = self.getMu(T, self.p, self.Y)
            nu = mu / self.r
            k = self.getLoc(T, self.p, self.Y) * cp * self.F
            alpha = k / (self.r * cp)
            if self.physics == "FPV":
                # unity Lewis number
                diff = alpha
            elif self.physics == "FRC":
                diff = np.zeros_like(self.x)
                for i, Ti in enumerate(T):
                    self.gas.TP = Ti, self.p[i]
                    if self.gas.n_species > 1:
                        self.gas.Y = self.Y[i, :]
                    diff[i] = np.max(self.gas.mix_diff_coeffs) * self.F[i]
            viscousDts = (
                0.5 * self.dx**2.0 / np.maximum(4.0 / 3.0 * nu, np.maximum(alpha, diff))
            )
            localDts = np.minimum(localDts, viscousDts)
        return self.cfl * min(localDts)

    def applyBoundaryConditions(self, rLR, uLR, pLR, YLR):
        """
        Method: applyBoundaryConditions
        ----------------------------------------------------------------------
        This method applies the prescribed BCs declared by the user.
        Currently, only reflecting (adiabatic wall) and outflow (symmetry)
        boundary conditions are supported. The user may include Dirichlet
        condition as well. This method returns the updated primitives.
            inputs:
                rLR=density on left and right face [2,n+1]
                uLR=velocity on left and right face [2,n+1]
                pLR=pressure on left and right face [2,n+1]
                YLR=scalar on left and right face [2,n+1,nsp]
            outputs:
                rLR=density on left and right face [2,n+1]
                uLR=velocity on left and right face [2,n+1]
                pLR=pressure on left and right face [2,n+1]
                YLR=scalar on left and right face [2,n+1,nsp]
        """
        for ibc in [0, 1]:
            NAssign = ibc
            NUse = 1 - ibc
            iX = -ibc
            rLR[NAssign, iX] = rLR[NUse, iX]
            uLR[NAssign, iX] = uLR[NUse, iX]
            pLR[NAssign, iX] = pLR[NUse, iX]
            YLR[NAssign, iX, :] = YLR[NUse, iX, :]
            if type(self.boundaryConditions[ibc]) is str:
                if (
                    self.boundaryConditions[ibc].lower() == "reflecting"
                    or self.boundaryConditions[ibc].lower() == "symmetry"
                ):
                    uLR[NAssign, iX] = 0.0
                elif self.verbose and self.boundaryConditions[ibc].lower() != "outflow":
                    print(
                        """Unrecognized Boundary Condition. Applying outflow by default.\n"""
                    )
            else:
                # assign Dirichlet conditions to (r,u,p,Y)
                if self.boundaryConditions[ibc][0] is not None:
                    rLR[NAssign, iX] = self.boundaryConditions[ibc][0]
                if self.boundaryConditions[ibc][1] is not None:
                    uLR[NAssign, iX] = self.boundaryConditions[ibc][1]
                if self.boundaryConditions[ibc][2] is not None:
                    pLR[NAssign, iX] = self.boundaryConditions[ibc][2]
                if self.boundaryConditions[ibc][3] is not None:
                    YLR[NAssign, iX, :] = self.boundaryConditions[ibc][3]
        return (rLR, uLR, pLR, YLR)

    def primitiveToConservative(self, r, u, p, Y, gamma):
        """
        Method: conservativeToPrimitive
        ----------------------------------------------------------------------
        This method transforms the primitive variables to conservative
            inputs:
                r=density
                u=velocity
                p=pressure
                Y=scalar matrix [x,scalar]
                gamma=specific heat ratio
            outputs:
                r=density
                ru=momentum
                E=total non-chemical energy
                rY=scalar density matrix
        """
        ru = r * u
        E = p / (gamma - 1.0) + 0.5 * r * u**2.0
        rY = Y * r.reshape((-1, 1))
        return (r, ru, E, rY)

    def conservativeToPrimitive(self, r, ru, E, rY, gamma):
        """
        Method: conservativeToPrimitive
        ----------------------------------------------------------------------
        This method transforms the conservative variables to the primitives
            inputs:
                r=density
                ru=momentum
                E=total non-chemical energy
                rY=scalar density matrix
                gamma=specific heat ratio
            outputs:
                r=density
                u=velocity
                p=pressure
                Y=scalar matrix [x,scalar]
        """
        u = ru / r
        p = (gamma - 1.0) * (E - 0.5 * r * u**2.0)
        Y = rY / r.reshape((-1, 1))
        # bound
        Y[Y > 1.0] = 1.0
        Y[Y < 0.0] = 0.0
        # scale
        if self.physics == "FRC":
            Y = Y / np.sum(Y, axis=1).reshape((-1, 1))
        return (r, u, p, Y)

    def initZBilger(self):
        """
        Method: initZBilger
        ----------------------------------------------------------------------
        This method initializes the Bilger mixture fraction
        """
        self.Z_weights = np.zeros(self.gas.n_species)
        self.Z_offset = 0.0
        denom = 0.0

        # Set the values for C, H, and O:
        stoich = {
            "C": 2.0,
            "H": 0.5,
            "O": -1.0,
        }

        for element in self.gas.element_names:
            if element not in stoich:
                continue
            C = stoich[element]

            idx_element = self.gas.element_index(element)
            W = self.gas.atomic_weight(element)

            self.gas.X = self.ox_def
            Yo = self.gas.elemental_mass_fraction(element)

            self.gas.X = self.fuel_def
            Yf = self.gas.elemental_mass_fraction(element)

            denom += C * (Yf - Yo) / W

            for k in range(self.gas.n_species):
                self.Z_weights[k] += C * self.gas.n_atoms(k, idx_element)

            self.Z_offset -= C * Yo / W

        self.Z_weights /= denom * self.gas.molecular_weights
        self.Z_offset /= denom

    def ZBilger(self, Y):
        """
        Method: ZBilger
        ----------------------------------------------------------------------
        This method calculates the Bilger mixture fraction
            inputs:
                Y=species mass fraction
            outputs:
                Z=mixture fraction
        """
        return np.clip(np.dot(Y, self.Z_weights) + self.Z_offset, 0.0, 1.0)

    def initProg(self):
        """
        Method: initProg
        ----------------------------------------------------------------------
        This method initializes the progress variable
        """
        if self.prog_def is None:
            msg = "Progress Variable Not Defined"
            raise Exception(msg)
        self.prog_weights = np.zeros(self.gas.n_species)
        for sp, val in self.prog_def.items():
            self.prog_weights[self.gas.species_index(sp)] = val
        if np.sum(self.prog_weights) == 0.0:
            msg = "Progress Variable Weights Sum to Zero"
            raise Exception(msg)
        self.prog_weights /= np.sum(self.prog_weights)

    def Prog(self, Y):
        """
        Method: Prog
        ----------------------------------------------------------------------
        This method computes the progress variable
            inputs:
                Y=species mass fraction
            outputs:
                progress variable
        """
        return np.clip(np.dot(Y, self.prog_weights), 0.0, 1.0)

    def flux(self, r, u, p, Y, gamma):
        """
        Method: flux
        ----------------------------------------------------------------------
        This method calculates the advective flux
            inputs:
                r=density
                u=velocity
                p=pressure
                Y=scalar matrix [x,scalar]
                gamma=specific heat ratio
            outputs:
                rhs=the update due to the flux
        """
        mt = self.mt
        mn = self.mn
        # find the left and right WENO states from the WENO interpolation
        nx = len(r)
        PLR = WENO5(r, u, p, Y, gamma)
        # extract and apply boundary conditions
        rLR = PLR[:, :, 0]
        uLR = PLR[:, :, 1]
        pLR = PLR[:, :, 2]
        YLR = PLR[:, :, mt:]
        rLR, uLR, pLR, YLR = self.applyBoundaryConditions(rLR, uLR, pLR, YLR)
        # calculate the flux
        fL = self.fluxFunction(rLR, uLR, pLR, YLR, gamma[mt : -mt + 1])
        fR = self.fluxFunction(rLR, uLR, pLR, YLR, gamma[mt - 1 : -mt])
        rhs = np.zeros((nx, mn + self.n_scalars))
        rhs[mt:-mt, :] = -(fR[1:] - fL[:-1]) / self.dx
        return rhs

    def viscousFlux(self, r, u, p, Y, gamma):
        """
        Method: viscousFlux
        ----------------------------------------------------------------------
        This method calculates the viscous flux
            inputs:
                r=density
                u=velocity
                p=pressure
                Y=scalar matrix [x,scalar]
                gamma=specific heat ratio
            outputs:
                rhs=the update due to the viscous flux
        """
        mt = self.mt
        mn = self.mn

        # first order interpolation to the edge states and apply boundary conditions
        rLR = np.concatenate(
            (r[mt - 1 : -mt].reshape(1, -1), r[mt : -mt + 1].reshape(1, -1)), axis=0
        )
        uLR = np.concatenate(
            (u[mt - 1 : -mt].reshape(1, -1), u[mt : -mt + 1].reshape(1, -1)), axis=0
        )
        pLR = np.concatenate(
            (p[mt - 1 : -mt].reshape(1, -1), p[mt : -mt + 1].reshape(1, -1)), axis=0
        )
        YLR = np.concatenate(
            (
                Y[mt - 1 : -mt, :].reshape(1, -1, self.n_scalars),
                Y[mt : -mt + 1, :].reshape(1, -1, self.n_scalars),
            ),
            axis=0,
        )
        rLR, uLR, pLR, YLR = self.applyBoundaryConditions(rLR, uLR, pLR, YLR)
        # calculate the flux
        f = viscousFluxFunction(self, rLR, uLR, pLR, YLR)
        rhs = np.zeros((self.n + 2 * mt, mn + self.n_scalars))
        rhs[mt:-mt, :] = (f[1:, :] - f[:-1, :]) / self.dx  # central difference
        return rhs

    def advanceAdvection(self, dt):
        """
        Method: advanceAdvection
        ----------------------------------------------------------------------
        This method advances the advection terms by the prescribed timestep.
        The advection terms are integrated using RK3.
            inputs
                dt=time step
        """
        # initialize
        mt = self.mt
        mn = self.mn
        r = np.ones(self.n + 2 * mt)
        u = np.ones(self.n + 2 * mt)
        p = np.ones(self.n + 2 * mt)
        gamma = np.ones(self.n + 2 * mt)
        gamma[:mt], gamma[-mt:] = self.gamma[0], self.gamma[-1]
        Y = np.ones((self.n + 2 * mt, self.n_scalars))
        (r[mt:-mt], u[mt:-mt], p[mt:-mt], Y[mt:-mt, :], gamma[mt:-mt]) = (
            self.r,
            self.u,
            self.p,
            self.Y,
            self.gamma,
        )
        (r, ru, E, rY) = self.primitiveToConservative(r, u, p, Y, gamma)
        # 1st stage of RK3
        rhs = self.flux(r, u, p, Y, gamma)
        r1 = r + dt * rhs[:, 0]
        ru1 = ru + dt * rhs[:, 1]
        E1 = E + dt * rhs[:, 2]
        rY1 = rY + dt * rhs[:, mn:]
        (r1, u1, p1, Y1) = self.conservativeToPrimitive(r1, ru1, E1, rY1, gamma)
        # 2nd stage of RK3
        rhs = self.flux(r1, u1, p1, Y1, gamma)
        r2 = 0.75 * r + 0.25 * r1 + 0.25 * dt * rhs[:, 0]
        ru2 = 0.75 * ru + 0.25 * ru1 + 0.25 * dt * rhs[:, 1]
        E2 = 0.75 * E + 0.25 * E1 + 0.25 * dt * rhs[:, 2]
        rY2 = 0.75 * rY + 0.25 * rY1 + 0.25 * dt * rhs[:, mn:]
        (r2, u2, p2, Y2) = self.conservativeToPrimitive(r2, ru2, E2, rY2, gamma)
        # 3rd stage of RK3
        rhs = self.flux(r2, u2, p2, Y2, gamma)
        r = (1.0 / 3.0) * r + (2.0 / 3.0) * r2 + (2.0 / 3.0) * dt * rhs[:, 0]
        ru = (1.0 / 3.0) * ru + (2.0 / 3.0) * ru2 + (2.0 / 3.0) * dt * rhs[:, 1]
        E = (1.0 / 3.0) * E + (2.0 / 3.0) * E2 + (2.0 / 3.0) * dt * rhs[:, 2]
        rY = (1.0 / 3.0) * rY + (2.0 / 3.0) * rY2 + (2.0 / 3.0) * dt * rhs[:, mn:]
        (r, u, p, Y) = self.conservativeToPrimitive(r, ru, E, rY, gamma)
        # update
        T0 = self.getTemperature(r[mt:-mt], p[mt:-mt], Y[mt:-mt])
        gamma[mt:-mt] = self.getGamma(T0, Y[mt:-mt])
        (self.r, self.u, self.p, self.Y, self.gamma) = (
            r[mt:-mt],
            u[mt:-mt],
            p[mt:-mt],
            Y[mt:-mt],
            gamma[mt:-mt],
        )

    def advanceChemistry(self, dt):
        """
        Method: advanceChemistry
        ----------------------------------------------------------------------
        This method advances the combustion chemistry of a reacting system. It
        is only called if the "reacting" flag is set to True.
            inputs
                dt=time step
        """
        if not self.reacting:
            return
        if self.physics == "FPV":
            self.advanceChemistryFPV(dt)
        elif self.physics == "FRC":
            self.advanceChemistryFRC(dt)

    def advanceChemistryFPV(self, dt):
        """
        Method: advanceChemistryFPV
        ----------------------------------------------------------------------
        This method advances the combustion chemistry of a reacting system using
        the flamelet progress variable approach. It is only called if the "reacting"
        flag is set to True.
            inputs
                dt=time step
        """
        # Using mixed-is-burned (MIB)
        # (r,ru,E,rY)=self.primitiveToConservative(self.r,self.u,self.p,self.Y,self.gamma)
        # Z = rY[:, 0] / r
        # Q = np.zeros(self.n)
        # C = rY[:, 1] / r
        # L = self.fpv_table.L_from_C(Z, C)
        # e_chem0 = r * self.fpv_table.lookup('E0_CHEM', Z, Q, L)
        # C1, e_chem1 = self.injector.get_MIB_profiles()
        # e_chem1 *= r
        # E1 = E + e_chem0 - e_chem1
        # rY1 = rY
        # rY1[:, 1] = r * C1
        # (r,u,p,Y)=self.conservativeToPrimitive(r,ru,E1,rY1,self.gamma)

        # Using FPV
        # initialize
        (r, ru, E, rY) = self.primitiveToConservative(
            self.r, self.u, self.p, self.Y, self.gamma
        )
        Q = np.zeros(self.n)
        L = self.fpv_table.L_from_C(rY[:, 0] / r, rY[:, 1] / r)
        e_chem0 = r * self.fpv_table.lookup("E0_CHEM", self.Y[:, 0], Q, L)
        # 1st stage of RK2
        rhsY = np.zeros((self.n, self.n_scalars))
        omegaC = self.injector.get_chemical_sources(self.Y[:, 0], self.Y[:, 1])
        rhsY[:, 1] = omegaC * r
        rY1 = rY + dt * rhsY
        L1 = self.fpv_table.L_from_C(rY1[:, 0] / r, rY1[:, 1] / r)
        e_chem1 = r * self.fpv_table.lookup("E0_CHEM", rY1[:, 0] / r, Q, L1)
        E1 = E + e_chem0 - e_chem1
        (r1, u1, p1, Y1) = self.conservativeToPrimitive(r, ru, E1, rY1, self.gamma)
        # 2nd stage of RK2
        omegaC1 = self.injector.get_chemical_sources(Y1[:, 0], Y1[:, 1])
        rhsY[:, 1] = omegaC1 * r1
        rY = 0.5 * (rY + rY1 + dt * rhsY)
        L = self.fpv_table.L_from_C(rY[:, 0] / r, rY[:, 1] / r)
        e_chem2 = r * self.fpv_table.lookup("E0_CHEM", rY[:, 0] / r, Q, L)
        E = E + e_chem0 - e_chem2
        (r, u, p, Y) = self.conservativeToPrimitive(r, ru, E, rY, self.gamma)

        # update properties
        T0 = self.getTemperature(r, p, Y)
        self.gamma = self.getGamma(T0, Y)
        (self.r, self.u, self.p, self.Y) = (r, u, p, Y)

    def advanceChemistryFRC(self, dt):
        """
        Method: advanceChemistryFRC
        ----------------------------------------------------------------------
        This method advances the combustion chemistry of a reacting system using
        finite rate chemistry. It is only called if the "reacting" flag is set to True.
            inputs
                dt=time step
        """

        #######################################################################
        def dydt(t, y, args):
            """
            function: dydt
            -------------------------------------------------------------------
            this function gives the source terms of a constant volume reactor
                inputs
                    dt=time step
            """
            # unpack the input
            r = args[0]
            F = args[1]
            Y = y[:-1]
            T = y[-1]
            # set the state for the gas object
            self.gas.TDY = T, r, Y
            # gas properties
            cv = self.gas.cv_mass
            W = self.gas.molecular_weights
            wHatDot = self.gas.net_production_rates  # kmol/m^3.s
            wDot = wHatDot * W  # kg/m^3.s
            eRT = self.gas.standard_int_energies_RT
            # compute the derivatives
            YDot = wDot / r
            TDot = -np.sum(eRT * wHatDot) * ct.gas_constant * T / (r * cv)
            f = np.zeros(self.n_scalars + 1)
            f[:-1] = YDot
            f[-1] = TDot
            return f / F

        #######################################################################
        from scipy import integrate

        # get indices
        indices = [k for k in range(self.n) if self.inReactingRegion(self.x[k], self.t)]
        Ts = self.getTemperature(self.r[indices], self.p[indices], self.Y[indices, :])
        # initialize integrator
        y0 = np.zeros(self.gas.n_species + 1)
        integrator = integrate.ode(dydt).set_integrator("lsoda")
        for TIndex, k in enumerate(indices):
            # initialize
            y0[:-1] = self.Y[k, :]
            y0[-1] = Ts[TIndex]
            args = [self.r[k], self.F[k]]
            integrator.set_initial_value(y0, 0.0)
            integrator.set_f_params(args)
            # solve
            integrator.integrate(dt)
            # clip and normalize
            Y = integrator.y[:-1]
            Y[Y > 1.0] = 1.0
            Y[Y < 0.0] = 0.0
            Y /= np.sum(Y)
            # update
            self.Y[k, :] = Y
            T = integrator.y[-1]
            self.gas.TDY = T, self.r[k], Y
            self.p[k] = self.gas.P
        # update gamma
        T = self.getTemperature(self.r, self.p, self.Y)
        self.gamma = self.getGamma(T, self.Y)

    def advanceQuasi1D(self, dt):
        """
        Method: advanceQuasi1D
        ----------------------------------------------------------------------
        This method advances the quasi-1D terms used to model area changes in
        the shock tube. The client must supply the functions dlnAdt and dlnAdx
        to the StanScram object.
            inputs
                dt=time step
        """
        mn = self.mn

        #######################################################################
        def dydt(t, y, args):
            """
            function: dydt
            -------------------------------------------------------------------
            this function gives the source terms for the quasi 1D
                inputs
                    dt=time step
            """
            # unpack the input and initialize
            x, gamma = args
            r, ru, E = y
            p = (gamma - 1.0) * (E - 0.5 * ru**2.0 / r)
            f = np.zeros(3)
            # create quasi-1D right hand side
            if self.dlnAdt is not None:
                dlnAdt = self.dlnAdt(x, t)[0]
                f[0] -= r * dlnAdt
                f[1] -= ru * dlnAdt
                f[2] -= E * dlnAdt
            if self.dlnAdx is not None:
                dlnAdx = self.dlnAdx(x, t)[0]
                f[0] -= ru * dlnAdx
                f[1] -= (ru**2.0 / r) * dlnAdx
                f[2] -= (ru / r * (E + p)) * dlnAdx
            return f

        #######################################################################
        from scipy import integrate

        # initialize integrator
        y0 = np.zeros(3)
        integrator = integrate.ode(dydt).set_integrator("lsoda")
        (r, ru, E, _) = self.primitiveToConservative(
            self.r, self.u, self.p, self.Y, self.gamma
        )
        # determine the indices
        iIn = []
        eIn = np.arange(self.x.shape[0])
        if self.dlnAdt is not None:
            dlnAdt = self.dlnAdt(self.x, self.t)
            iIn = np.arange(self.x.shape[0])[dlnAdt != 0.0]
            eIn = np.arange(self.x.shape[0])[dlnAdt == 0.0]
        # integrate implicitly
        for i in iIn:
            # initialize
            y0[:] = r[i], ru[i], E[i]
            args = np.array([self.x[i]]), self.gamma[i]
            integrator.set_initial_value(y0, self.t)
            integrator.set_f_params(args)
            # solve
            integrator.integrate(self.t + dt)
            # update
            r[i], ru[i], E[i] = integrator.y
        # integrate explicitly
        rhs = np.zeros((mn, eIn.shape[0]))
        if self.dlnAdt is not None:
            dlnAdt = self.dlnAdt(self.x, self.t)[eIn]
            rhs[0] -= r[eIn] * dlnAdt
            rhs[1] -= ru[eIn] * dlnAdt
            rhs[2] -= E[eIn] * dlnAdt
        if self.dlnAdx is not None:
            dlnAdx = self.dlnAdx(self.x, self.t)[eIn]
            rhs[0] -= ru[eIn] * dlnAdx
            rhs[1] -= (ru[eIn] ** 2.0 / r[eIn]) * dlnAdx
            rhs[2] -= (self.u[eIn] * (E[eIn] + self.p[eIn])) * dlnAdx
        # update
        r[eIn] += dt * rhs[0]
        ru[eIn] += dt * rhs[1]
        E[eIn] += dt * rhs[2]
        rY = r.reshape((r.shape[0], 1)) * self.Y
        (self.r, self.u, self.p, _) = self.conservativeToPrimitive(
            r, ru, E, rY, self.gamma
        )
        T = self.getTemperature(self.r, self.p, self.Y)
        self.gamma = self.getGamma(T, self.Y)

    def advanceBoundaryLayer(self, dt):
        """
        Method: advanceBoundaryLayer
        ----------------------------------------------------------------------
        This method advances the boundary layer terms
            inputs
                dt=time step
        """

        #######################################################################
        def nusseltNumber(Re, Pr, cf):
            """
            Function: nusseltNumber
            ----------------------------------------------------------------------
            This function defines the nusselt Number as a function of the
            Reynolds number. These functions are empirical correlations taken
            from Kayes. The selection of the correlations assumes that this solver
            will be used for gasses.
                inputs:
                    Re=Reynolds number
                    Pr=Prandtl number
                    cf=skin friction
                outputs:
                    Nu=Nusselt number
            """
            # define the transitional Reynolds number
            ReCrit = 2300
            ReLowTurbulent = 2e5  # taken frkom figure 14-5 of Kayes for Pr=0.7
            Nu = np.zeros_like(Re)
            # laminar portion of the flow
            laminarIndices = np.logical_and(Re > 0.0, Re <= ReCrit)
            Nu[laminarIndices] = 3.657  # from the analytical solution
            # low turbulent portion of the flow (accounts for isothermal wall)
            lowTurublentIndices = np.logical_and(Re > ReCrit, Re <= ReLowTurbulent)
            ReLT, PrLT = Re[lowTurublentIndices], Pr[lowTurublentIndices]
            Nu[lowTurublentIndices] = (
                0.021 * PrLT**0.5 * ReLT**0.8
            )  # empircal correlation for isothermal case
            # highly turbulent portion of the flow (data shows that boundary condition is less important)
            # highTurublentIndices = Re > ReLowTurbulent
            highTurublentIndices = Re > 2300.0
            ReHT, PrHT, cfHT = (
                Re[highTurublentIndices],
                Pr[highTurublentIndices],
                cf[highTurublentIndices],
            )
            Nu[highTurublentIndices] = (
                ReHT
                * PrHT
                * cfHT
                / 2.0
                / (0.88 + 13.39 * (PrHT ** (2.0 / 3.0) - 0.78) * np.sqrt(cfHT / 2.0))
            )
            return Nu

        #######################################################################
        if self.h is None or self.w is None or self.Tw is None:
            msg = "stanShock improperly initialized for boundary layer terms"
            raise Exception(msg)
        D = 2 * self.h * self.w / (self.h + self.w)
        # compute gas properties
        T = self.getTemperature(self.r, self.p, self.Y)
        cp = self.getCp(T, self.Y)
        mu = self.getMu(T, self.p, self.Y)
        k = self.getLoc(T, self.p, self.Y) * cp
        # compute non-dimensional numbers
        Re = abs(self.r * self.u * D / mu)
        Pr = cp * mu / k
        # skin friction coefficent
        if self.cf is None:
            self.cf = skinFriction()  # initialize the functor
        cf = self.cf(Re)
        # shear stress on wall
        shear = cf * (0.5 * self.r * self.u**2.0) * (np.sign(self.u))
        # Stanton number and heat transfer to wall
        Nu = nusseltNumber(Re, Pr, cf)
        qloss = Nu * k / D * (T - self.Tw)
        # update
        (r, ru, E, rY) = self.primitiveToConservative(
            self.r, self.u, self.p, self.Y, self.gamma
        )
        ru -= shear * 4.0 / D * dt
        E -= qloss * 4.0 / D * dt
        (self.r, self.u, self.p, _) = self.conservativeToPrimitive(
            r, ru, E, rY, self.gamma
        )
        T = self.getTemperature(self.r, self.p, self.Y)
        self.gamma = self.getGamma(T, self.Y)

    def advanceDiffusion(self, dt):
        """
        Method: advanceDiffusion
        ----------------------------------------------------------------------
        This method advances the diffusion terms in the axial direction
            inputs
                dt=time step
        """
        # initialize
        mt = self.mt
        mn = self.mn
        r = np.ones(self.n + 2 * mt)
        u = np.ones(self.n + 2 * mt)
        p = np.ones(self.n + 2 * mt)
        gamma = np.ones(self.n + 2 * mt)
        gamma[:mt], gamma[-mt:] = self.gamma[0], self.gamma[-1]
        Y = np.ones((self.n + 2 * mt, self.n_scalars))
        (r[mt:-mt], u[mt:-mt], p[mt:-mt], Y[mt:-mt, :], gamma[mt:-mt]) = (
            self.r,
            self.u,
            self.p,
            self.Y,
            self.gamma,
        )
        (r, ru, E, rY) = self.primitiveToConservative(r, u, p, Y, gamma)
        if self.thickening is not None:
            self.F = self.thickening(self)
        # 1st stage of RK2
        rhs = self.viscousFlux(r, u, p, Y, gamma)
        r1 = r + dt * rhs[:, 0]
        ru1 = ru + dt * rhs[:, 1]
        E1 = E + dt * rhs[:, 2]
        rY1 = rY + dt * rhs[:, mn:]
        (r1, u1, p1, Y1) = self.conservativeToPrimitive(r1, ru1, E1, rY1, gamma)
        # 2nd stage of RK2
        rhs = self.viscousFlux(r1, u1, p1, Y1, gamma)
        r = 0.5 * (r + r1 + dt * rhs[:, 0])
        ru = 0.5 * (ru + ru1 + dt * rhs[:, 1])
        E = 0.5 * (E + E1 + dt * rhs[:, 2])
        rY = 0.5 * (rY + rY1 + dt * rhs[:, mn:])
        (r, u, p, Y) = self.conservativeToPrimitive(r, ru, E, rY, gamma)
        # update
        T0 = self.getTemperature(r[mt:-mt], p[mt:-mt], Y[mt:-mt])
        gamma[mt:-mt] = self.getGamma(T0, Y[mt:-mt])
        (self.r, self.u, self.p, self.Y, self.gamma) = (
            r[mt:-mt],
            u[mt:-mt],
            p[mt:-mt],
            Y[mt:-mt],
            gamma[mt:-mt],
        )

    def advanceSourceTerms(self, dt):
        """
        Method: advanceSourceTerms
        ----------------------------------------------------------------------
        This method advances the source terms in the axial direction
            inputs
                dt=time step
        """
        # initialize
        mn = self.mn
        (r, ru, E, rY) = self.primitiveToConservative(
            self.r, self.u, self.p, self.Y, self.gamma
        )
        # 1st stage of RK2
        rhs = self.sourceTerms(r, ru, E, rY, self.gamma, self.x, self.t)
        r1 = r + dt * rhs[:, 0]
        ru1 = ru + dt * rhs[:, 1]
        E1 = E + dt * rhs[:, 2]
        rY1 = rY + dt * rhs[:, mn:]
        (r1, u1, p1, Y1) = self.conservativeToPrimitive(r1, ru1, E1, rY1, self.gamma)
        # 2nd stage of RK2
        rhs = self.sourceTerms(r1, ru1, E1, rY1, self.gamma, self.x, self.t + dt)
        r = 0.5 * (r + r1 + dt * rhs[:, 0])
        ru = 0.5 * (ru + ru1 + dt * rhs[:, 1])
        E = 0.5 * (E + E1 + dt * rhs[:, 2])
        rY = 0.5 * (rY + rY1 + dt * rhs[:, mn:])
        (r, u, p, Y) = self.conservativeToPrimitive(r, ru, E, rY, self.gamma)
        # update
        T0 = self.getTemperature(r, p, Y)
        self.gamma = self.getGamma(T0, Y)
        (self.r, self.u, self.p, self.Y) = (r, u, p, Y)

    def advanceInjector(self, dt):
        """
        Method: advanceInjector
        ----------------------------------------------------------------------
        This method advances the source terms from the injector using the
        jet-in-crossflow model.
            inputs
                dt=time step
        """
        # initialize
        mn = self.mn
        (r, ru, E, rY) = self.primitiveToConservative(
            self.r, self.u, self.p, self.Y, self.gamma
        )
        self.injector.update_fluid_tip_positions(dt, self.t, self.u)
        # 1st stage of RK2
        rhs = self.injector.get_injector_sources(
            r, ru, E, rY[:, 0], rY[:, 1], self.gamma, self.t
        )
        r1 = r + dt * rhs[:, 0]
        ru1 = ru + dt * rhs[:, 1]
        E1 = E + dt * rhs[:, 2]
        rY1 = rY + dt * rhs[:, mn:]
        (r1, u1, p1, Y1) = self.conservativeToPrimitive(r1, ru1, E1, rY1, self.gamma)
        # 2nd stage of RK2
        rhs = self.injector.get_injector_sources(
            r1, ru1, E1, rY1[:, 0], rY1[:, 1], self.gamma, self.t + dt
        )
        r = 0.5 * (r + r1 + dt * rhs[:, 0])
        ru = 0.5 * (ru + ru1 + dt * rhs[:, 1])
        E = 0.5 * (E + E1 + dt * rhs[:, 2])
        rY = 0.5 * (rY + rY1 + dt * rhs[:, mn:])
        (r, u, p, Y) = self.conservativeToPrimitive(r, ru, E, rY, self.gamma)
        # update
        T0 = self.getTemperature(r, p, Y)
        self.gamma = self.getGamma(T0, Y)
        (self.r, self.u, self.p, self.Y) = (r, u, p, Y)

    def updateProbes(self, iters):
        """
        Method: updateProbes
        ----------------------------------------------------------------------
        This method updates all the probes to the current value
        """

        # update probes
        for probe in self.probes:
            if iters % (probe.skipSteps + 1) == 0:
                probe.update(self)

    def updateXTDiagrams(self, iters):
        """
        Method: updateXTDiagrams
        ----------------------------------------------------------------------
        This method updates all the XT Diagrams to the current value.
        """
        # update diagrams
        for XTDiagram in self.XTDiagrams:
            if iters % (XTDiagram.skipSteps + 1) == 0:
                XTDiagram.update(self)

    def advanceSimulation(self, tFinal, res_p_target=-1.0):
        """
        Method: advanceSimulation
        ----------------------------------------------------------------------
        This method advances the simulation until the prescribed time, tFinal
            inputs
                    tFinal=final time
        """
        iters = 0
        res_p = np.inf
        while self.t < tFinal and res_p > res_p_target:
            p_old = self.p
            dt = min(tFinal - self.t, self.timeStep())
            # advance advection and chemistry
            if self.physics == "FPV":
                self.advanceAdvection(dt)
                self.advanceChemistry(dt)
            elif self.physics == "FRC":
                # use Strang splitting
                self.advanceChemistry(dt / 2.0)
                self.advanceAdvection(dt)
                self.advanceChemistry(dt / 2.0)
            # advance other terms
            if self.includeDiffusion:
                self.advanceDiffusion(dt)
            if self.dlnAdt is not None or self.dlnAdx is not None:
                self.advanceQuasi1D(dt)
            if self.includeBoundaryLayerTerms:
                self.advanceBoundaryLayer(dt)
            if self.sourceTerms is not None:
                self.advanceSourceTerms(dt)
            if self.injector is not None:
                self.advanceInjector(dt)
            # perform other updates
            self.t += dt
            self.updateProbes(iters)
            self.updateXTDiagrams(iters)
            iters += 1
            res_p = np.linalg.norm(self.p - p_old)
            if self.verbose and iters % self.outputEvery == 0:
                print(
                    f"Iteration: {iters}. Current time: {self.t}. Time step: {dt:e}. "
                    + f"Max T[K]: {self.getTemperature(self.r, self.p, self.Y).max()}. "
                    + f"Residual(p): {res_p}."
                )
            if (self.plotStateInterval > 0) and (iters % self.plotStateInterval == 0):
                plotState(
                    self, f"figures/anim/test_{iters // self.plotStateInterval:05d}.png"
                )
