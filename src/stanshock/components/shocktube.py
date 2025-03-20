from __future__ import annotations

import cantera as ct
import numpy as np

from stanshock.numerics.face_extrapolation import weno5
from stanshock.numerics.inviscid_flux import hllc_flux
from stanshock.numerics.viscous_flux import viscous_flux
from stanshock.physics.skinfriction import SkinFriction
from stanshock.physics.thermo.table import ThermoTable
from stanshock.processing.initialize import (
    initialize_constant,
    initialize_diffuse_interface,
    initialize_riemann_problem,
)
from stanshock.processing.plot import plot_state
from stanshock.processing.probe import Probe


class ShockTube:
    """
    This is a class defined to encapsulate the data and methods used for the
    1D gasdynamics solver.
    """

    def __init__(self, gas, **kwargs):
        """
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
        self.fluxFunction = hllc_flux
        self.initialization = None  # initialization options
        self.probes = []  # list of probe objects
        self.XTDiagrams = []  # list of XT diagram objects
        self.cf = None  # skin friction functor
        self.DInner = (
            None  # Inner diameter of the shock tube as a function of x (needed for BL)
        )
        self.DOuter = (
            None  # Outer diameter of the shock tube as a function of x (needed for BL)
        )
        self.Tw = None  # temperature of the wall (needed for BL)
        self.thermoTable = ThermoTable(gas)  # thermodynamic table object
        self.get_temperature = self.thermoTable.get_temperature
        self.optimizationIteration = 0  # counter to keep track of optimization
        self.physics = "FRC"  # flag to determine the physics model
        self.reacting = False  # flag to solver about whether to solve source terms
        self.inReactingRegion = (
            lambda _x, _t: True
        )  # the reacting region of the shock tube.
        self.includeDiffusion = False  # exclude diffusion
        self.thickening = None  # thickening function
        self.plotStateInterval = -1  # plot the state every n iterations
        # overwrite the default data
        for key, item in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key] = item

        # set the number of scalars
        self.n_scalars = self.gas.n_species
        self.Y = np.zeros((self.n, self.n_scalars))  # scalars

        # initialize the state
        if self.initialization is None:
            msg = "No initialization method selected"
            raise Exception(msg)
        if self.initialization[0].lower() == "constant":
            initialize_constant(self, *self.initialization[1:])
        elif self.initialization[0].lower() == "riemann":
            initialize_riemann_problem(self, *self.initialization[1:])
        elif self.initialization[0].lower() == "diffuse_interface":
            initialize_diffuse_interface(self, *self.initialization[1:])
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

    def get_sound_speed(self, r, p, gamma):
        """
        This method returns the speed of sound for the gas at its current state
            outputs:
                speed of sound
        """
        return np.sqrt(gamma * p / r)

    def get_wave_speed(self):
        """
        This method determines the absolute maximum of the wave speed
            outputs:
                speed of acoustic wave
        """
        return abs(self.u) + self.get_sound_speed(self.r, self.p, self.gamma)

    def get_time_step(self):
        """
        This method determines the maximal timestep in accord with the CFL
        condition
            outputs:
                timestep
        """
        localDts = self.dx / self.get_wave_speed()
        if self.includeDiffusion:
            T = self.thermoTable.get_temperature(self.r, self.p, self.Y)
            cv = self.thermoTable.get_cp(T, self.Y) / self.gamma
            alpha, nu, diff = np.zeros_like(T), np.zeros_like(T), np.zeros_like(T)
            for i, Ti in enumerate(T):
                self.gas.TP = Ti, self.p[i]
                if self.gas.n_species > 1:
                    self.gas.Y = self.Y[i, :]
                nu[i] = self.gas.viscosity / self.gas.density
                alpha[i] = (
                    self.gas.thermal_conductivity / self.gas.density / cv[i] * self.F[i]
                )
                diff[i] = np.max(self.gas.mix_diff_coeffs) * self.F[i]
            viscousDts = (
                0.5 * self.dx**2.0 / np.maximum(4.0 / 3.0 * nu, np.maximum(alpha, diff))
            )
            localDts = np.minimum(localDts, viscousDts)
        return self.cfl * min(localDts)

    def apply_boundary_conditions(self, rLR, uLR, pLR, YLR):
        """
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

    def primitive_to_conservative(self, r, u, p, Y, gamma):
        """
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

    def conservative_to_primitive(self, r, ru, E, rY, gamma):
        """
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
        Y = Y / np.sum(Y, axis=1).reshape((-1, 1))
        return (r, u, p, Y)

    def get_inviscid_flux(self, r, u, p, Y, gamma):
        """
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
        PLR = weno5(r, u, p, Y, gamma)
        # extract and apply boundary conditions
        rLR = PLR[:, :, 0]
        uLR = PLR[:, :, 1]
        pLR = PLR[:, :, 2]
        YLR = PLR[:, :, mt:]
        rLR, uLR, pLR, YLR = self.apply_boundary_conditions(rLR, uLR, pLR, YLR)
        # calculate the flux
        fL = self.fluxFunction(rLR, uLR, pLR, YLR, gamma[mt : -mt + 1])
        fR = self.fluxFunction(rLR, uLR, pLR, YLR, gamma[mt - 1 : -mt])
        rhs = np.zeros((nx, mn + self.n_scalars))
        rhs[mt:-mt, :] = -(fR[1:] - fL[:-1]) / self.dx
        return rhs

    def get_viscous_flux(self, r, u, p, Y, gamma):
        """
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
        _ = gamma  # Hack to silence linter

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
        rLR, uLR, pLR, YLR = self.apply_boundary_conditions(rLR, uLR, pLR, YLR)
        # calculate the flux
        f = viscous_flux(self, rLR, uLR, pLR, YLR)
        rhs = np.zeros((self.n + 2 * mt, mn + self.n_scalars))
        rhs[mt:-mt, :] = (f[1:, :] - f[:-1, :]) / self.dx  # central difference
        return rhs

    def advance_advection(self, dt):
        """
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
        (r, ru, E, rY) = self.primitive_to_conservative(r, u, p, Y, gamma)
        # 1st stage of RK3
        rhs = self.get_inviscid_flux(r, u, p, Y, gamma)
        r1 = r + dt * rhs[:, 0]
        ru1 = ru + dt * rhs[:, 1]
        E1 = E + dt * rhs[:, 2]
        rY1 = rY + dt * rhs[:, mn:]
        (r1, u1, p1, Y1) = self.conservative_to_primitive(r1, ru1, E1, rY1, gamma)
        # 2nd stage of RK3
        rhs = self.get_inviscid_flux(r1, u1, p1, Y1, gamma)
        r2 = 0.75 * r + 0.25 * r1 + 0.25 * dt * rhs[:, 0]
        ru2 = 0.75 * ru + 0.25 * ru1 + 0.25 * dt * rhs[:, 1]
        E2 = 0.75 * E + 0.25 * E1 + 0.25 * dt * rhs[:, 2]
        rY2 = 0.75 * rY + 0.25 * rY1 + 0.25 * dt * rhs[:, mn:]
        (r2, u2, p2, Y2) = self.conservative_to_primitive(r2, ru2, E2, rY2, gamma)
        # 3rd stage of RK3
        rhs = self.get_inviscid_flux(r2, u2, p2, Y2, gamma)
        r = (1.0 / 3.0) * r + (2.0 / 3.0) * r2 + (2.0 / 3.0) * dt * rhs[:, 0]
        ru = (1.0 / 3.0) * ru + (2.0 / 3.0) * ru2 + (2.0 / 3.0) * dt * rhs[:, 1]
        E = (1.0 / 3.0) * E + (2.0 / 3.0) * E2 + (2.0 / 3.0) * dt * rhs[:, 2]
        rY = (1.0 / 3.0) * rY + (2.0 / 3.0) * rY2 + (2.0 / 3.0) * dt * rhs[:, mn:]
        (r, u, p, Y) = self.conservative_to_primitive(r, ru, E, rY, gamma)
        # update
        T0 = self.thermoTable.get_temperature(r[mt:-mt], p[mt:-mt], Y[mt:-mt])
        gamma[mt:-mt] = self.thermoTable.get_gamma(T0, Y[mt:-mt])
        (self.r, self.u, self.p, self.Y, self.gamma) = (
            r[mt:-mt],
            u[mt:-mt],
            p[mt:-mt],
            Y[mt:-mt],
            gamma[mt:-mt],
        )

    def advance_chemistry(self, dt):
        """
        This method advances the combustion chemistry of a reacting system. It
        is only called if the "reacting" flag is set to True.
            inputs
                dt=time step
        """
        if not self.reacting:
            return

        #######################################################################
        def dydt(t, y, args):
            """
            function: dydt
            -------------------------------------------------------------------
            this function gives the source terms of a constant volume reactor
                inputs
                    dt=time step
            """
            _ = t  # Hack to silence linter
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
        Ts = self.thermoTable.get_temperature(
            self.r[indices], self.p[indices], self.Y[indices, :]
        )
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
        T = self.thermoTable.get_temperature(self.r, self.p, self.Y)
        self.gamma = self.thermoTable.get_gamma(T, self.Y)

    def advance_quasi_1d(self, dt):
        """
        This method advances the quasi-1D terms used to model area changes in
        the shock tube. The client must supply the functions dlnAdt and dlnAdx
        to the Combustor object.
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
        (r, ru, E, _) = self.primitive_to_conservative(
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
        (self.r, self.u, self.p, _) = self.conservative_to_primitive(
            r, ru, E, rY, self.gamma
        )
        T = self.thermoTable.get_temperature(self.r, self.p, self.Y)
        self.gamma = self.thermoTable.get_gamma(T, self.Y)

    def advance_boundary_layer(self, dt):
        """
        This method advances the boundary layer terms
            inputs
                dt=time step
        """

        #######################################################################
        def get_nusselt_number(Re, Pr, cf):
            """
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
        if self.DOuter is None or self.Tw is None:
            msg = "stanShock improperly initialized for boundary layer terms"
            raise Exception(msg)
        nx = len(self.x)
        if self.DInner is None:
            D = self.DOuter(self.x)
            H = D
        else:
            D = self.DOuter(self.x) - self.DInner(self.x)
            H = D
            noInsert = self.DInner(self.x) == 0.0
            H[noInsert] = D[noInsert]
        # compute gas properties
        T = self.thermoTable.get_temperature(self.r, self.p, self.Y)
        cp = self.thermoTable.get_cp(T, self.Y)
        viscosity = np.zeros(nx)
        conductivity = np.zeros(nx)
        for i, Ti in enumerate(T):
            # compute gas properties
            self.gas.TP = Ti, self.p[i]
            if self.gas.n_species > 1:
                self.gas.Y = self.Y[i, :]
            viscosity[i] = self.gas.viscosity
            conductivity[i] = self.gas.thermal_conductivity
        # compute non-dimensional numbers
        Re = abs(self.r * self.u * H / viscosity)
        Pr = cp * viscosity / conductivity
        # skin friction coefficient
        if self.cf is None:
            self.cf = SkinFriction()  # initialize the functor
        cf = self.cf(Re)
        # shear stress on wall
        shear = cf * (0.5 * self.r * self.u**2.0) * (np.sign(self.u))
        # Stanton number and heat transfer to wall
        Nu = get_nusselt_number(Re, Pr, cf)
        qloss = Nu * conductivity / H * (T - self.Tw)
        # update
        (r, ru, E, rY) = self.primitive_to_conservative(
            self.r, self.u, self.p, self.Y, self.gamma
        )
        ru -= shear * 4.0 / D * dt
        E -= qloss * 4.0 / D * dt
        (self.r, self.u, self.p, _) = self.conservative_to_primitive(
            r, ru, E, rY, self.gamma
        )
        T = self.thermoTable.get_temperature(self.r, self.p, self.Y)
        self.gamma = self.thermoTable.get_gamma(T, self.Y)

    def advance_diffusion(self, dt):
        """
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
        (r, ru, E, rY) = self.primitive_to_conservative(r, u, p, Y, gamma)
        if self.thickening is not None:
            self.F = self.thickening(self)
        # 1st stage of RK2
        rhs = self.get_viscous_flux(r, u, p, Y, gamma)
        r1 = r + dt * rhs[:, 0]
        ru1 = ru + dt * rhs[:, 1]
        E1 = E + dt * rhs[:, 2]
        rY1 = rY + dt * rhs[:, mn:]
        (r1, u1, p1, Y1) = self.conservative_to_primitive(r1, ru1, E1, rY1, gamma)
        # 2nd stage of RK2
        rhs = self.get_viscous_flux(r1, u1, p1, Y1, gamma)
        r = 0.5 * (r + r1 + dt * rhs[:, 0])
        ru = 0.5 * (ru + ru1 + dt * rhs[:, 1])
        E = 0.5 * (E + E1 + dt * rhs[:, 2])
        rY = 0.5 * (rY + rY1 + dt * rhs[:, mn:])
        (r, u, p, Y) = self.conservative_to_primitive(r, ru, E, rY, gamma)
        # update
        T0 = self.thermoTable.get_temperature(r[mt:-mt], p[mt:-mt], Y[mt:-mt])
        gamma[mt:-mt] = self.thermoTable.get_gamma(T0, Y[mt:-mt])
        (self.r, self.u, self.p, self.Y, self.gamma) = (
            r[mt:-mt],
            u[mt:-mt],
            p[mt:-mt],
            Y[mt:-mt],
            gamma[mt:-mt],
        )

    def advance_source_terms(self, dt):
        """
        This method advances the source terms in the axial direction
            inputs
                dt=time step
        """
        # initialize
        mn = self.mn
        (r, ru, E, rY) = self.primitive_to_conservative(
            self.r, self.u, self.p, self.Y, self.gamma
        )
        # 1st stage of RK2
        rhs = self.sourceTerms(r, ru, E, rY, self.gamma, self.x, self.t)
        r1 = r + dt * rhs[:, 0]
        ru1 = ru + dt * rhs[:, 1]
        E1 = E + dt * rhs[:, 2]
        rY1 = rY + dt * rhs[:, mn:]
        (r1, u1, p1, Y1) = self.conservative_to_primitive(r1, ru1, E1, rY1, self.gamma)
        # 2nd stage of RK2
        rhs = self.sourceTerms(r1, ru1, E1, rY1, self.gamma, self.x, self.t + dt)
        r = 0.5 * (r + r1 + dt * rhs[:, 0])
        ru = 0.5 * (ru + ru1 + dt * rhs[:, 1])
        E = 0.5 * (E + E1 + dt * rhs[:, 2])
        rY = 0.5 * (rY + rY1 + dt * rhs[:, mn:])
        (r, u, p, Y) = self.conservative_to_primitive(r, ru, E, rY, self.gamma)
        # update
        T0 = self.thermoTable.get_temperature(r, p, Y)
        self.gamma = self.thermoTable.get_gamma(T0, Y)
        (self.r, self.u, self.p, self.Y) = (r, u, p, Y)

    def update_probes(self, iters):
        """
        This method updates all the probes to the current value
        """

        # update probes
        for probe in self.probes:
            if iters % (probe.skipSteps + 1) == 0:
                probe.update(self)

    def update_XT_diagrams(self, iters):
        """
        This method updates all the XT Diagrams to the current value.
        """
        # update diagrams
        for XTDiagram in self.XTDiagrams:
            if iters % (XTDiagram.skipSteps + 1) == 0:
                XTDiagram.update(self)

    def pressure_rise(self, t, p, peakWidth=10):
        """
        This method attemps to determine the pressure rise based on the separation
        of the first two peaks in the logarithmic derivative of the of the
        endwall pressure. This method is the most robust when only the incident
        shock provides the only peak in the logarithmic derivative of pressure.
            inputs:
                t = time [s]
                p = pressure [pa]
                peakWidth (optional) =  # of samples to define a peak; this is
                                        also the number of the number of points
                                        used to defind the pressure rise region.
            output:
                dlnpdt: mean logaritmic slope in the pressure rise region.
                p5: the mean test pressure
        """

        from scipy import signal

        # find the logarithmic time-derivative of pressure
        lnp = np.log(p)
        dlnpdt = np.diff(lnp) / np.diff(t)
        # use a toolbox to find the peaks of the order of the peakWidth
        peakIndices = signal.find_peaks_cwt(dlnpdt, np.array([peakWidth]))
        peakIndices = np.append(peakIndices, len(t) - 2)  # add final point
        # assume the top peak is the incident shock and remove peaks that come before
        peakIndices = peakIndices[np.argsort(-np.abs(dlnpdt[peakIndices]))]
        peakIndices = [
            peakIndex for peakIndex in peakIndices if peakIndex >= peakIndices[0]
        ]
        lowerIndex = int(float(peakIndices[0] + peakIndices[1] - peakWidth) / 2)
        upperIndex = int(float(peakIndices[0] + peakIndices[1] + peakWidth) / 2)
        return np.mean(dlnpdt[lowerIndex:upperIndex]), np.mean(p[lowerIndex:upperIndex])

    def advance_simulation(self, tFinal, res_p_target=-1.0):
        """
        This method advances the simulation until the prescribed time, tFinal
            inputs
                    tFinal=final time
        """
        iters = 0
        res_p = np.inf
        while self.t < tFinal and res_p > res_p_target:
            p_old = self.p
            dt = min(tFinal - self.t, self.get_time_step())
            # advance advection and chemistry with Strang splitting
            self.advance_chemistry(dt / 2.0)
            self.advance_advection(dt)
            self.advance_chemistry(dt / 2.0)
            # advance other terms
            if self.includeDiffusion:
                self.advance_diffusion(dt)
            if self.dlnAdt is not None or self.dlnAdx is not None:
                self.advance_quasi_1d(dt)
            if self.includeBoundaryLayerTerms:
                self.advance_boundary_layer(dt)
            if self.sourceTerms is not None:
                self.advance_source_terms(dt)
            # perform other updates
            self.t += dt
            self.update_probes(iters)
            self.update_XT_diagrams(iters)
            iters += 1
            res_p = np.linalg.norm(self.p - p_old)
            if self.verbose and iters % self.outputEvery == 0:
                print(
                    f"Iteration: {iters}. Current time: {self.t}. Time step: {dt:e}. "
                    + f"Max T[K]: {self.thermoTable.get_temperature(self.r, self.p, self.Y).max()}. "
                    + f"Residual(p): {res_p}."
                )
            if (self.plotStateInterval > 0) and (iters % self.plotStateInterval == 0):
                plot_state(
                    self, f"figures/anim/test_{iters // self.plotStateInterval:05d}.png"
                )

    def optimize_driver_insert(
        self, tFinal, tradeoffParam=1.0, tTest=None, p5=None, eps=1e-4, maxIter=100
    ):
        """
        This method finds the driver insert geometry, which minimizes the
        pressure rise due to boundary layer effects while obtaining the test
        pressure. The final state of this function is the optimized state.
            inputs
                    tFinal = final time
                    tTest = the test time. This is used for normalization in the
                        cost function.
                    tradeoffParam = emphasis for experiment. A higher value
                        indicates that a correct test pressure is more valuable.
                    p5 = test pressure
                    eps = cutoff parameter for the global search. A higher value
                        indicates a tighter tolerence.
                    maxIter = maximum number of iterations
        """
        from scipy.optimize import newton
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF,
        )  # RBF is the gaussian correlation

        # Check for boundary layer terms
        if not self.includeBoundaryLayerTerms:
            self.includeBoundaryLayerTerms = True
            if self.verbose:
                print("WARNING: Boundary Layer Terms Included")

        msg = None
        if self.DOuter is None or self.dlnAdx is None:
            msg = "Driver optimization must have DOuter and dlnAdx definied"
        if self.DInner is not None:
            msg = "Driver optimization cannot have an inner diameter"
        if self.p[0] < self.p[-1]:
            msg = "Optimization routine requires the driver gas to be on the left."
        if msg is not None:
            raise Exception(msg)

        if tTest is None:
            if self.verbose:
                print(
                    "WARNING: the normalization time is not included. Setting to the simulation time."
                )
            tTest = tFinal
        if p5 is None and tradeoffParam != 0:
            if self.verbose:
                print(
                    "WARNING: the target test pressure is not provided. Determining p5 via normal shock relations"
                )
            g1, g4 = self.gamma[-1], self.gamma[0]
            p4op1 = self.p[0] / self.p[-1]
            r4or1 = self.r[0] / self.r[-1]
            a4oa1 = np.sqrt(g4 / g1 * p4op1 / r4or1)

            def res(Ms1):
                p4op1 - (1.0 + 2.0 * g1 / (g1 + 1.0) * (Ms1**2.0 - 1.0)) * (
                    1.0 - (g4 - 1.0) / (g4 + 1.0) / a4oa1 * (Ms1 - 1.0 / Ms1)
                ) ** (-2.0 * g4 / (g4 - 1.0))

            Ms1 = newton(res, 2.0)
            p5op1 = ((2.0 * g1 * Ms1**2.0 - (g1 - 1.0)) / (g1 + 1.0)) * (
                (-2.0 * (g1 - 1.0) + Ms1**2.0 * (3.0 * g1 - 1.0))
                / (2.0 + Ms1**2.0 * (g1 - 1.0))
            )
            p5 = p5op1 * self.p[-1]
        # Get initial state for reinitialization
        rInitial = np.copy(self.r)
        uInitial = np.copy(self.u)
        pInitial = np.copy(self.p)
        YInitial = np.copy(self.Y)
        gammaInitial = np.copy(self.gamma)
        dlnAdxInitial = self.dlnAdx

        def dDOuterdx(x):
            return (
                self.DOuter(x) / 2.0 * dlnAdxInitial(x, 0.0)
            )  # assume temporally constant area

        # Determine geometry from pressure
        dpAbs = np.abs(pInitial[1:] - pInitial[:-1])
        xShock = max(zip(dpAbs, self.x[1:]))[
            1
        ]  # maximum pressure gradient corresponds to shock
        (xMin, xMax, probeLocation) = (self.x[0], xShock, self.x[-1])
        LMax = xMax - xMin  # maximum length of constrained optimization
        DMax = min(
            self.DOuter(np.linspace(xMin, xMax))
        )  # maximum diameter of constrained optimization
        smoothingLength = 10 * self.dx
        if LMax <= smoothingLength:
            msg = "This calculation will likely be unstable. Refine the grid"
            raise Exception(msg)

        def alphaMax(L):
            return 1.0 - smoothingLength / L

        (LMin, DMin, alphaMin) = (
            smoothingLength,
            0.0,
            0.0,
        )  # no driver bound (no negative lengths)

        # calculate a smoothing length for numerical stability
        #######################################################################
        def optimization_function(design):
            """
            This function solves the shocktube problem and returns the absolute pressure rise. The insert geometry is assumed to
            vary linearly in area
                inputs:
                    design=tuple with
                        (total length of insert, maximum diameter of insert, ratio of constant portion of insert to entire insert)

            """
            # determine the geometry
            (LInsert, DInsert, alpha) = design
            xIns0, xIns1 = xMin + LInsert * alpha, xMin + LInsert
            AIns0 = np.pi * DInsert**2.0 / 4.0

            def AInsert(x):
                AIns = np.zeros_like(x)
                inds = np.logical_and(x >= xIns0, x < xIns1)
                AIns[inds] = AIns0 * (1.0 - (x[inds] - xIns0) / (xIns1 - xIns0))
                AIns[x < xIns0] = AIns0
                return AIns

            def dAInsertdx(x):
                dAInsdx = np.zeros_like(x)
                inds = np.logical_and(x >= xIns0, x < xIns1)
                dAInsdx[inds] = -AIns0 / (xIns1 - xIns0)
                return dAInsdx

            def DInner(x):
                return np.sqrt(4.0 * AInsert(x) / np.pi)

            def dDInnerdx(x):
                dDIndx = np.zeros_like(x)
                inds = np.logical_and(x >= xIns0, x < xIns1)
                dDIndx[inds] = (
                    0.5
                    * (4.0 * AInsert(x[inds]) / np.pi) ** -0.5
                    * (4.0 * dAInsertdx(x[inds]) / np.pi)
                )
                return dDIndx

            def A(x):
                return np.pi / 4.0 * (self.DOuter(x) ** 2.0 - DInner(x) ** 2.0)

            def dAdx(x):
                return (
                    np.pi
                    / 2.0
                    * (self.DOuter(x) * dDOuterdx(x) - DInner(x) * dDInnerdx(x))
                )

            # initialize (may be at a previous state in the optimization)
            self.dlnAdx = lambda x, t: dAdx(x) / A(x)
            self.DInner = DInner
            self.r = np.copy(rInitial)
            self.u = np.copy(uInitial)
            self.p = np.copy(pInitial)
            self.Y = np.copy(YInitial)
            self.gamma = np.copy(gammaInitial)
            # delete previous probes and create an endwall probe
            self.probes = [
                Probe(self, probeLocation, skipSteps=0, probeName="endwall probe"),
            ]
            # solve
            if self.verbose:
                print(
                    f"Solving Optimization. Iteration={self.optimizationIteration}, L={LInsert:.3f}, D={DInsert:.3f}, alpha={alpha:.3f}"
                )
            self.t = 0.0
            self.advance_simulation(tFinal)
            self.optimizationIteration += 1
            # return
            dlnpdt, p5Act = self.pressure_rise(
                np.array(self.probes[0].t), np.array(self.probes[0].p)
            )
            if self.verbose:
                print(
                    f"Finished with optimization iteration. dlnpdt={dlnpdt:f}, p5={p5Act:f}"
                )
            return (dlnpdt * tTest) ** 2.0 + tradeoffParam * (p5Act / p5 - 1.0) ** 2.0

        #######################################################################
        def midpoint_vector(xMin, xMax, nMidpoints):
            """
            This function returns a vector to sample from. The vector is
            uniformly space and is non-inclusive of the bounds
                inputs:
                    xMin=lower bound
                    xMax=upper bound
                    nMidpoints=number of sampling locations
                output:
                    sample vector
                    spacing (dx)
            """
            dx = (xMax - xMin) / float(nMidpoints)
            return xMin + dx / 2.0 + dx * np.arange(0, nMidpoints)

        #######################################################################
        # develop initial grid of points
        nGrid = 3
        self.designs = []
        for L in midpoint_vector(LMin, LMax, nGrid):
            for D in midpoint_vector(DMin, DMax, nGrid):
                for a in midpoint_vector(alphaMin, alphaMax(L), nGrid):
                    self.designs.append((L, D, a))
        # solve for each grid point on the initial parameter space
        nDesigns = len(self.designs)
        self.yOpt = []  # evaluated points
        if self.verbose:
            print(f"Solving initial grid of {nDesigns} points.")
        for _iDesign, design in enumerate(self.designs):
            self.yOpt.append(optimization_function(design))
        # initialize the Gaussian Random Process as a surrogate
        kernel = 1.0 * RBF(
            length_scale=1.0, length_scale_bounds=(1e-1, 10.0)
        )  # iniitialize
        gp = GaussianProcessRegressor(kernel=kernel)
        # determine the grid to search over with the surrogate model
        nHat = 30
        XHat = []
        for L in midpoint_vector(LMin, LMax, nHat):
            for D in midpoint_vector(DMin, DMax, nHat):
                for a in midpoint_vector(alphaMin, alphaMax(L), nHat):
                    XHat.append((L, D, a))
        XHat = np.array(XHat)
        # iterate until optimum is found
        ymin = min(self.yOpt)
        if self.verbose:
            print("Finding Optimum.")
        minImprovement = eps / 10.0
        maxImprovement = minImprovement + 1.0
        while (
            ymin > eps
            and self.optimizationIteration < maxIter
            and maxImprovement > minImprovement
        ):
            # fit the GP
            X = np.array(self.designs)
            gp.fit(X, np.array(self.yOpt))
            YHat = gp.predict(XHat)
            # find the improvement
            parameters = gp.kernel_.get_params()
            sigmaSqrd = parameters["k1__constant_value"]
            R = gp.kernel_.k2  # correlation matrix
            RInv = np.linalg.inv(R(X))
            ExpImprovements = np.zeros(XHat.shape[0])
            ymin = min(self.yOpt)
            rs = R(X, Y=XHat)
            r = np.sum(RInv.dot(rs) * rs, axis=0)
            sSqrds = sigmaSqrd * (1.0 - r)
            ind = sSqrds <= 0
            ExpImprovements[ind] = np.maximum(
                ymin - YHat[ind], np.zeros_like(YHat[ind])
            )
            ind = sSqrds > 0
            s = np.sqrt(sSqrds[ind])
            T = (ymin - YHat[ind]) / s
            ExpImprovements[ind] = s * (T * norm.cdf(T) + norm.pdf(T))
            # find the maximum improvement and compute the new datum
            maxImprovement = np.max(ExpImprovements)
            self.designs.append(XHat[np.argmax(ExpImprovements), :])
            self.yOpt.append(optimization_function(self.designs[-1]))
            if self.verbose:
                print(
                    f"Minimum of current iteration: {ymin:f}. Expected improvement of the next iteration: {maxImprovement:f}"
                )
        if self.optimizationIteration >= maxIter and self.verbose:
            print("No minimum found within tolerance.")
        elif maxImprovement <= minImprovement and self.verbose:
            print(
                "Search stopped due to no sample points found yielding enough improvement."
            )
        elif self.verbose:
            print("Minimum Found. Setting to minimum state.")
        optimization_function(self.designs[np.argmin(self.yOpt)])
