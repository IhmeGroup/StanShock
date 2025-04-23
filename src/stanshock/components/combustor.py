from __future__ import annotations

import cantera as ct
import numpy as np

from stanshock.models.boundary_layer import BoundaryLayer
from stanshock.numerics.face_extrapolation import FifthOrderWeno, FirstOrder
from stanshock.numerics.inviscid_flux import InviscidFlux, hllc_flux
from stanshock.numerics.viscous_flux import ViscousFlux
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.processing.initialize import (
    initialize_constant,
    initialize_diffuse_interface,
    initialize_riemann_problem,
)
from stanshock.processing.plot import plot_state


class Combustor:
    """
    This is a class defined to encapsulate the data and methods used for the
    1D gasdynamics solver.
    """

    def __init__(
        self,
        physics: FluidPhysics,
        n: int = 10,
        **kwargs,
    ):
        """
        initialization of the object with default values. The keyword arguments
        allow the user to initialize the state
        """
        # initialize the class
        self.mt = 3  # number of ghost nodes
        self.mn = 3  # number of 1D Euler equations

        self.cfl = 1.0  # stability condition
        self.dx = 1.0  # grid spacing
        self.n = n  # grid size
        self.boundaryConditions = ["outflow", "outflow"]
        self.x = np.linspace(0.0, self.dx * (self.n - 1), self.n)
        self.F = np.ones(self.n)  # thickening
        self.t = 0.0  # time
        self.verbose = True  # console output switch
        self.outputEvery = (
            1  # number of iterations of simulation advancement between logging updates
        )
        self.h = None  # height of the channel
        self.w = None  # width of the channel
        self.DInner = (
            None  # Inner diameter of the shock tube as a function of x (needed for BL)
        )
        self.DOuter = (
            None  # Outer diameter of the shock tube as a function of x (needed for BL)
        )
        self.dlnAdt = (
            None  # area of the shock tube as a function of time (needed for quasi-1D)
        )
        self.dlnAdx = (
            None  # area of the shock tube as a function of x (needed for quasi-1D)
        )
        self.includeBoundaryLayerTerms = False  # flag to include boundary layer terms
        self.wall_temperature = None  # wall temperature (needed for BL)
        self.sourceTerms = None  # source term function
        self.injector = None  # injector model
        self.fluxFunction = hllc_flux
        self.initialization = None  # initialization options
        self.probes = []  # list of probe objects
        self.XTDiagrams = []  # list of XT diagram objects
        self.skin_friction_coefficient = None  # skin friction functor
        self.optimizationIteration = 0  # counter to keep track of optimization
        self.physics = physics  # Model handling all fluid property evaluations
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

        # Ensure the mesh parameters are consistent
        self.n = len(self.x)
        self.dx = self.x[1] - self.x[0]

        # set the number of scalars
        self.n_scalars = self.physics.n_scalars
        if not self.physics.is_flamelet and self.injector is not None:
            msg = "JIC injector model requires FPVTable physics."
            raise Exception(msg)

        # initialize the state
        if self.initialization is None:
            msg = "No initialization method selected"
            raise Exception(msg)
        if self.initialization[0].lower() == "constant":
            self.state = initialize_constant(self, *self.initialization[1:])
        elif self.initialization[0].lower() == "riemann":
            self.state = initialize_riemann_problem(self, *self.initialization[1:])
        elif self.initialization[0].lower() == "diffuse_interface":
            self.state = initialize_diffuse_interface(self, *self.initialization[1:])

        # Initialize the key physics
        self.inviscid_flux_source = InviscidFlux(
            face_extrapolator=FifthOrderWeno(),
            riemann_solver=self.fluxFunction,
            dx=self.dx,
        )

        if self.includeDiffusion:
            self.viscous_flux_source = ViscousFlux(
                face_extrapolator=FirstOrder(),
                dx=self.dx,
            )

        # Compute the hydraulic diameter and characteristic length scale
        if self.includeBoundaryLayerTerms:
            if self.h is not None and self.w is not None:
                self.hydraulic_diameter = 2 * self.h * self.w / (self.h + self.w)
                self.characteristic_length = self.hydraulic_diameter.copy()
            else:
                self.hydraulic_diameter = self.DOuter(self.x)
                self.characteristic_length = self.hydraulic_diameter.copy()

                if self.DInner is not None:
                    self.hydraulic_diameter -= self.DInner(self.x)
                    self.characteristic_length = 0.5 * self.hydraulic_diameter

                    noInsert = self.DInner(self.x) == 0.0
                    self.characteristic_length[noInsert] = self.hydraulic_diameter[
                        noInsert
                    ]

            # Initialize the boundary layer source terms
            self.boundary_layer_source = BoundaryLayer(
                hydraulic_diameter=self.hydraulic_diameter,
                characteristic_length=self.characteristic_length,
                wall_temperature=self.wall_temperature,
                skin_friction_coefficient=self.skin_friction_coefficient,
            )

    def get_wave_speed(self):
        """
        This method determines the absolute maximum of the wave speed
            outputs:
                speed of acoustic wave
        """
        return abs(self.state.velocity) + self.physics.get_sound_speed(self.state)

    def get_time_step(self):
        """
        This method determines the maximal timestep in accord with the CFL
        condition
            outputs:
                timestep
        """
        local_timescale = self.dx / self.get_wave_speed()
        if self.includeDiffusion:
            mu = self.physics.get_mu(self.state)
            nu = mu / self.state.density
            alpha = self.physics.get_thermal_diffusivity(self.state) * self.F
            diff = (
                np.max(self.physics.get_mass_diffusivity(self.state), axis=1) * self.F
            )
            viscous_timescale = (
                0.5 * self.dx**2.0 / np.maximum(4.0 / 3.0 * nu, np.maximum(alpha, diff))
            )
            local_timescale = np.minimum(local_timescale, viscous_timescale)
        return self.cfl * min(local_timescale)

    def apply_boundary_conditions(self, face_states: FluidState):
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
        rLR = face_states.density
        uLR = face_states.velocity
        pLR = face_states.pressure
        YLR = face_states.composition

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
        return face_states

    def advance_advection(self, dt):
        """
        This method advances the advection terms by the prescribed timestep.
        The advection terms are integrated using RK3.
            inputs
                dt=time step
        """
        # Add ghost layers to the fluid state
        face_extrapolator = self.inviscid_flux_source.face_extrapolator
        mt = face_extrapolator.mt
        state = face_extrapolator.add_ghost_layers(self.state)
        y = self.physics.primitive_to_conservative(state)
        gamma_star = state.gamma  # Double-flux gamma* held constant over time step

        # 1st stage of RK3
        face_states = face_extrapolator(state)
        face_states = self.apply_boundary_conditions(face_states)
        dydt = self.inviscid_flux_source(self.t, face_states, self.physics)
        y1 = y.copy()
        y1[mt:-mt] += dt * dydt
        state1 = self.physics.conservative_to_primitive(y1, gamma_star)
        state1.gamma = gamma_star

        # 2nd stage of RK3
        face_states = face_extrapolator(state1)
        face_states = self.apply_boundary_conditions(face_states)
        dydt = self.inviscid_flux_source(self.t, face_states, self.physics)
        y2 = 0.75 * y + 0.25 * y1
        y2[mt:-mt] += 0.25 * dt * dydt
        state2 = self.physics.conservative_to_primitive(y2, gamma_star)
        state2.gamma = gamma_star

        # 3rd stage of RK3
        face_states = face_extrapolator(state2)
        face_states = self.apply_boundary_conditions(face_states)
        dydt = self.inviscid_flux_source(self.t, face_states, gamma_star)
        y = (1.0 / 3.0) * y[mt:-mt] + (2.0 / 3.0) * y2[mt:-mt] + (2.0 / 3.0) * dt * dydt

        # Remove ghost layers and update gamma
        self.state = self.physics.conservative_to_primitive(y, gamma_star[mt:-mt])
        self.state.temperature = self.physics.get_temperature(self.state)
        self.state.gamma = self.physics.get_gamma(self.state)

    def advance_diffusion(self, dt):
        """
        This method advances the diffusion terms in the axial direction
            inputs
                dt=time step
        """
        # Add ghost layers to the fluid state
        face_extrapolator = self.viscous_flux_source.face_extrapolator
        mt = face_extrapolator.mt
        state = face_extrapolator.add_ghost_layers(self.state)
        y = self.physics.primitive_to_conservative(state)
        gamma_star = state.gamma  # Double-flux gamma* held constant over time step

        if self.thickening is not None:
            self.F = self.thickening(self)

            # No gradient in F at boundary
            self.viscous_flux_source.F = np.pad(self.F, mt, mode="edge")

        # 1st stage of RK2
        face_states = face_extrapolator(state)
        face_states = self.apply_boundary_conditions(face_states)
        dydt = self.viscous_flux_source(self.t, face_states, self.physics)
        y1 = y.copy()
        y1[mt:-mt] += dt * dydt
        state1 = self.physics.conservative_to_primitive(y1, gamma_star)
        state1.gamma = gamma_star

        # 2nd stage of RK2
        face_states = face_extrapolator(state1)
        face_states = self.apply_boundary_conditions(face_states)
        dydt = self.viscous_flux_source(self.t, face_states, self.physics)
        y = 0.5 * (y[mt:-mt] + y1[mt:-mt] + dt * dydt)

        # Remove ghost layers and update gamma
        self.state = self.physics.conservative_to_primitive(y, gamma_star[mt:-mt])
        self.state.temperature = self.physics.get_temperature(self.state)
        self.state.gamma = self.physics.get_gamma(self.state)

    def advance_chemistry(self, dt):
        """
        This method advances the combustion chemistry of a reacting system. It
        is only called if the "reacting" flag is set to True.
            inputs
                dt=time step
        """
        if not self.reacting:
            return
        if self.physics.is_flamelet:
            self.advance_chemistry_FPV(dt)
        else:
            self.advance_chemistry_FRC(dt)

    def advance_chemistry_FPV(self, dt):
        """
        This method advances the combustion chemistry of a reacting system using
        the flamelet progress variable approach. It is only called if the "reacting"
        flag is set to True.
            inputs
                dt=time step
        """
        # Using mixed-is-burned (MIB)
        # (r,ru,E,rY)=self.primitiveToConservative(self.r,self.u,self.p,self.Y,self.state.gamma)
        # Z = rY[:, 0] / r
        # Q = np.zeros(self.n)
        # C = rY[:, 1] / r
        # L = self.physics.get_normalized_progress_variable(Z, C)
        # e_chem0 = r * self.physics.lookup('E0_CHEM', Z, Q, L)
        # C1, e_chem1 = self.injector.get_MIB_profiles()
        # e_chem1 *= r
        # E1 = E + e_chem0 - e_chem1
        # rY1 = rY
        # rY1[:, 1] = r * C1
        # (r,u,p,Y)=self.conservativeToPrimitive(r,ru,E1,rY1,self.state.gamma)

        # Using FPV
        # initialize
        y = self.physics.primitive_to_conservative(self.state)
        (r, rZ, rC) = y[:, 0], y[:, 3], y[:, 4]
        Z = rZ / r
        C = rC / r
        Q = np.zeros(self.n)
        L = self.physics.get_normalized_progress_variable(Z, C)
        e_chem0 = r * self.physics.lookup_direct("E0_CHEM", Z, Q, L)

        # 1st stage of RK2
        omegaC = r * self.injector.get_chemical_sources(Z, C)
        y1 = y.copy()
        y1[:, 4] += dt * omegaC
        C1 = y1[:, 4] / r
        L1 = self.physics.get_normalized_progress_variable(Z, C1)
        e_chem1 = r * self.physics.lookup_direct("E0_CHEM", Z, Q, L1)
        y1[:, 2] += e_chem0 - e_chem1
        state1 = self.physics.conservative_to_primitive(y1, self.state.gamma)
        state1.gamma = self.state.gamma

        # 2nd stage of RK2
        self.physics.set_state(state1)
        omegaC1 = state1.density * self.injector.get_chemical_sources(
            state1.Z, state1.C
        )
        rC = y[:, 4] = 0.5 * (y[:, 4] + y1[:, 4] + dt * omegaC1)
        L = self.physics.get_normalized_progress_variable(Z, rC / r)
        e_chem2 = r * self.physics.lookup_direct("E0_CHEM", Z, Q, L)
        y[:, 2] += e_chem0 - e_chem2
        self.state = self.physics.conservative_to_primitive(y, self.state.gamma)

        # update properties
        self.state.temperature = self.physics.get_temperature(self.state)
        self.state.gamma = self.physics.get_gamma(self.state)

    def advance_chemistry_FRC(self, dt):
        """
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
            _ = t  # Hack to silence linter
            # unpack the input
            r = args[0]
            F = args[1]
            Y = y[:-1]
            T = y[-1]
            # set the state for the gas object
            self.physics.gas.TDY = T, r, Y
            # gas properties
            cv = self.physics.gas.cv_mass
            W = self.physics.gas.molecular_weights
            wHatDot = self.physics.gas.net_production_rates  # kmol/m^3.s
            wDot = wHatDot * W  # kg/m^3.s
            eRT = self.physics.gas.standard_int_energies_RT
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
        state_temp = FluidState(
            shape=(len(indices),),
            density=self.state.density[indices].copy(),
            pressure=self.state.pressure[indices].copy(),
            composition=self.state.composition[indices, :].copy(),
        )
        state_temp.temperature = Ts = self.physics.get_temperature(state_temp)
        state_temp.pressure = None
        state_temp._cache_valid = False

        # initialize integrator
        y0 = np.zeros(self.physics.n_scalars + 1)
        integrator = integrate.ode(dydt).set_integrator("lsoda")
        for TIndex, k in enumerate(indices):
            # initialize
            y0[:-1] = self.state.composition[k, :]
            y0[-1] = Ts[TIndex]
            args = [self.state.density[k], self.F[k]]
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
            state_temp.composition[TIndex, :] = Y
            state_temp.temperature[TIndex] = integrator.y[-1]

        # update state
        self.state.pressure[indices] = self.physics.get_pressure(state_temp)
        self.state.composition[indices, :] = state_temp.composition
        self.state.temperature = None
        self.state._cache_valid = False
        self.state.gamma = self.physics.get_gamma(self.state)

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
        y = self.physics.primitive_to_conservative(self.state)

        # determine the indices
        iIn = []
        eIn = np.arange(self.x.shape[0])
        if self.dlnAdt is not None:
            dlnAdt = self.dlnAdt(self.x, self.t)
            iIn = np.where(dlnAdt != 0.0)
            eIn = np.where(dlnAdt == 0.0)

        # integrate implicitly
        for i in iIn:
            # initialize
            y0[:] = y[i, :3]
            args = np.array([self.x[i]]), self.state.gamma[i]
            integrator.set_initial_value(y0, self.t)
            integrator.set_f_params(args)
            # solve
            integrator.integrate(self.t + dt)
            # update
            y[i, :3] = integrator.y

        # integrate explicitly
        rhs = np.zeros((eIn.shape[0], mn))
        if self.dlnAdt is not None:
            dlnAdt = self.dlnAdt(self.x, self.t)[eIn]
            rhs -= y[eIn, :3] * dlnAdt

        if self.dlnAdx is not None:
            dlnAdx = self.dlnAdx(self.x, self.t)[eIn]
            rhs[:, 0] -= y[eIn, 1] * dlnAdx
            rhs[:, 1] -= (y[eIn, 1] ** 2.0 / y[eIn, 0]) * dlnAdx
            rhs[:, 2] -= (
                self.state.velocity[eIn] * (y[eIn, 2] + self.state.pressure[eIn])
            ) * dlnAdx

        # update
        y[eIn, :3] += dt * rhs
        y[:, 3:] = y[:, [0]] * self.state.composition
        self.state = self.physics.conservative_to_primitive(y, self.state.gamma)
        self.state.temperature = self.physics.get_temperature(self.state)
        self.state.gamma = self.physics.get_gamma(self.state)

    def advance_boundary_layer(self, dt):
        """
        This method advances the boundary layer terms
            inputs
                dt=time step
        """
        y = self.physics.primitive_to_conservative(self.state)

        # Get RHS
        dydt = self.boundary_layer_source(self.t, self.state, self.physics)

        # Single forward-Euler step
        y += dydt * dt

        # Update
        self.state = self.physics.conservative_to_primitive(y, self.state.gamma)
        self.state.temperature = self.physics.get_temperature(self.state)
        self.state.gamma = self.physics.get_gamma(self.state)

    def advance_source_terms(self, dt):
        """
        This method advances the source terms in the axial direction
            inputs
                dt=time step
        """
        # initialize
        y = self.physics.primitive_to_conservative(self.state)

        # 1st stage of RK2
        dydt = self.sourceTerms(y, self.state.gamma, self.x, self.t)
        y1 = y + dt * dydt
        # state1 = self.physics.conservative_to_primitive(y1, self.state.gamma)

        # 2nd stage of RK2
        dydt = self.sourceTerms(y1, self.state.gamma, self.x, self.t + dt)
        y = 0.5 * (y + y1 + dt * dydt)
        self.state = self.physics.conservative_to_primitive(y, self.state.gamma)

        # update
        self.state.temperature = self.physics.get_temperature(self.state)
        self.state.gamma = self.physics.get_gamma(self.state)

    def advance_injector(self, dt):
        """
        This method advances the source terms from the injector using the
        jet-in-crossflow model.
            inputs
                dt=time step
        """
        # initialize
        mn = self.mn
        y = self.physics.primitive_to_conservative(self.state)
        (r, ru, E, rY) = y[:, 0], y[:, 1], y[:, 2], y[:, mn:]
        self.injector.update_fluid_tip_positions(dt, self.t, self.state.velocity)

        # 1st stage of RK2
        dydt = self.injector.get_injector_sources(
            r, ru, E, rY[:, 0], rY[:, 1], self.state.gamma, self.t
        )
        y1 = y + dt * dydt
        # state1 = self.physics.conservative_to_primitive(y1, self.state.gamma)

        # 2nd stage of RK2
        (r1, ru1, E1, rY1) = y1[:, 0], y1[:, 1], y1[:, 2], y1[:, mn:]
        dydt = self.injector.get_injector_sources(
            r1, ru1, E1, rY1[:, 0], rY1[:, 1], self.state.gamma, self.t + dt
        )
        y = 0.5 * (y + y1 + dt * dydt)

        # update
        self.state = self.physics.conservative_to_primitive(y, self.state.gamma)
        self.state.temperature = self.physics.get_temperature(self.state)
        self.state.gamma = self.physics.get_gamma(self.state)

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

    def advance_simulation(self, tFinal, res_p_target=-1.0):
        """
        This method advances the simulation until the prescribed time, tFinal
            inputs
                    tFinal=final time
        """
        iters = 0
        res_p = np.inf
        while self.t < tFinal and res_p > res_p_target:
            p_old = self.state.pressure
            dt = min(tFinal - self.t, self.get_time_step())
            # advance advection and chemistry
            if self.physics.is_flamelet:
                self.advance_advection(dt)
                self.advance_chemistry(dt)
            else:
                # use Strang splitting
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
            if self.injector is not None:
                self.advance_injector(dt)
            # perform other updates
            self.t += dt
            self.update_probes(iters)
            self.update_XT_diagrams(iters)
            iters += 1
            res_p = np.linalg.norm(self.state.pressure - p_old)
            if self.verbose and iters % self.outputEvery == 0:
                print(
                    f"Iteration: {iters}. Current time: {self.t}. Time step: {dt:e}. "
                    + f"Max T[K]: {self.physics.get_temperature(self.state).max()}. "
                    + f"Residual(p): {res_p}."
                )
            if (self.plotStateInterval > 0) and (iters % self.plotStateInterval == 0):
                plot_state(
                    self, f"figures/anim/test_{iters // self.plotStateInterval:05d}.png"
                )
