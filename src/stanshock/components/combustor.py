from __future__ import annotations

import cantera as ct
import numpy as np

from stanshock.models.area_change import AreaChange
from stanshock.models.boundary_layer import BoundaryLayer
from stanshock.numerics.boundary_conditions import (
    BCNamesType,
    BCType,
    BoundaryConditions,
    set_boundary_conditions,
)
from stanshock.numerics.face_extrapolation import (
    FaceExtrapolator,
    FifthOrderWeno,
    FirstOrder,
)
from stanshock.numerics.gradient import CentralDifference
from stanshock.numerics.inviscid_flux import InviscidFlux, RiemannSolver, hllc_flux
from stanshock.numerics.viscous_flux import ViscousFlux
from stanshock.physics.flamelet import FPVTable
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.processing.initialize import (
    initialize_constant,
    initialize_diffuse_interface,
    initialize_riemann_problem,
)
from stanshock.processing.plot import plot_state
from stanshock.system.backend import Array, Index
from stanshock.system.base import RightHandSide
from stanshock.system.geometry import Geometry, initialize_geometry


class Combustor:
    """
    This is a class defined to encapsulate the data and methods used for the
    1D gasdynamics solver.
    """

    def __init__(
        self,
        physics: FluidPhysics,
        n: int = 10,
        geometry: Geometry | None = None,
        **kwargs,
    ):
        """
        initialization of the object with default values. The keyword arguments
        allow the user to initialize the state
        """
        # initialize the class
        self.cfl = 1.0  # stability condition
        self.dx = 1.0  # grid spacing
        self.n = n  # grid size
        self.boundary_conditions: BoundaryConditions | list[BCNamesType | BCType] = [
            "outflow",
            "outflow",
        ]
        self.x: Array = np.linspace(
            0.0, self.dx * (self.n - 1), self.n, dtype=np.float64
        )
        self.F = np.ones(self.n)  # thickening
        self.t = 0.0  # time
        self.verbose = True  # console output switch
        self.output_every = (
            1  # number of iterations of simulation advancement between logging updates
        )
        self.area_change: RightHandSide | None = None
        self.include_boundary_layer = False  # flag to include boundary layer terms
        self.wall_temperature = None  # wall temperature (needed for BL)
        self.source_terms: RightHandSide | None = None  # source term function
        self.injector = None  # injector model
        self.flux_function: RiemannSolver = hllc_flux
        self.inviscid_face_extrapolator: FaceExtrapolator = FifthOrderWeno
        self.viscous_face_extrapolator: FaceExtrapolator = FirstOrder
        self.initialization = None  # initialization options
        self.probes = []  # list of probe objects
        self.xt_diagrams = []  # list of XT diagram objects
        self.skin_friction_coefficient = None  # skin friction functor
        self.optimization_iteration = 0  # counter to keep track of optimization
        self.physics = physics  # Model handling all fluid property evaluations
        self.reacting = False  # flag to solver about whether to solve source terms
        self.in_reacting_region = (
            lambda _x, _t: True
        )  # the reacting region of the shock tube.
        self.include_diffusion = False  # exclude diffusion
        self.thickening = None  # thickening function
        self.plot_state_interval = -1  # plot the state every n iterations
        # overwrite the default data
        for key, item in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key] = item

        # Initialize the geometry of the domain
        if geometry is None:
            kwargs.pop("x")
            self.geometry: Geometry = initialize_geometry(x=self.x, **kwargs)
        else:
            self.geometry = geometry

        # Add area-change related source terms
        if self.geometry.dlnA_dt is not None or self.geometry.dlnA_dx is not None:
            self.area_change = AreaChange(geometry=self.geometry)

        # set the number of scalars
        self.n_scalars = self.physics.n_scalars
        if not self.physics.is_flamelet and self.injector is not None:
            msg = "JIC injector model requires FPVTable physics."
            raise Exception(msg)

        # Determine the number of ghost layers required by the spatial scheme
        self.n_ghost_layers: int = self.inviscid_face_extrapolator.minimum_ghost_layers
        if self.include_diffusion:
            self.n_ghost_layers = max(
                self.n_ghost_layers, self.viscous_face_extrapolator.minimum_ghost_layers
            )

        # Set up boundary conditions
        self.boundary_conditions: BoundaryConditions = set_boundary_conditions(
            self.boundary_conditions, self.n_ghost_layers
        )

        # initialize the state
        if self.initialization is None:
            msg = "No initialization method selected"
            raise Exception(msg)
        if self.initialization[0].lower() == "constant":
            self.state = initialize_constant(
                self.geometry, self.physics, *self.initialization[1:]
            )
        elif self.initialization[0].lower() == "riemann":
            self.state = initialize_riemann_problem(
                self.geometry, self.physics, *self.initialization[1:]
            )
        elif self.initialization[0].lower() == "diffuse_interface":
            self.state = initialize_diffuse_interface(
                self.geometry, self.physics, *self.initialization[1:]
            )

        # Initialize the key physics
        self.inviscid_flux = InviscidFlux(
            face_extrapolator=self.inviscid_face_extrapolator(
                n_scalars_rho_sum=self.physics.n_scalars_rho_sum,
                n_ghost_layers=self.n_ghost_layers,
            ),
            boundary_conditions=self.boundary_conditions,
            riemann_solver=self.flux_function,
            dx=self.geometry.dx,
        )

        if self.include_diffusion:
            self.viscous_flux = ViscousFlux(
                boundary_conditions=self.boundary_conditions,
                face_extrapolator=self.viscous_face_extrapolator(
                    n_scalars_rho_sum=self.physics.n_scalars_rho_sum,
                    n_ghost_layers=self.n_ghost_layers,
                ),
                geometry=self.geometry,
                gradient=CentralDifference(n_ghost_layers=self.n_ghost_layers),
            )

        if self.include_boundary_layer:
            # Initialize the boundary layer source terms
            self.boundary_layer = BoundaryLayer(
                geometry=self.geometry,
                wall_temperature=self.wall_temperature,
                skin_friction_coefficient=self.skin_friction_coefficient,
            )

        # Extend the domain to include the ghost layers
        self.state = self.inviscid_flux.face_extrapolator.add_ghost_layers(self.state)
        self.idx_cells: Index = np.s_[self.n_ghost_layers : -self.n_ghost_layers]
        self.F = np.pad(self.F, self.n_ghost_layers, mode="edge")

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
        local_timescale = self.geometry.dx / self.get_wave_speed()
        if self.include_diffusion:
            mu = self.physics.get_mu(self.state)
            nu = mu / self.state.density
            alpha = self.physics.get_thermal_diffusivity(self.state) * self.F
            diff = (
                np.max(self.physics.get_mass_diffusivity(self.state), axis=1) * self.F
            )
            viscous_timescale = (
                0.5
                * self.geometry.dx**2.0
                / np.maximum(4.0 / 3.0 * nu, np.maximum(alpha, diff))
            )
            local_timescale = np.minimum(local_timescale, viscous_timescale)
        return self.cfl * min(local_timescale)

    def advance_advection(self, dt):
        """
        This method advances the advection terms by the prescribed timestep.
        The advection terms are integrated using RK3.
            inputs
                dt=time step
        """
        y = self.physics.primitive_to_conservative(self.state)
        gamma_star = self.state.gamma  # Double-flux gamma* held constant over time step

        # 1st stage of RK3
        dydt = self.inviscid_flux.source(self.t, y, self.physics, gamma_star)
        y1 = y.copy()
        y1[self.idx_cells] += dt * dydt

        # 2nd stage of RK3
        dydt = self.inviscid_flux.source(self.t, y1, self.physics, gamma_star)
        y2 = 0.75 * y + 0.25 * y1
        y2[self.idx_cells] += 0.25 * dt * dydt

        # 3rd stage of RK3
        dydt = self.inviscid_flux.source(self.t, y2, self.physics, gamma_star)

        y[self.idx_cells] = (
            (1.0 / 3.0) * y[self.idx_cells]
            + (2.0 / 3.0) * y2[self.idx_cells]
            + (2.0 / 3.0) * dt * dydt
        )

        # Remove ghost layers and update gamma
        self.state = self.physics.conservative_to_primitive(y, gamma_star)
        self.state.temperature = self.physics.get_temperature(self.state)
        self.state.gamma = self.physics.get_gamma(self.state)

    def advance_diffusion(self, dt):
        """
        This method advances the diffusion terms in the axial direction
            inputs
                dt=time step
        """
        mt: int = self.n_ghost_layers
        y = self.physics.primitive_to_conservative(self.state)
        gamma_star = self.state.gamma  # Double-flux gamma* held constant over time step

        if self.thickening is not None:
            self.F = self.thickening(self)

            # No gradient in F at boundary
            self.viscous_flux.F = np.pad(self.F, mt, mode="edge")

        # 1st stage of RK2
        dydt = self.viscous_flux.source(self.t, y, self.physics, gamma_star)
        y1 = y.copy()
        y1[self.idx_cells] += dt * dydt

        # 2nd stage of RK2
        dydt = self.viscous_flux.source(self.t, y1, self.physics, gamma_star)
        y[self.idx_cells] = 0.5 * (y[self.idx_cells] + y1[self.idx_cells] + dt * dydt)

        # Remove ghost layers and update gamma
        self.state = self.physics.conservative_to_primitive(y, gamma_star)
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
        # initialize
        gamma_star = self.state.gamma
        y = self.physics.primitive_to_conservative(self.state)
        (r, rZ, rC) = y[self.idx_cells, 2], y[self.idx_cells, 3], y[self.idx_cells, 4]
        Z = rZ / r
        C = rC / r
        Q = np.zeros(self.geometry.n)
        L = self.physics.get_normalized_progress_variable(Z, C)
        e_chem0 = r * self.physics.lookup_direct("E_CHEM", Z, Q, L)

        # 1st stage of RK2
        omegaC = self.injector.get_chemical_sources(
            self.t, y[self.idx_cells], self.physics, gamma_star[self.idx_cells]
        )
        y1 = y.copy()
        y1[self.idx_cells, 4] += dt * omegaC
        C1 = y1[self.idx_cells, 4] / r
        L1 = self.physics.get_normalized_progress_variable(Z, C1)
        e_chem1 = r * self.physics.lookup_direct("E_CHEM", Z, Q, L1)
        y1[self.idx_cells, 1] += e_chem0 - e_chem1

        # 2nd stage of RK2
        omegaC1 = self.injector.get_chemical_sources(
            self.t + dt, y1[self.idx_cells], self.physics, gamma_star[self.idx_cells]
        )
        y[self.idx_cells, 4] = 0.5 * (
            y[self.idx_cells, 4] + y1[self.idx_cells, 4] + dt * omegaC1
        )
        C = y[self.idx_cells, 4] / r
        L = self.physics.get_normalized_progress_variable(Z, C)
        e_chem2 = r * self.physics.lookup_direct("E_CHEM", Z, Q, L)
        y[self.idx_cells, 1] += e_chem0 - e_chem2

        # update properties
        self.state = self.physics.conservative_to_primitive(y, gamma_star)
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
        indices = [
            k + self.n_ghost_layers
            for k in range(self.n)
            if self.in_reacting_region(self.geometry.x[k], self.t)
        ]
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
            Y = np.clip(Y, 0.0, 1.0)
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
        the shock tube. The client must supply the functions dlnA_dt and dlnA_dx
        to the Combustor object.
        """
        y = self.physics.primitive_to_conservative(self.state)

        dydt = self.area_change.source(
            self.t,
            y[self.idx_cells],
            self.physics,
            self.state.gamma[self.idx_cells],
            dt,
        )

        # Update
        y[self.idx_cells] += dt * dydt
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

        dydt = self.boundary_layer.source(
            self.t, y[self.idx_cells], self.physics, self.state.gamma[self.idx_cells]
        )

        # Update
        y[self.idx_cells] += dydt * dt
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
        dydt = self.source_terms(
            self.t, y[self.idx_cells], self.state.gamma, self.geometry.x
        )
        y1 = y[self.idx_cells] + dt * dydt
        # state1 = self.physics.conservative_to_primitive(y1, self.state.gamma)

        # 2nd stage of RK2
        dydt = self.source_terms(self.t + dt, y1, self.state.gamma, self.geometry.x)
        y[self.idx_cells] = 0.5 * (y[self.idx_cells] + y1 + dt * dydt)
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
        if not isinstance(self.physics, FPVTable):
            msg = "JIC injector model requires FPVTable physics."
            raise Exception(msg)

        # initialize
        y = self.physics.primitive_to_conservative(self.state)
        self.injector.update_fluid_tip_positions(
            dt, self.t, self.state.velocity[self.idx_cells]
        )

        # 1st stage of RK2
        dydt = self.injector.source(
            self.t, y[self.idx_cells], self.physics, self.state.gamma[self.idx_cells]
        )
        y1 = y[self.idx_cells] + dt * dydt
        # state1 = self.physics.conservative_to_primitive(y1, self.state.gamma)

        # 2nd stage of RK2
        dydt = self.injector.source(
            self.t + dt, y1, self.physics, self.state.gamma[self.idx_cells]
        )
        y[self.idx_cells] = 0.5 * (y[self.idx_cells] + y1 + dt * dydt)

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
        for XTDiagram in self.xt_diagrams:
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
            if self.include_diffusion:
                self.advance_diffusion(dt)
            if self.area_change is not None:
                self.advance_quasi_1d(dt)
            if self.include_boundary_layer:
                self.advance_boundary_layer(dt)
            if self.source_terms is not None:
                self.advance_source_terms(dt)
            if self.injector is not None:
                self.advance_injector(dt)
            # perform other updates
            self.t += dt
            self.update_probes(iters)
            self.update_XT_diagrams(iters)
            iters += 1
            res_p = np.linalg.norm(self.state.pressure - p_old)
            if self.verbose and iters % self.output_every == 0:
                print(
                    f"Iteration: {iters}. Current time: {self.t}. Time step: {dt:e}. "
                    + f"Max T[K]: {self.physics.get_temperature(self.state).max()}. "
                    + f"Residual(p): {res_p}."
                )
            if (self.plot_state_interval > 0) and (
                iters % self.plot_state_interval == 0
            ):
                plot_state(
                    self,
                    f"figures/anim/test_{iters // self.plot_state_interval:05d}.png",
                )
