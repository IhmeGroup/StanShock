from __future__ import annotations

import numpy as np

from stanshock.models.area_change import AreaChange
from stanshock.models.boundary_layer import BoundaryLayer, SkinFriction
from stanshock.models.jicf import JICModel
from stanshock.numerics.boundary_conditions import (
    BCInput,
    BoundaryConditions,
    set_boundary_conditions,
)
from stanshock.numerics.face_average import SimpleAverage
from stanshock.numerics.face_extrapolation import (
    FaceExtrapolator,
    FifthOrderWeno,
    FirstOrder,
)
from stanshock.numerics.gradient import CentralDifference
from stanshock.numerics.inviscid_flux import (
    InviscidFlux,
    RiemannSolver,
    hllc_flux_vectorized,
)
from stanshock.numerics.time_integration import (
    SSPRK3,
    FastSlowIntegrator,
    ForwardEuler,
    HeunsMethod,
    LieSplitting,
    ScipyIVP,
    StrangSplitting,
    TimeIntegrator,
)
from stanshock.numerics.viscous_flux import ViscousFlux
from stanshock.physics.chemistry_source import ChemistrySource, ConstantVolumeChemistry
from stanshock.physics.fluid_base import FluidPhysics
from stanshock.processing.initialize import Initialization, InitializeRestart
from stanshock.processing.plot import XTDiagram, plot_state
from stanshock.processing.probe import Probe
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide
from stanshock.system.geometry import Geometry


class Combustor:
    """
    This is a class defined to encapsulate the data and methods used for the
    1D gasdynamics solver.
    """

    def __init__(
        self,
        physics: FluidPhysics,  # Model handling all fluid property evaluations
        geometry: Geometry,
        boundary_conditions: BoundaryConditions | BCInput,
        initialization: Initialization,  # initialization options
        cfl: float = 1.0,  # stability condition
        t: float = 0.0,  # time
        verbose: bool = True,  # console output switch
        output_every: int = 1,  # number of iterations of simulation advancement between logging updates
        include_boundary_layer: bool = False,  # flag to include boundary layer terms
        wall_temperature: float | None = None,  # wall temperature (needed for BL)
        skin_friction_coefficient: SkinFriction | None = None,  # skin friction functor
        source_terms: RightHandSide
        | list[RightHandSide]
        | None = None,  # Catch-all source term(s)
        injector: JICModel | None = None,  # injector model
        flux_function: RiemannSolver = hllc_flux_vectorized,
        inviscid_face_extrapolator: type[FaceExtrapolator] = FifthOrderWeno,
        viscous_face_extrapolator: type[FaceExtrapolator] = FirstOrder,
        probes: list[Probe] | None = None,  # list of probe objects
        xt_diagrams: list[XTDiagram] | None = None,  # list of XT diagram objects
        optimization_iteration: int = 0,  # counter to keep track of optimization
        reacting: bool = False,  # flag to solver about whether to solve source terms
        include_diffusion: bool = False,  # exclude diffusion
        thickening: None = None,  # thickening function
        plot_state_interval: int = -1,  # plot the state every n iterations
        iteration: int = 0,  # Iteration to start from
        n_restart_interval: int = -1,  # If >0, saves the fluid state to a file every n_restart_interval iterations
    ) -> None:
        """
        initialization of the object with default values. The keyword arguments
        allow the user to initialize the state
        """
        # initialize the class
        self.cfl = cfl
        self.t = t
        self.verbose = verbose
        self.output_every = output_every
        self.injector = injector
        self.optimization_iteration = optimization_iteration
        self.physics: FluidPhysics = physics
        self.initialization: Initialization = initialization
        self.include_diffusion = include_diffusion
        self.thickening = thickening
        self.plot_state_interval = plot_state_interval
        self.iteration = iteration
        self.n_restart_interval = n_restart_interval

        # Initialize values which are passed in as None
        self.probes: list[Probe] = [] if probes is None else probes
        self.xt_diagrams: list[XTDiagram] = [] if xt_diagrams is None else xt_diagrams

        # Determine the number of ghost layers required by the spatial scheme
        n_ghost_layers: int = inviscid_face_extrapolator.minimum_ghost_layers
        if self.include_diffusion:
            n_ghost_layers = max(
                n_ghost_layers, viscous_face_extrapolator.minimum_ghost_layers
            )

        # Initialize the geometry of the domain
        self.geometry = geometry
        self.geometry.setup_ghost_layers(n_ghost_layers=n_ghost_layers)

        # Set up boundary conditions
        self.boundary_conditions = set_boundary_conditions(
            boundary_conditions, self.geometry.n_ghost_layers
        )

        # Initialize the key physics
        self.inviscid_flux = InviscidFlux(
            face_extrapolator=inviscid_face_extrapolator(
                n_scalars_rho_sum=self.physics.n_scalars_rho_sum,
                n_ghost_layers=self.geometry.n_ghost_layers,
            ),
            boundary_conditions=self.boundary_conditions,
            riemann_solver=flux_function,
            geometry=self.geometry,
            physics=self.physics,
        )

        # Set up time integrators
        integrators: list[TimeIntegrator] = []
        advection = SSPRK3(self.inviscid_flux)
        chemistry: TimeIntegrator
        if reacting and self.physics.is_flamelet:
            integrators += [
                HeunsMethod(
                    ChemistrySource(geometry=self.geometry, physics=self.physics)
                ),
                advection,
            ]
        elif reacting and self.physics.gas.n_reactions > 0:
            chemistry = ScipyIVP(
                ConstantVolumeChemistry(geometry=self.geometry, physics=self.physics)
            )
            integrators += [StrangSplitting((chemistry, advection))]
        else:
            integrators += [advection]

        if self.include_diffusion:
            self.viscous_flux = ViscousFlux(
                geometry=self.geometry,
                physics=self.physics,
                boundary_conditions=self.boundary_conditions,
                face_extrapolator=viscous_face_extrapolator(
                    n_scalars_rho_sum=self.physics.n_scalars_rho_sum,
                    n_ghost_layers=self.geometry.n_ghost_layers,
                ),
                face_average=SimpleAverage(),
                gradient=CentralDifference(n_ghost_layers=self.geometry.n_ghost_layers),
            )
            integrators += [HeunsMethod(self.viscous_flux)]

        # Add area-change related source terms
        if self.geometry.dlnA_dt is not None or self.geometry.dlnA_dx is not None:
            self.area_change = AreaChange(geometry=self.geometry, physics=self.physics)
            integrators += [FastSlowIntegrator(self.area_change)]

        if include_boundary_layer:
            # Initialize the boundary layer source terms
            self.boundary_layer = BoundaryLayer(
                wall_temperature=wall_temperature,
                skin_friction_coefficient=skin_friction_coefficient,
                geometry=self.geometry,
                physics=self.physics,
            )
            integrators += [ForwardEuler(self.boundary_layer)]

        if source_terms is not None:
            if isinstance(source_terms, list):
                integrators += [HeunsMethod(rhs) for rhs in source_terms]
            else:
                integrators += [HeunsMethod(source_terms)]

        if self.injector is not None:
            if not self.physics.is_flamelet:
                msg = "JIC injector model requires FPVTable physics."
                raise Exception(msg)
            integrators += [HeunsMethod(self.injector)]

        # Apply Lie splitting approach
        self.time_integrator = LieSplitting(tuple(integrators), update_double_flux=True)

        self.F = np.ones(self.geometry.n_cells)  # thickening

    def get_wave_speed(self) -> Array:
        """
        This method determines the absolute maximum of the wave speed
            outputs:
                speed of acoustic wave
        """
        assert self.state.velocity is not None
        return abs(self.state.velocity) + self.physics.get_sound_speed(self.state)

    def get_time_step(self) -> float:
        """
        This method determines the maximal timestep in accord with the CFL
        condition
            outputs:
                timestep
        """
        local_timescale = self.geometry.dx / self.get_wave_speed()
        if self.include_diffusion:
            assert self.state.density is not None
            mu = self.physics.get_mu(self.state)
            nu = mu / self.state.density
            alpha = self.physics.get_thermal_diffusivity(self.state) * self.F
            diff: Array = (
                np.max(self.physics.get_mass_diffusivity(self.state), axis=1) * self.F
            )
            viscous_timescale: Array = (
                0.5
                * self.geometry.dx**2.0
                / np.maximum(4.0 / 3.0 * nu, np.maximum(alpha, diff))
            )
            local_timescale = np.minimum(local_timescale, viscous_timescale)
        return self.cfl * np.min(local_timescale)

    def update_probes(self, iters: int) -> None:
        """
        This method updates all the probes to the current value
        """

        # update probes
        for probe in self.probes:
            if iters % (probe.skipSteps + 1) == 0:
                probe.update(self)

    def update_XT_diagrams(self, iters: int) -> None:
        """
        This method updates all the XT Diagrams to the current value.
        """
        # update diagrams
        for diagram in self.xt_diagrams:
            if iters % (diagram.skipSteps + 1) == 0:
                diagram.update(self)

    def advance_simulation(self, tFinal: float, res_p_target: float = -1.0) -> None:
        """
        This method advances the simulation until the prescribed time, tFinal
            inputs
                    tFinal=final time
        """
        iters = self.iteration

        # Initialize the fluid state
        if not hasattr(self, "state"):
            self.state = self.initialization()

            if isinstance(self.initialization, InitializeRestart):
                iters = int(self.initialization.groupname)
            elif self.n_restart_interval > 0:
                # Save the initial conditions to a file
                self.state.save(groupname=str(iters))

            # Update plots at initial condition
            self.update_probes(iters)
            self.update_XT_diagrams(iters)

        res_p = np.inf
        gamma_star: Array | None
        e0_star: Array | None
        gamma_star, e0_star = self.physics.get_double_flux_variables(self.state)
        state_array = self.physics.primitive_to_conservative(self.state)
        self.shape_full: tuple[int, ...] = state_array.shape
        state_array = np.ravel(state_array)
        p_new = self.physics.get_pressure(self.state)
        while self.t < tFinal and res_p > res_p_target:
            dt = min(tFinal - self.t, self.get_time_step())
            p_old = p_new + 0.0

            # Update the system state
            self.t, state_array, gamma_star, e0_star = self.time_integrator.advance(
                dt=dt,
                time=self.t,
                state_array=state_array,
                gamma_star=gamma_star,
                e0_star=e0_star,
            )
            self.state = self.physics.conservative_to_primitive(
                np.reshape(state_array, self.shape_full), gamma_star, e0_star
            )
            p_new = self.physics.get_pressure(self.state)
            self.state.gamma = self.physics.get_gamma(self.state)

            # perform other updates
            self.update_probes(iters)
            self.update_XT_diagrams(iters)
            iters += 1
            res_p = float(np.linalg.norm(p_new - p_old))
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

            # Periodically save the fluid state
            if (self.n_restart_interval > 0) and (iters % self.n_restart_interval == 0):
                self.state.save(groupname=str(iters))

        # Save the final state
        self.iteration = iters
        if self.n_restart_interval > 0:
            self.state.save(groupname="stop")
