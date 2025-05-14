from __future__ import annotations

import numpy as np

from stanshock.models.area_change import AreaChange
from stanshock.models.boundary_layer import BoundaryLayer
from stanshock.numerics.boundary_conditions import (
    BCInput,
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
from stanshock.numerics.time_integration import (
    SSPRK3,
    ForwardEuler,
    HeunsMethod,
    LieSplitting,
    ScipyIVP,
    StrangSplitting,
)
from stanshock.numerics.viscous_flux import ViscousFlux
from stanshock.physics.fluid_base import ChemistrySource, FluidPhysics
from stanshock.processing.initialize import (
    initialize_constant,
    initialize_diffuse_interface,
    initialize_isentropic,
    initialize_riemann_problem,
)
from stanshock.processing.plot import plot_state
from stanshock.system.backend import Array
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
        n_cells: int = 10,
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
        self.n_cells = n_cells  # grid size
        self.boundary_conditions: BoundaryConditions | BCInput = {
            "left": "outflow",
            "right": "outflow",
        }
        self.xf: Array = np.linspace(
            0.0, self.dx * self.n_cells, self.n_cells + 1, dtype=np.float64
        )
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
        self.inviscid_face_extrapolator: type[FaceExtrapolator] = FifthOrderWeno
        self.viscous_face_extrapolator: type[FaceExtrapolator] = FirstOrder
        self.initialization = None  # initialization options
        self.probes = []  # list of probe objects
        self.xt_diagrams = []  # list of XT diagram objects
        self.skin_friction_coefficient = None  # skin friction functor
        self.optimization_iteration = 0  # counter to keep track of optimization
        self.physics = physics  # Model handling all fluid property evaluations
        self.reacting = False  # flag to solver about whether to solve source terms
        self.in_reacting_region = lambda _t, x: np.ones_like(
            x, dtype=bool
        )  # the reacting region of the shock tube.
        self.include_diffusion = False  # exclude diffusion
        self.thickening = None  # thickening function
        self.plot_state_interval = -1  # plot the state every n iterations
        # overwrite the default data
        for key, item in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key] = item

        # Determine the number of ghost layers required by the spatial scheme
        n_ghost_layers: int = self.inviscid_face_extrapolator.minimum_ghost_layers
        if self.include_diffusion:
            n_ghost_layers = max(
                n_ghost_layers, self.viscous_face_extrapolator.minimum_ghost_layers
            )

        # Initialize the geometry of the domain
        if geometry is None:
            kwargs.pop("xf")
            self.geometry: Geometry = initialize_geometry(
                xf=self.xf, n_ghost_layers=n_ghost_layers, **kwargs
            )
        else:
            self.geometry = geometry
            self.geometry.setup_ghost_layers(n_ghost_layers=n_ghost_layers)

        # Add area-change related source terms
        if self.geometry.dlnA_dt is not None or self.geometry.dlnA_dx is not None:
            self.area_change = AreaChange(geometry=self.geometry)

        # set the number of scalars
        self.n_scalars = self.physics.n_scalars
        if not self.physics.is_flamelet and self.injector is not None:
            msg = "JIC injector model requires FPVTable physics."
            raise Exception(msg)

        # Set up boundary conditions
        self.boundary_conditions = set_boundary_conditions(
            self.boundary_conditions, self.geometry.n_ghost_layers
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
        elif self.initialization[0].lower() == "isentropic":
            self.state = initialize_isentropic(
                self.geometry, self.physics, *self.initialization[1:]
            )

        # Initialize the key physics
        self.inviscid_flux = InviscidFlux(
            face_extrapolator=self.inviscid_face_extrapolator(
                n_scalars_rho_sum=self.physics.n_scalars_rho_sum,
                n_ghost_layers=self.geometry.n_ghost_layers,
            ),
            boundary_conditions=self.boundary_conditions,
            riemann_solver=self.flux_function,
            geometry=self.geometry,
        )

        # Set up time integrators
        integrators = []
        advection = SSPRK3(self.inviscid_flux)
        if self.physics.is_flamelet:
            integrators += [
                HeunsMethod(ChemistrySource),
                advection,
            ]
        else:
            integrators += [
                StrangSplitting(
                    transport_operator=advection,
                    reaction_operator=ScipyIVP(ChemistrySource),
                )
            ]

        if self.include_diffusion:
            self.viscous_flux = ViscousFlux(
                boundary_conditions=self.boundary_conditions,
                face_extrapolator=self.viscous_face_extrapolator(
                    n_scalars_rho_sum=self.physics.n_scalars_rho_sum,
                    n_ghost_layers=self.geometry.n_ghost_layers,
                ),
                geometry=self.geometry,
                gradient=CentralDifference(n_ghost_layers=self.geometry.n_ghost_layers),
            )
            integrators += [HeunsMethod(self.viscous_flux)]

        if self.area_change is not None:
            integrators += [ForwardEuler(self.area_change)]

        if self.include_boundary_layer:
            # Initialize the boundary layer source terms
            self.boundary_layer = BoundaryLayer(
                geometry=self.geometry,
                wall_temperature=self.wall_temperature,
                skin_friction_coefficient=self.skin_friction_coefficient,
            )
            integrators += [ForwardEuler(self.boundary_layer)]

        if self.source_terms is not None:
            integrators += [HeunsMethod(self.source_terms)]

        if self.injector is not None:
            integrators += [HeunsMethod(self.injector)]

        # Apply Lie splitting approach
        self.time_integrator = LieSplitting(integrators, update_double_flux=True)

        self.F = np.ones(self.geometry.n_cells)  # thickening

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
        gamma_star, e0_star = self.physics.get_double_flux_variables(self.state)
        state_array = self.physics.primitive_to_conservative(self.state)
        while self.t < tFinal and res_p > res_p_target:
            p_old = self.state.pressure
            dt = min(tFinal - self.t, self.get_time_step())

            # Update the system state
            self.t, state_array = self.time_integrator.advance(
                dt=dt,
                time=self.t,
                state_array=state_array,
                physics=self.physics,
                gamma_star=gamma_star,
                e0_star=e0_star,
            )
            self.state = self.physics.conservative_to_primitive(
                state_array, gamma_star, e0_star
            )
            gamma_star, e0_star = self.physics.get_double_flux_variables(self.state)

            # perform other updates
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
