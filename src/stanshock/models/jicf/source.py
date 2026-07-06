from __future__ import annotations

import numpy as np
from scipy import interpolate

from stanshock.models.jicf.generate import JICModel
from stanshock.physics.flamelet import FPVTable
from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Unpack
from stanshock.system.base import PrecomputeSteps, RightHandSide
from stanshock.system.geometry import Box


class FuelInjector(RightHandSide):
    """Source terms for a fuel injector based on a JICF model.

    The injected mass flow rate is driven by a runtime *throttle schedule*
    (``t_inj`` -> ``mdot_inj``/``u_inj``/``E_inj``), which is decoupled from the
    throttle-agnostic :class:`JICModel` tables. The original fixed-schedule
    behaviour is recovered by passing that schedule here. Table lookups are
    performed on the momentum-flux ratio ``J``, computed from the injected jet
    state and the crossflow inflow so that they remain correct if the inflow is
    allowed to vary.
    """

    def __init__(
        self,
        jicf: JICModel,
        t_inj: Array,
        phi_inj: Array,
        mdot_inj: Array,
        u_inj: Array,
        E_inj: Array,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        """
        jicf: JICModel
            The underlying jet-in-crossflow model (tabulated over ``J``)
        t_inj: Array
            Time stamps of the throttle schedule
        phi_inj: Array
            Scheduled global equivalence ratio (used for reporting/plotting)
        mdot_inj: Array
            Scheduled total injected mass flow rate at each time stamp
        u_inj: Array
            Scheduled injected jet velocity at each time stamp
        E_inj: Array
            Scheduled injected total energy per unit mass at each time stamp
        geometry: Box
            The geometry object describing the mesh and cross-section
        physics: FPVTable
            The FPV table object, used for the chemical source terms
        """
        super().__init__(**precompute_steps)
        assert isinstance(self.geometry, Box)
        assert isinstance(self.physics, FPVTable)
        self.jicf = jicf

        # Runtime throttle schedule
        self.t_inj = np.asarray(t_inj, dtype=float)
        self.phi_inj = np.asarray(phi_inj, dtype=float)
        self.mdot_inj = np.asarray(mdot_inj, dtype=float)
        self.u_inj = np.asarray(u_inj, dtype=float)
        self.E_inj = np.asarray(E_inj, dtype=float)

        # Interpolants over the throttle schedule (also used for reporting/plotting)
        self.mdot_f_interp = interpolate.interp1d(
            self.t_inj, self.mdot_inj, bounds_error=False, fill_value=0.0
        )
        self.phi_f_interp = interpolate.interp1d(
            self.t_inj, self.phi_inj, bounds_error=False, fill_value=0.0
        )

        # Extract some information about the geometry
        self.xc = self.geometry.xc[self.idx_input]
        self.cell_volumes: Array | None = None
        if self.geometry.dlnA_dt is None:
            self.cell_volumes = self.geometry.volume()[:, None]

        # Position of the first injected fluid particle: [x, mdot, J]. Both the
        # mass flow rate (for reporting) and the momentum-flux ratio (the table
        # lookup axis) are convected with each parcel.
        self.fluid_tips = np.array(
            [[self.jicf.x_inj, self.mdot_inj[0], self._J(self.t_inj[0])]]
        )

    def _J(self, t: float) -> float:
        """Momentum-flux ratio ``J = rho_inj u_inj**2 / (rho u**2)`` at time ``t``.

        The crossflow inflow (``rho``, ``u``) is currently fixed on the model;
        when it becomes a live flow-field state this is where it would be read.
        """
        mdot = float(np.interp(t, self.t_inj, self.mdot_inj))
        u_inj = float(np.interp(t, self.t_inj, self.u_inj))
        if mdot <= 0.0 or u_inj <= 0.0:
            return 0.0
        rho_inj = mdot / (self.jicf.n_inj * u_inj * self.jicf.A_inj)
        return rho_inj * u_inj**2 / (self.jicf.rho * self.jicf.u**2)

    def update_fluid_tip_positions(self, dt, t, u):
        """
        This method updates the position of the fluid tips based on the velocity
        of the fluid.
        t: float
            The current time
        dt: float
            The time step
        x: float
            The current x-coordinate of the fluid tips
        u: float
            The current velocity of the fluid
        """
        # Update the fluid tip positions
        self.fluid_tips[:, 0] += dt * np.interp(self.fluid_tips[:, 0], self.xc, u)

        # Emit a new fluid tip, carrying its mass flow rate and momentum-flux ratio
        mdot = np.interp(t, self.t_inj, self.mdot_inj)
        next_tip = np.array([[self.jicf.x_inj, mdot, self._J(t)]])
        self.fluid_tips = np.concatenate([self.fluid_tips, next_tip], axis=0)

        # Drop fluid tips that have passed the end of the domain
        self.fluid_tips = self.fluid_tips[self.fluid_tips[:, 0] < self.xc[-1]]

    def source(
        self,
        time: float,
        state_array_local: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        """Compute a fuel injector source term to target the desired mixture fraction profile."""
        _ = gamma_star, e0_star
        assert self.geometry is not None
        state_array_local = np.reshape(state_array_local, self.shape_input)
        rhs = np.zeros_like(state_array_local)

        mdot = np.interp(time, self.t_inj, self.mdot_inj)
        u_inj = np.interp(time, self.t_inj, self.u_inj)
        E_inj = np.interp(time, self.t_inj, self.E_inj)
        L_src = 3e-2
        xf = self.geometry.xf
        idx = (
            np.where(
                np.logical_and(xf >= self.jicf.x_inj, xf < self.jicf.x_inj + L_src)
            )[0]
            + self.geometry.n_ghost_layers
        )
        dx = self.geometry.dx
        if isinstance(dx, np.ndarray):
            dx = dx[idx]
            mdot *= dx / np.sum(dx)
        else:
            mdot *= 1.0 / len(idx)

        # Compute the source term
        rhs[idx, 0] = mdot * u_inj * np.cos(self.jicf.theta_inj)  # momentum
        rhs[idx, 1] = mdot * E_inj  # total energy
        rhs[idx, 2] = mdot  # density
        rhs[idx, 3] = mdot  # mixture fraction
        rhs[idx, 4] = 0.0  # progress variable

        # Divide by the cell volumes
        if self.cell_volumes is None:
            assert self.geometry is not None
            vol = self.geometry.volume(time)[:, None]
            return np.ravel(rhs / vol)

        return np.ravel(rhs / self.cell_volumes)


class JICFChemistrySource(RightHandSide):
    def __init__(
        self, injector: FuelInjector, **precompute_steps: Unpack[PrecomputeSteps]
    ) -> None:
        super().__init__(**precompute_steps)
        self.injector = injector

        # Reactions only directly affect progress variable
        self.idx_source = np.array([4])
        self.shape_output = (self.shape_output[0], 1)

    def source_implementation(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Compute the chemical source terms [1/s] using the FPV table."""
        _ = time, state_array_local, face_states, avg_face_states, face_gradients
        assert isinstance(self.physics, FPVTable)
        assert state is not None
        assert state.density is not None
        assert state.mixture_fraction is not None
        assert state.normalized_progress_variable is not None
        factor = self.physics.get_source_progress_variable_compressibility_factor(state)

        # Get the mixture fraction variance profile. The variance table is
        # indexed by momentum-flux ratio J (column 2 of the fluid tips), which
        # is convected with each injected parcel.
        J_inj = np.interp(
            self.injector.xc,
            np.flip(self.injector.fluid_tips, axis=0)[:, 0],
            np.flip(self.injector.fluid_tips, axis=0)[:, 2],
        )
        Zvar = self.injector.jicf.Z_var_profile_interp((J_inj, self.injector.xc))

        # The progress-variable source term is a standalone FPVgen-format table
        # keyed on (mixture-fraction mean, log10 variance, normalized progress
        # variable). Clamp log10(variance) to the tabulated range to avoid
        # extrapolating the variance axis.
        src_table = self.injector.jicf.src_table
        log_Zvar = np.clip(np.log10(Zvar), src_table.Q.min(), src_table.Q.max())

        return (
            factor
            * state.density
            * src_table.lookup_direct(
                "SRC_PROG",
                state.mixture_fraction,
                log_Zvar,
                state.normalized_progress_variable,
            )
        )
