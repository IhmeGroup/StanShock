from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline

from stanshock.models.jicf.generate import JICModel
from stanshock.physics.flamelet import FPVTable
from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Unpack
from stanshock.system.base import PrecomputeSteps, RightHandSide
from stanshock.system.geometry import Box


class FuelInjector(RightHandSide):
    """Source terms for a fuel injector based on a JICF model."""

    def __init__(
        self, jicf: JICModel, **precompute_steps: Unpack[PrecomputeSteps]
    ) -> None:
        """
        jicf: JICModel
            The underlying jet-in-crossflow model
        geometry: Box
            The geometry object describing the mesh and cross-section
        physics: FPVTable
            The FPV table object, used for the chemical source terms
        """
        super().__init__(**precompute_steps)
        assert isinstance(self.geometry, Box)
        assert isinstance(self.physics, FPVTable)
        self.jicf = jicf

        # Extract some information about the geometry
        self.xc = self.geometry.xc[self.idx_input]
        self.cell_volumes: Array | None = None
        if self.geometry.dlnA_dt is None:
            self.cell_volumes = self.geometry.volume()[:, None]

        # Position of the first injected fluid particle
        self.fluid_tips = np.array([[self.jicf.x_inj, self.jicf.mdot_inj[0]]])

    def update_fluid_tip_positions(self, dt: float, t: float, u: Array) -> None:
        """
        This method updates the position of the fluid tips based on the velocity
        of the fluid.
        t: float
            The current time
        dt: float
            The time step
        u: Array
            The current velocity of the fluid
        """
        # Update the fluid tip positions
        self.fluid_tips[:, 0] += dt * make_interp_spline(self.xc, u, k=1)(
            self.fluid_tips[:, 0]
        )

        # Emit a new fluid tip
        mdot = self.jicf.mdot_f_interp(t)
        next_tip = np.array([[self.jicf.x_inj, mdot]])
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

        mdot: Array | float = self.jicf.mdot_f_interp(time)
        u_inj = np.interp(time, self.jicf.t_inj, self.jicf.u_inj)
        E_inj = np.interp(time, self.jicf.t_inj, self.jicf.E_inj)
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

        # Get the mixture fraction variance profile
        mdot_inj = make_interp_spline(
            np.flip(self.injector.fluid_tips, axis=0)[:, 0],
            np.flip(self.injector.fluid_tips, axis=0)[:, 1],
            k=1,
        )(self.injector.xc)
        Zvar = self.injector.jicf.Z_var_profile_interp((mdot_inj, self.injector.xc))
        Zvar = np.maximum(Zvar, 10 ** self.injector.jicf.logsigma2_vec.min())

        return (
            factor
            * state.density
            * self.injector.jicf.omega_C_int_interp(
                (
                    state.mixture_fraction,
                    state.normalized_progress_variable,
                    np.log10(Zvar),
                )
            )
        )
