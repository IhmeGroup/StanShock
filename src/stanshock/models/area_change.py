from __future__ import annotations

import numpy as np
from scipy import integrate

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array, Index
from stanshock.system.base import RightHandSide
from stanshock.system.geometry import Geometry


class AreaChange(RightHandSide):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry: Geometry = geometry
        self.integrator = integrate.ode(f=self.source_fast).set_integrator(name="lsoda")

        # Define global indices
        self.idx_locations = np.s_[:]
        self.idx_source_terms = np.s_[:]

        # For geometries with no area change, replace the source term with a no-op
        self.no_area_change: bool = (
            self.geometry.dlnA_dt is None and self.geometry.dlnA_dx is None
        )

    def source(
        self,
        time: float,
        state_array: Array,
        physics: FluidPhysics,
        gamma_star: Array,
        dt: float,
    ) -> Array:
        if self.no_area_change:
            return np.zeros((1,))

        # Compute the density from the state array
        rY0 = state_array[:, 2:]
        r0 = rY0[..., : physics.n_scalars_rho_sum].sum(axis=-1)
        Y0 = rY0 / r0[..., None]

        state0_compact = np.zeros((state_array.shape[0], 3))
        state0_compact[:, 0] = r0
        state0_compact[:, 1:] = state_array[:, :2]  # ru and rE

        # Divide domain between explicit and implicit source terms
        idx_explicit: Index = np.arange(self.geometry.x.shape[0], dtype=np.int64)
        assert isinstance(idx_explicit, np.ndarray)
        idx_implicit: Index = np.array([], dtype=np.int64)
        assert isinstance(idx_implicit, np.ndarray)
        if self.geometry.dlnA_dt is not None:
            dlnA_dt: Array | float = self.geometry.dlnA_dt(time, self.geometry.x)
            assert isinstance(dlnA_dt, np.ndarray)
            idx_implicit = np.where(dlnA_dt != 0.0)[0]
            idx_explicit = np.where(dlnA_dt == 0.0)[0]

        # Integrate fast terms implicitly
        rhs: Array = np.zeros_like(
            state_array[self.idx_locations, self.idx_source_terms]
        )
        for i in idx_implicit:
            # Initialize
            y0: Array = state0_compact[i, :].copy()
            args: tuple[float, float] = self.geometry.x[i], gamma_star[i]
            self.integrator.set_initial_value(y=y0, t=time)
            self.integrator.set_f_params(args)

            # Solve
            self.integrator.integrate(t=time + dt)

            # Store RHS source term
            rhs_compact: Array = (self.integrator.y - state0_compact[i, :]) / dt
            rhs[i, 0:2] += rhs_compact[1:]  # ru and rE
            rhs[i, 2:] += rhs_compact[0] * Y0[i, :]  # rY sources

        # Add slow source terms
        state: FluidState = physics.conservative_to_primitive(
            state_array, gamma=gamma_star
        )
        rhs_compact = self.source_slow(
            time=time, state0_compact=state0_compact, state=state, idx=idx_explicit
        )
        rhs[idx_explicit, 0:2] += rhs_compact[idx_explicit, 1:]  # ru and rE
        rhs[idx_explicit, 2:] += (
            rhs_compact[idx_explicit, 0:1] * Y0[idx_explicit, :]
        )  # rY sources

        return rhs

    def source_slow(
        self, time: float, state0_compact: Array, state: FluidState, idx: Index
    ) -> Array:
        """Area change contributions to RHS."""
        rhs_compact: Array = np.zeros_like(state0_compact[idx])
        x: Array = self.geometry.x[idx]

        if self.geometry.dlnA_dt is not None:
            dlnA_dt: Array | float = self.geometry.dlnA_dt(time, x)
            rhs_compact -= state0_compact[idx, :] * dlnA_dt

        if self.geometry.dlnA_dx is not None:
            assert state.pressure is not None
            assert state.velocity is not None

            dlnA_dx: Array | float = self.geometry.dlnA_dx(time, x)
            rhs_compact[:, 0] -= state0_compact[idx, 1] * dlnA_dx
            rhs_compact[:, 1] -= (
                state0_compact[idx, 1] ** 2.0 / state0_compact[idx, 0]
            ) * dlnA_dx
            rhs_compact[:, 2] -= (
                state.velocity[idx] * (state0_compact[idx, 2] + state.pressure[idx])
            ) * dlnA_dx

        return rhs_compact

    def source_fast(self, time: float, y: Array, args: tuple[float, float]) -> Array:
        """Fast source terms for quasi-1D geometry."""
        # Unpack the input and initialize
        x0, gamma = args
        x: Array = np.array([x0])
        r, ru, rE = y
        p: float = (gamma - 1.0) * (rE - 0.5 * ru**2.0 / r)
        rhs_compact: Array = np.zeros(3)

        # create quasi-1D right hand side
        if self.geometry.dlnA_dt is not None:
            dlnA_dt: Array | float = self.geometry.dlnA_dt(time, x)
            rhs_compact[0] -= r * dlnA_dt
            rhs_compact[1] -= ru * dlnA_dt
            rhs_compact[2] -= rE * dlnA_dt

        if self.geometry.dlnA_dx is not None:
            dlnA_dx: Array | float = self.geometry.dlnA_dx(time, x)
            rhs_compact[0] -= ru * dlnA_dx
            rhs_compact[1] -= (ru**2.0 / r) * dlnA_dx
            rhs_compact[2] -= (ru / r * (rE + p)) * dlnA_dx

        return rhs_compact
