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
        # self.integrator = integrate.ode(f=self.source_fast).set_integrator(name="lsoda")
        self.integrator = integrate.ode(
            f=self.source_fast, jac=self.source_fast_jacobian_banded
        ).set_integrator("lsoda", lband=0, uband=0)

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

        state = physics.conservative_to_primitive(state_array, gamma_star)
        state.gamma = gamma_star

        # Compute the density from the state array
        ru0 = state_array[:, 0]
        rY0 = state_array[:, 2:]
        r0 = rY0[..., : physics.n_scalars_rho_sum].sum(axis=-1)
        Y0 = rY0 / r0[..., None]

        rE0 = r0 * state.internal_energy + 0.5 * r0 * state.velocity**2

        state0_compact = np.zeros((state_array.shape[0], 3))
        state0_compact[:, 0] = r0
        state0_compact[:, 1] = ru0
        state0_compact[:, 2] = rE0
        state0_compact[:, 3] = state.pressure

        # Divide domain between explicit and implicit source terms
        idx_explicit: Index = np.arange(self.geometry.x.shape[0], dtype=np.int64)
        assert isinstance(idx_explicit, np.ndarray)
        idx_implicit: Index = np.array([], dtype=np.int64)
        assert isinstance(idx_implicit, np.ndarray)
        rhs: Array = np.zeros_like(
            state_array[self.idx_locations, self.idx_source_terms]
        )

        if self.geometry.dlnA_dt is not None:
            dlnA_dt: Array | float = self.geometry.dlnA_dt(time, self.geometry.x)
            assert isinstance(dlnA_dt, np.ndarray)
            idx_implicit = np.where(dlnA_dt != 0.0)[0]
            idx_explicit = np.where(dlnA_dt == 0.0)[0]

            # Integrate fast terms implicitly
            if idx_implicit.size != 0:
                # Initialize
                y0: Array = state0_compact[idx_implicit, :].copy()
                args: tuple[Array, Array] = (
                    self.geometry.x[idx_implicit],
                    gamma_star[idx_implicit],
                )
                self.integrator.set_initial_value(y=y0, t=time)
                self.integrator.set_f_params(args)
                self.integrator.set_jac_params(args)

                # Solve
                self.integrator.integrate(t=time + dt)

                # Store RHS source term
                rhs_compact: Array = (self.integrator.y - y0) / dt
                rhs[idx_implicit, 0:2] += rhs_compact[:, 1:]  # ru and re_t
                rhs[idx_implicit, 2:] += (
                    rhs_compact[:, 0:1] * Y0[idx_implicit, :]
                )  # rY sources

        # Add slow source terms
        state: FluidState = physics.conservative_to_primitive(
            state_array, gamma=gamma_star
        )
        rhs_compact = self.source_slow(
            time=time, state0_compact=state0_compact, state=state, idx=idx_explicit
        )
        rhs[idx_explicit, 0:2] += rhs_compact[idx_explicit, 1:]  # ru and re_t
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

    def source_fast(self, time: float, y: Array, args: tuple[Array, Array]) -> Array:
        """Fast source terms for quasi-1D geometry."""
        # Unpack the input and initialize
        x: Array = args[0]
        # gamma: Array = args[1]
        n: int = len(x)
        r: Array = y[0:n]
        ru: Array = y[n : 2 * n]
        rE: Array = y[2 * n : 3 * n]
        p: Array = y[3 * n : 4 * n]
        rhs: Array = np.zeros_like(y)

        # create quasi-1D right hand side
        if self.geometry.dlnA_dt is not None:
            dlnA_dt: Array | float = self.geometry.dlnA_dt(time, x)
            rhs[0:n] -= r * dlnA_dt
            rhs[n : 2 * n] -= ru * dlnA_dt
            rhs[2 * n : 3 * n] -= rE * dlnA_dt

        if self.geometry.dlnA_dx is not None:
            dlnA_dx: Array | float = self.geometry.dlnA_dx(time, x)
            rhs[0:n] -= ru * dlnA_dx
            rhs[n : 2 * n] -= (ru**2 / r) * dlnA_dx
            rhs[2 * n : 3 * n] -= (ru / r * (rE + p)) * dlnA_dx

        return rhs

    def source_fast_jacobian_banded(
        self, time: float, y: Array, args: tuple[Array, Array]
    ) -> Array:
        x: Array = args[0]
        gamma: Array = args[1]
        n: int = len(x)
        r: Array = y[0:n]
        ru: Array = y[n : 2 * n]
        # rE: Array = y[2 * n : 3 * n]
        # p = (gamma - 1) * (rE - 0.5 * ru**2 / r)

        dlnAdt: Array | float = (
            self.geometry.dlnA_dt(time, x) if self.geometry.dlnA_dt is not None else 0.0
        )
        dlnAdx: Array | float = (
            self.geometry.dlnA_dx(time, x) if self.geometry.dlnA_dx is not None else 0.0
        )

        J_diag: Array = np.zeros(3 * n)

        # Diagonal entries per variable
        J_diag[0:n] = -dlnAdt  # ∂R/∂ρ  # noqa: RUF003
        J_diag[n : 2 * n] = -dlnAdt  # ∂R/∂(ρu)  # noqa: RUF003
        J_diag[2 * n : 3 * n] = -dlnAdt - (ru / r) * gamma * dlnAdx  # ∂R/∂E
        return J_diag.reshape(
            1, 3 * n
        )  # shape (1, 3n): banded with 0 lower and upper bandwidth
