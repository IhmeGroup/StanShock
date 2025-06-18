from __future__ import annotations

from typing import Unpack

import numpy as np
from scipy import integrate

from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Index
from stanshock.system.base import FastSlowSource, PrecomputeSteps


class AreaChange(FastSlowSource):
    PRECOMPUTE_STEPS = ("geometry", "physics")

    def __init__(self, **precompute_steps: Unpack[PrecomputeSteps]) -> None:
        super().__init__(**precompute_steps)
        # self.integrator = integrate.ode(f=self.source_fast).set_integrator(name="lsoda")
        self.integrator = integrate.ode(
            f=self.source_fast, jac=self.source_fast_jacobian_banded
        ).set_integrator("lsoda", lband=0, uband=0)

        # For geometries with no area change, replace the source term with a no-op
        self.no_area_change: bool = (
            self.geometry.dlnA_dt is None and self.geometry.dlnA_dx is None
        )

    def update_indices(self, time: float) -> None:
        """Divide domain between explicit and implicit source terms."""
        idx_explicit: Index = self.geometry.idx_cells
        idx_implicit: Index = np.array([], dtype=np.int64)

        if self.geometry.dlnA_dt is not None:
            x = self.geometry.xc[self.idx_domain]
            dlnA_dt: Array | float = self.geometry.dlnA_dt(time, x)
            assert isinstance(dlnA_dt, np.ndarray)
            idx_implicit = np.where(dlnA_dt != 0.0)[0]
            idx_explicit = np.where(dlnA_dt == 0.0)[0]

        self.idx_implicit = idx_implicit
        self.idx_explicit = idx_explicit

    def precompute(
        self,
        time: float,
        state_array: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> tuple[
        Array,
        FluidState | None,
        FluidState | None,
        FluidState | None,
        FluidState | None,
    ]:
        state_array, state, _, _, _ = super().precompute(
            time, state_array, gamma_star, e0_star
        )
        assert state is not None
        assert state.density is not None
        assert state.pressure is not None

        idx = self.idx_domain
        state_array = state_array[idx]

        state0_compact = np.zeros((state_array.shape[0], 4))
        state0_compact[:, 0] = state.density[idx]
        state0_compact[:, 1] = state_array[:, 0]
        state0_compact[:, 2] = state_array[:, 1]
        state0_compact[:, 3] = state.pressure[idx]

        return state0_compact, state, None, None, None

    def source(
        self,
        time: float,
        state_array: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        rhs_full = np.zeros_like(state_array)
        if self.no_area_change:
            return rhs_full

        state_array, state, _, _, _ = self.precompute(
            time, state_array, gamma_star, e0_star
        )

        return self.source_implementation(time, state_array, state, None, None, None)

        # # Integrate fast terms implicitly
        # if idx_implicit.size != 0:
        #     # Initialize
        #     y0: Array = state0_compact[idx_implicit, :].copy()
        #     args: tuple[Array, Array] = (
        #         xc[idx_implicit],
        #         gamma_star[idx][idx_implicit],
        #     )
        #     self.integrator.set_initial_value(y=y0, t=time)
        #     self.integrator.set_f_params(args)
        #     self.integrator.set_jac_params(args)

        #     # Solve
        #     self.integrator.integrate(t=time + dt)

        #     # Store RHS source term
        #     rhs_compact: Array = (self.integrator.y - y0) / dt
        #     rhs[idx_implicit, 0:2] += rhs_compact[:, 1:]  # ru and re_t
        #     rhs[idx_implicit, 2:] += (
        #         rhs_compact[:, 0:1] * Y0[idx_implicit, :]
        #     )  # rY sources

        # # Add slow source terms
        # rhs_compact = self.source_slow(
        #     time=time, state0_compact=state0_compact, state=state, idx=idx_explicit
        # )
        # rhs[idx_explicit, 0:2] += rhs_compact[idx_explicit, 1:3]  # ru and re_t
        # rhs[idx_explicit, 2:] += (
        #     rhs_compact[idx_explicit, 0:1] * Y0[idx_explicit, :]
        # )  # rY sources

        # return rhs

    def postcompute(self, state_array: Array, state: FluidState) -> Array:
        """Undo any transforms to the state array during precompute steps."""
        state_array_full = np.zeros(self.shape)

        assert state.composition is not None
        Y = state.composition
        state_array_full[:, 0:2] = state_array[:, 1:3]  # ru and re_t
        state_array_full[:, 2:] = state_array[:, 0:1] * Y

        return np.ravel(state_array_full)

    def source_slow(
        self,
        time: float,
        state_array: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Area change contributions to RHS."""
        assert state_array is not None
        _ = face_states, avg_face_states, face_gradients
        idx = self.idx_explicit
        state_array = state_array[idx, :]
        rhs_compact: Array = np.zeros_like(state_array)

        if self.geometry.dlnA_dx is not None:
            assert state is not None
            assert state.pressure is not None
            assert state.velocity is not None

            x: Array = self.geometry.xc[idx]
            dlnA_dx: Array | float = self.geometry.dlnA_dx(time, x)
            rhs_compact[:, 0] -= state_array[:, 1] * dlnA_dx
            rhs_compact[:, 1] -= (
                state_array[:, 1] ** 2.0 / state_array[:, 0]
            ) * dlnA_dx
            rhs_compact[:, 2] -= (
                state.velocity[idx] * (state_array[:, 2] + state.pressure[idx])
            ) * dlnA_dx

        return rhs_compact

    def source_fast(
        self,
        time: float,
        state_array: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Fast source terms for quasi-1D geometry."""
        assert state_array is not None
        _ = face_states, avg_face_states, face_gradients
        x: Array = self.geometry.xc[self.idx_implicit]
        n: int = len(x)
        r: Array = state_array[0:n]
        ru: Array = state_array[n : 2 * n]
        rE: Array = state_array[2 * n : 3 * n]
        assert state is not None
        assert state.pressure is not None
        p: Array = state.pressure
        rhs: Array = np.zeros_like(state_array)

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
        gamma_star: Array = args[1]
        n: int = len(x)
        r: Array = y[0:n]
        ru: Array = y[n : 2 * n]
        # rE: Array = y[2 * n : 3 * n]
        # p = (gamma_star - 1) * (rE - r*e0_star - 0.5 * ru**2 / r)

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
        J_diag[2 * n : 3 * n] = -dlnAdt - (ru / r) * gamma_star * dlnAdx  # ∂R/∂E
        return J_diag.reshape(
            1, 3 * n
        )  # shape (1, 3n): banded with 0 lower and upper bandwidth
