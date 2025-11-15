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
            x = self.geometry.xc[self.idx_update]
            dlnA_dt: Array | float = self.geometry.dlnA_dt(time, x)
            assert isinstance(dlnA_dt, np.ndarray)
            idx_implicit = np.where(dlnA_dt != 0.0)[0]
            idx_explicit = np.where(dlnA_dt == 0.0)[0]

        self.idx_implicit = idx_implicit
        self.idx_explicit = idx_explicit

    def before_time_integration(
        self,
        time: float,
        state_array: Array,
        update_double_flux: bool = True,
    ) -> tuple[Array, Array | None, Array | None]:
        """Replace species transport equations with mass continuity."""
        state_array_local, gamma_star, e0_star = super().before_time_integration(
            time, state_array, update_double_flux
        )

        assert self.physics is not None
        state_compact = state_array_local[:, :3].copy()
        n = self.physics.n_scalars_rho_sum
        if n > 1:
            state_compact[:, 2] = np.sum(state_array_local[:, 2 : 2 + n], axis=1)

        # Store the frozen composition
        self.composition_frozen = state_array_local[:, 2:] / state_compact[:, 2:3]

        return state_compact, gamma_star, e0_star

    def after_time_integration(
        self, state_array: Array, state_array_local: Array
    ) -> Array:
        """Apply density update to all scalars."""
        assert self.physics is not None
        state_array_local = np.pad(
            state_array_local, (0, (0, self.physics.n_scalars - 1)), mode="edge"
        )
        state_array_local[:, 2:] *= self.composition_frozen
        state_array[self.idx_domain] = np.ravel(state_array_local)

        return state_array

    def precompute_for_source(
        self,
        time: float,
        state_array_local: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> tuple[
        Array,
        FluidState | None,
        FluidState | None,
        FluidState | None,
        FluidState | None,
    ]:
        _ = time
        ru = state_array_local[:, 0]
        re_t = state_array_local[:, 1]
        r = state_array_local[:, 2]

        u = ru / r
        e_int = (re_t / r) - 0.5 * u**2.0

        state = FluidState(
            shape=(state_array_local.shape[0],),
            density=r,
            velocity=u,
            internal_energy=e_int,
            composition=self.composition_frozen,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

        return state_array_local, state, None, None, None

    def source(
        self,
        time: float,
        state_array_local: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        if self.no_area_change:
            return np.zeros_like(state_array_local)
        return super().source(time, state_array_local, gamma_star, e0_star)

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

    def source_slow(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Area change contributions to RHS."""
        assert state_array_local is not None
        _ = face_states, avg_face_states, face_gradients
        idx = self.idx_explicit
        state_array_local = state_array_local[idx, :]
        rhs_compact: Array = np.zeros_like(state_array_local)

        if self.geometry.dlnA_dx is not None:
            assert state is not None
            assert state.pressure is not None
            assert state.velocity is not None

            x: Array = self.geometry.xc[idx]
            dlnA_dx: Array | float = self.geometry.dlnA_dx(time, x)
            rhs_compact[:, 0] -= state_array_local[:, 1] * dlnA_dx
            rhs_compact[:, 1] -= (
                state_array_local[:, 1] ** 2.0 / state_array_local[:, 0]
            ) * dlnA_dx
            rhs_compact[:, 2] -= (
                state.velocity[idx] * (state_array_local[:, 2] + state.pressure[idx])
            ) * dlnA_dx

        return rhs_compact

    def source_fast(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Fast source terms for quasi-1D geometry."""
        assert state_array_local is not None
        _ = face_states, avg_face_states, face_gradients
        x: Array = self.geometry.xc[self.idx_implicit]
        n: int = len(x)
        r: Array = state_array_local[0:n]
        ru: Array = state_array_local[n : 2 * n]
        rE: Array = state_array_local[2 * n : 3 * n]
        assert state is not None
        assert state.pressure is not None
        p: Array = state.pressure
        rhs: Array = np.zeros_like(state_array_local)

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
