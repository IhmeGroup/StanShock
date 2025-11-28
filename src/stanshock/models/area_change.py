from __future__ import annotations

import numpy as np

from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Index, Unpack
from stanshock.system.base import FastSlowSource, PrecomputeSteps


class AreaChange(FastSlowSource):
    PRECOMPUTE_STEPS = ("geometry", "physics")

    def __init__(self, **precompute_steps: Unpack[PrecomputeSteps]) -> None:
        super().__init__(**precompute_steps)
        assert self.geometry is not None
        self.shape_update = (-1, 3)
        self.jac = self.source_fast_jacobian_banded
        self.x: Array = self.geometry.xc[self.idx_domain]

        # For geometries with no area change, replace the source term with a no-op
        self.no_area_change: bool = (
            self.geometry.dlnA_dt is None and self.geometry.dlnA_dx is None
        )

    def update_indices(self, time: float, state: FluidState | None) -> None:
        """Divide domain between explicit and implicit source terms."""
        assert self.geometry is not None
        _ = state
        idx_update_implicit: Index = np.array([], dtype=np.int64)
        idx_update_explicit: Index = np.s_[:]

        if self.geometry.dlnA_dt is not None:
            dlnA_dt: Array | float = self.geometry.dlnA_dt(time, self.x)
            assert isinstance(dlnA_dt, np.ndarray)
            idx_update_implicit = np.where(dlnA_dt != 0.0)[0]
            idx_update_explicit = np.where(dlnA_dt == 0.0)[0]

        self.idx_update_implicit = idx_update_implicit
        self.idx_update_explicit = idx_update_explicit

    def before_time_integration(
        self,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[Array, Array | None, Array | None]:
        """Replace species transport equations with mass continuity."""
        assert self.physics is not None
        state_array_local, gamma_star, e0_star = super().before_time_integration(
            time, state_array, gamma_star, e0_star
        )

        state_array_local = np.reshape(state_array_local, self.shape_domain)
        state_compact = state_array_local[:, :3].copy()
        n = self.physics.n_scalars_rho_sum
        if n > 1:
            state_compact[:, 2] = np.sum(state_array_local[:, 2 : 2 + n], axis=1)

        # Store the frozen composition
        self.composition_frozen = state_array_local[:, 2:] / state_compact[:, 2:3]

        return np.ravel(state_compact), gamma_star, e0_star

    def after_time_integration(
        self,
        time: float,
        state_array_local: Array,
        gamma_star_local: Array | None,
        e0_star_local: Array | None,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[Array, Array | None, Array | None]:
        """Apply density update to all scalars."""
        assert self.physics is not None
        state_array_local = np.reshape(state_array_local, shape=self.shape_update)
        state_array_local = np.pad(
            state_array_local, ((0, 0), (0, self.physics.n_scalars - 1)), mode="edge"
        )
        state_array_local[:, 2:] *= self.composition_frozen

        return super().after_time_integration(
            time=time,
            state_array_local=state_array_local,
            gamma_star_local=gamma_star_local,
            e0_star_local=e0_star_local,
            state_array=state_array,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

    def add_source(self, y: Array, dy: Array) -> Array:
        """Add (2D) source term to the (1D) state array."""
        dy = np.reshape(dy, self.shape_update)
        state_array_local = np.reshape(y, self.shape_update)
        state_array_local[self.idx_update, self.idx_source] += dy
        return np.ravel(state_array_local)

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
        assert self.physics is not None
        _ = time
        state_array_local = np.reshape(state_array_local, shape=self.shape_update)
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
        state.pressure = self.physics.get_pressure(state)
        state.temperature = self.physics.get_temperature(state)

        return np.ravel(state_array_local), state, None, None, None

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

    def source_full(
        self,
        time: float,
        state_array_local: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        """Transform the RHS into standard format."""
        assert self.physics is not None
        rhs = self.source(time, state_array_local, gamma_star, e0_star)
        rhs = np.reshape(rhs, self.shape_update)
        rhs = np.pad(rhs, ((0, 0), (0, self.physics.n_scalars - 1)), mode="edge")
        rhs[:, 2:] *= self.composition_frozen

        rhs_full = np.zeros(self.shape_domain)
        rhs_full[self.idx_update, :] = rhs

        return np.ravel(rhs_full)

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
        assert self.geometry is not None
        assert state_array_local is not None
        _ = face_states, avg_face_states, face_gradients
        state_array_local = np.reshape(state_array_local, shape=self.shape_update)
        rhs_compact: Array = np.zeros_like(state_array_local)

        if self.geometry.dlnA_dx is not None:
            assert state is not None
            assert state.velocity is not None
            assert state.pressure is not None
            ru: Array = state_array_local[:, 0]
            rE: Array = state_array_local[:, 1]
            u: Array = state.velocity
            p: Array = state.pressure

            x: Array = self.x[self.idx_update_explicit]
            dlnA_dx: Array | float = self.geometry.dlnA_dx(time, x)

            rhs_compact[:, 0] -= u * ru * dlnA_dx
            rhs_compact[:, 1] -= u * (rE + p) * dlnA_dx
            rhs_compact[:, 2] -= ru * dlnA_dx

        return np.ravel(rhs_compact)

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
        assert self.geometry is not None
        assert state_array_local is not None
        _ = face_states, avg_face_states, face_gradients
        x: Array = self.x[self.idx_update_implicit]
        state_array_local = np.reshape(state_array_local, shape=self.shape_update)
        rhs_compact: Array = np.zeros_like(state_array_local)

        # create quasi-1D right hand side
        if self.geometry.dlnA_dt is not None:
            dlnA_dt: Array | float = self.geometry.dlnA_dt(time, x)
            if isinstance(dlnA_dt, np.ndarray):
                dlnA_dt = dlnA_dt[:, None]
            rhs_compact -= state_array_local * dlnA_dt

        if self.geometry.dlnA_dx is not None:
            assert state is not None
            assert state.velocity is not None
            assert state.pressure is not None
            ru: Array = state_array_local[:, 0]
            rE: Array = state_array_local[:, 1]
            u: Array = state.velocity
            p: Array = state.pressure
            dlnA_dx: Array | float = self.geometry.dlnA_dx(time, x)

            rhs_compact[:, 0] -= u * ru * dlnA_dx
            rhs_compact[:, 1] -= u * (rE + p) * dlnA_dx
            rhs_compact[:, 2] -= ru * dlnA_dx

        return np.ravel(rhs_compact)

    def source_fast_jacobian_banded(
        self,
        time: float,
        state_array_local: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> Array:
        assert self.geometry is not None
        assert gamma_star is not None
        _ = e0_star
        x: Array = self.geometry.xc[self.idx_update_implicit]
        n: int = len(x)
        state_array_local = np.reshape(state_array_local, shape=self.shape_update)
        ru: Array = state_array_local[:, 0]
        r: Array = state_array_local[:, 2]
        # rE: Array = y[2 * n : 3 * n]
        # p = (gamma_star - 1) * (rE - r*e0_star - 0.5 * ru**2 / r)

        dlnAdt: Array | float = (
            self.geometry.dlnA_dt(time, x) if self.geometry.dlnA_dt is not None else 0.0
        )
        dlnAdx: Array | float = (
            self.geometry.dlnA_dx(time, x) if self.geometry.dlnA_dx is not None else 0.0
        )

        J_diag: Array = np.zeros((n, 3))

        # Diagonal entries per variable
        J_diag[:, 0] = -dlnAdt  # ∂R/∂(ρu)  # noqa: RUF003
        J_diag[:, 1] = -dlnAdt - (ru / r) * gamma_star * dlnAdx  # ∂R/∂E
        J_diag[:, 2] = -dlnAdt  # ∂R/∂ρ  # noqa: RUF003
        return J_diag.reshape(
            1, 3 * n
        )  # shape (1, 3n): banded with 0 lower and upper bandwidth
