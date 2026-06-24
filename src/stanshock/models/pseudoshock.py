from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import integrate

from stanshock.models.boundary_layer import BoundaryLayer
from stanshock.models.wall_models import get_wall_state
from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Unpack
from stanshock.system.base import FastSlowMode, FastSlowSource, PrecomputeSteps

DhFunction = Callable[[Array | float], Array | float]


class Pseudoshock(FastSlowSource):
    """
    Compute the pressure profile found in a pseudoshock according to
    Fievet et al. (2018).
    """

    def __init__(
        self,
        boundary_layer: BoundaryLayer,
        parameters: dict[str, float] | None = None,
        mode: FastSlowMode = "slow",
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        super().__init__(mode, **precompute_steps)
        assert self.geometry is not None
        assert self.physics is not None

        self.boundary_layer = boundary_layer
        self.x = self.geometry.xc[self.geometry.idx_cells]
        self.all_idx = np.arange(len(self.x), dtype=np.int64)
        self.idx_output_explicit = np.array([], dtype=np.int64)
        self.idx_output_implicit = self.all_idx.copy()

        self.wall_temperature = boundary_layer.wall_temperature
        self.skin_friction_coefficient = boundary_layer.skin_friction

        self.sf_array: list[int] = []
        self.us: list[float] = []
        self.t_ps: list[float] = []
        self.L_scale: float = 0.0
        self.sigma_ss: list[float] = []
        self._cache_time: float | None = None
        self._last_logged_time: float | None = None
        self._reset_cached_solution()

        if parameters is None:
            parameters = {}

        self.Ap = parameters.get("Ap", 1.64)
        self.Bp = parameters.get("Bp", 25.9)
        self.Cp = parameters.get("Cp", 2.5)
        self.Dp = parameters.get("Dp", 3.5)
        self.Ep = parameters.get("Ep", 0.0)
        self.k_ref = parameters.get("k_ref", 135.5)
        self.Beta_p = parameters.get("Beta_p", 1.1)
        self.M_ref = parameters.get("M_ref", 1.90)
        self.Alpha_p = parameters.get("Alpha_p", 1.064)
        self.sigma = parameters.get("sigma", 0.769521)

    def _reset_cached_solution(self) -> None:
        self._ps_args: tuple[float, float, float, float, float, DhFunction] | None = (
            None
        )
        self._ps_r_idx: int | None = None
        self._ps_M_out = np.array([], dtype=np.float64)
        self._ps_P_out = np.array([], dtype=np.float64)

    def get_shock_idx(self, p: Array, u: Array, a: Array) -> None | int:
        p2p1a = np.maximum(p[1:], p[:-1]) / np.minimum(p[1:], p[:-1])
        dp_norm = np.sign(p[1:] - p[:-1])
        uL = u[:-1]
        uR = u[1:]
        aL = a[:-1]
        aR = a[1:]

        maskLP = (np.abs(p2p1a) >= 1.05) & (dp_norm == 1)
        l_entropy_condn = (uL - aL) > (uR - aR)
        l_states = maskLP & l_entropy_condn
        L = np.nonzero(l_states)[0]
        if L.size == 0:
            return None
        s_idx_temp = L[0]
        return int(np.clip(s_idx_temp - 1, 0, len(dp_norm) - 1))

    def get_shock_properties(
        self, time: float, state: FluidState
    ) -> (
        tuple[
            tuple[float, float, float, float, float, DhFunction],
            list[float],
            int,
            float,
        ]
        | None
    ):
        assert self.geometry is not None
        assert self.physics is not None

        p = self.physics.get_pressure(state)
        u = self.physics.get_velocity(state)
        a = self.physics.get_sound_speed(state)

        shock_idx = self.get_shock_idx(p, u, a)
        if shock_idx is None:
            return None

        p1 = float(p[shock_idx])
        u1 = float(u[shock_idx])
        a1 = float(a[shock_idx])
        gamma1 = float(self.physics.get_gamma(state)[shock_idx])
        self.x_shock = float(self.x[shock_idx])

        def Dh_func(x_shift: Array | float) -> Array | float:
            assert self.geometry is not None
            return self.geometry.hydraulic_diameter(time, x_shift + self.x_shock)

        if not self.L_scale:
            self.L_scale = float(Dh_func(0.0))

        d_ind = int(np.rint(self.L_scale / (4 * self.geometry.dx)))
        i0 = max(shock_idx - d_ind, 0)
        i1 = min(shock_idx + d_ind, len(p))
        p_wind = p[i0:i1]
        p_rat = float(np.max(p_wind / p1))

        us = u1 - a1 * np.sqrt(
            (p_rat * (gamma1 + 1.0) / (2.0 * gamma1)) + (gamma1 - 1.0) / (2.0 * gamma1)
        )
        M1 = u1 / a1
        M1_rel = M1 - (us / a1)
        if np.abs(us) >= 600.0 or M1_rel <= 1.3:
            return None

        state_point = state[shock_idx]
        x_shock = np.array([self.x_shock], dtype=np.float64)
        wall = get_wall_state(
            rho=state_point.density,
            U=state_point.velocity,
            mu=self.physics.get_mu(state_point),
            a=self.physics.get_sound_speed(state_point),
            cp=self.physics.get_cp(state_point),
            k=self.physics.get_thermal_conductivity(state_point),
            gamma=self.physics.get_gamma(state_point),
            Lc=self.geometry.characteristic_length(time, x_shock),
            T=self.physics.get_temperature(state_point),
            wall_temperature=self.wall_temperature,
        )
        cf0 = float(np.atleast_1d(self.skin_friction_coefficient(wall))[0])

        q1 = gamma1 * M1_rel**2 * p1 / 2.0
        kappa, k_c, cf_model = self.get_constants(M1_rel, cf0)
        y0 = [float(M1_rel**2), 1.0, p1]
        args = (gamma1, q1, kappa, k_c, cf_model, Dh_func)
        return args, y0, shock_idx, us

    def get_constants(
        self, M1: float, cf0: float, sigma: float | None = None
    ) -> tuple[float, float, float]:
        if sigma is None:
            sigma = self.sigma
        kappa = self.Bp * (1.0 - np.tanh(self.Cp * (M1 - self.M_ref)))
        norm_int = ((self.Ap + 1.0) ** (kappa + 1.0) - self.Ap ** (kappa + 1.0)) / (
            kappa + 1.0
        )
        k_c = (self.k_ref * cf0**self.Alpha_p * sigma**self.Beta_p) / norm_int
        cf_model = self.Ep + (self.Dp * cf0)
        return kappa, k_c, cf_model

    def M_Ar_Derivatives(
        self,
        y: Array,
        x: float,
        gamma: float,
        q1: float,
        kappa: float,
        k_c: float,
        cf_model: float,
        Dh_func: DhFunction,
    ) -> Array:
        Dh = Dh_func(x)
        M2, AcA, p = y
        q = gamma * M2 * p / 2.0
        dp_dx = (q / Dh) * k_c * (self.Ap + (q / q1)) ** kappa

        mult = -M2 * (1.0 + ((gamma - 1.0) / 2.0) * M2)
        term1M2 = (2.0 / (gamma * M2)) * (dp_dx / p)
        term2M2 = 4.0 * cf_model / Dh
        dM2_dx = mult * ((term1M2 + term2M2) / AcA)

        term1AcA = (1.0 - M2 * (1.0 - gamma * (1.0 - AcA))) / (gamma * M2 * AcA)
        term2AcA = ((1.0 + (gamma - 1.0) * M2) / (2.0 * AcA)) * (4.0 * cf_model / Dh)
        dAcA_dx = ((term1AcA) * (dp_dx / p) + term2AcA) * AcA

        return np.array([dM2_dx, dAcA_dx, dp_dx])

    def pseudoshock_solver(
        self,
        x: Array,
        args: tuple[float, float, float, float, float, DhFunction],
        y0: list[float],
    ) -> tuple[Array, Array, Array]:
        y_out = integrate.odeint(self.M_Ar_Derivatives, y0, x, args)
        M_out_temp = np.sqrt(y_out[:, 0])
        AcA_temp = y_out[:, 1]
        P_out_temp = y_out[:, 2]

        search_inds = np.where(np.gradient(AcA_temp) > 0)[0]
        if search_inds.size > 0:
            abs_diff = np.abs(AcA_temp[search_inds] - 1.0)
            end_ind = search_inds[np.argmin(abs_diff)] + 1
            M_out = M_out_temp[:end_ind]
            P_out = P_out_temp[:end_ind]
        else:
            M_out = M_out_temp
            P_out = P_out_temp
        return M_out, AcA_temp, P_out

    def update_indices(self, time: float, state: FluidState | None) -> None:
        assert state is not None
        self.idx_output_explicit = np.array([], dtype=np.int64)
        self.idx_output_implicit = self.all_idx.copy()
        self._reset_cached_solution()

        shock_properties = self.get_shock_properties(time, state)
        if shock_properties is None:
            return

        args, y0, s_idx, us = shock_properties
        M_out, _AcA, P_out = self.pseudoshock_solver(self.x, args, y0)
        n_ps = len(P_out)
        if n_ps < 2:
            return

        r_idx = int(np.clip(s_idx + n_ps, 0, len(self.x) - 1))
        idx_ps = np.arange(s_idx, r_idx, dtype=np.int64)
        if idx_ps.size == 0:
            return

        self.idx_output_explicit = idx_ps
        self.idx_output_implicit = np.concatenate(
            (self.all_idx[:s_idx], self.all_idx[r_idx:])
        )
        self._ps_args = args
        self._ps_r_idx = r_idx
        self._ps_M_out = M_out[: idx_ps.size]
        self._ps_P_out = P_out[: idx_ps.size]

        if self._last_logged_time != time:
            self.sf_array.append(s_idx)
            self.us.append(us)
            self.t_ps.append(time)
            self._last_logged_time = time

    def before_time_integration(
        self,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[Array, Array | None, Array | None]:
        if self.mode == "slow":
            self._cache_time = None
        return super().before_time_integration(time, state_array, gamma_star, e0_star)

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
        _ = time
        state_array_local = state_array_local.reshape(self.shape_input)

        if self.physics is not None:
            state = self.physics.conservative_to_primitive(
                state_array_local, gamma_star_local, e0_star_local
            )
            if gamma_star is not None:
                state.temperature = self.physics.get_temperature(state)
                state.internal_energy = None
                state_array_local = self.physics.primitive_to_conservative(state)
                gamma_star_local, e0_star_local = (
                    self.physics.get_double_flux_variables(state)
                )

                gamma_star[self.idx_input] = gamma_star_local
                assert e0_star is not None
                e0_star[self.idx_input] = e0_star_local

        state_array = np.reshape(state_array, self.shape_full)
        state_array[self.idx_input] = state_array_local

        return np.ravel(state_array), gamma_star, e0_star

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
        data = super().precompute_for_source(
            time, state_array_local, gamma_star, e0_star
        )
        state = data[1]
        if self._cache_time != time:
            assert state is not None
            self.update_indices(time, state)
            self._cache_time = time
        return data

    def add_source(self, y: Array, dy: Array) -> Array:
        state_array_local = np.reshape(y, self.shape_input)
        dydt = np.reshape(dy, self.shape_input)
        state_array_local += dydt
        return np.ravel(state_array_local)

    def source_slow(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        _ = face_states, avg_face_states, face_gradients
        assert self.geometry is not None
        assert state_array_local is not None
        assert state is not None
        rhs: Array = np.zeros_like(state_array_local)
        idx = self.idx_output_explicit
        if idx.size == 0 or self._ps_args is None or self._ps_r_idx is None:
            return rhs

        x_ps = self.x[idx]
        Dh = self.geometry.hydraulic_diameter(time, x_ps)
        gamma, q1, kappa, k_c, _cf_model, _Dh_func = self._ps_args
        q = gamma * self._ps_M_out**2 * self._ps_P_out / 2.0
        K_PS = (k_c / Dh) * (self.Ap + (q / q1)) ** kappa

        P_error = (state.pressure[idx] - self._ps_P_out) / self._ps_P_out
        gain = np.clip(P_error, -5.0, 5.0)
        dp_dx_PS = q * K_PS
        rhs[idx, 0] += dp_dx_PS * gain

        sigma_ps = float(state.pressure[self._ps_r_idx] / self._ps_P_out[-1])
        self.sigma_ss.append(sigma_ps)
        return rhs

    def source_fast(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        assert state_array_local is not None
        rhs: Array = np.zeros_like(state_array_local)
        idx = self.idx_output_implicit
        if idx.size == 0:
            return rhs

        rhs_bl = self.boundary_layer.source_implementation(
            time,
            state_array_local,
            state,
            face_states,
            avg_face_states,
            face_gradients,
        ).reshape(len(self.x), 2)
        rhs[idx, :2] = rhs_bl[idx]
        return rhs
