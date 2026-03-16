from __future__ import annotations

from typing import Unpack

import numpy as np
from scipy import integrate

from stanshock.models.boundary_layer import BoundaryLayer
from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array
from stanshock.system.base import FastSlowMode, FastSlowSource, PrecomputeSteps


class Pseudoshock(FastSlowSource):
    """
    This function computes the pressure profile found in a pseudoshock, according to the analysis performed by Fievet et. al (2018).
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

        self.idx_locations = self.geometry.idx_cells
        self.idx_source_terms = np.s_[:]
        self.x = self.geometry.xc[self.idx_locations]

        self.wall_temperature = boundary_layer.wall_temperature
        self.skin_friction_coefficient = boundary_layer.skin_friction

        self.sf_array = []  # shock foot idx
        self.us = []  # shock speed
        self.t_ps = []
        self.L_scale = []
        self.sigma_ss = []

        if parameters is None:
            parameters = {}

        self.Ap = parameters.get("Ap", 1.64)
        self.Bp = parameters.get("Bp", 25.9)

        self.Cp = parameters.get("Cp", 2.5)
        self.Dp = parameters.get("Dp", 3.5)
        self.Ep = parameters.get("Ep", 0)

        self.k_ref = parameters.get("k_ref", 135.5)

        self.Beta_p = parameters.get("Beta_p", 1.1)
        self.M_ref = parameters.get("M_ref", 1.90)
        self.Alpha_p = parameters.get("Alpha_p", 1.064)
        self.sigma = parameters.get("sigma", 0.769521)

    def get_shock_idx(
        self,
        p: Array,
        u: Array,
        a: Array,
    ) -> None | int:
        p2p1a = np.maximum(p[1:], p[:-1]) / np.minimum(p[1:], p[:-1])
        dp_norm = np.sign(p[1:] - p[:-1])
        uL = u[:-1]
        uR = u[1:]
        aL = a[:-1]
        aR = a[1:]

        maskLP = (np.abs(p2p1a) >= 1.05) & (dp_norm == 1)
        l_entropy_condn = (uL - aL) > (uR - aR)  # gamma- char.
        l_states = maskLP & l_entropy_condn
        L = np.nonzero(l_states)[0]  # left-facing shock candidates
        if L.size == 0:
            return None
        s_idx_temp = L[0]
        return int(np.clip(s_idx_temp - 1, 0, len(dp_norm) - 1))

    def get_shock_properties(self, time, state: FluidState) -> None:
        assert self.geometry is not None
        assert self.physics is not None

        p = self.physics.get_pressure(state)
        u = self.physics.get_velocity(state)
        a = self.physics.get_sound_speed(state)

        shock_idx = self.get_shock_idx(p, u, a)

        if shock_idx is None:
            return None

        # Compute shock speed
        state = state[shock_idx]
        p1 = p[shock_idx]
        u1 = u[shock_idx]
        a1 = a[shock_idx]
        gamma1 = self.physics.get_gamma(state)

        # Compute length scale for pressure ratio
        x_shock = self.x[shock_idx]

        def Dh_func(x_shift) -> Array | float:
            return self.geometry.hydraulic_diameter(time, x_shift + x_shock)

        if not self.L_scale:
            self.L_scale = float(Dh_func(0))

        d_ind = np.rint(self.L_scale / (4 * self.geometry.dx)).astype(int)
        i0 = max(shock_idx - d_ind, 0)
        i1 = min(shock_idx + d_ind, len(p))
        p_wind = p[i0:i1]
        p_rat = np.max(p_wind / p1)

        # Compute shock speed
        us = u1 - a1 * np.sqrt(
            (p_rat * (gamma1 + 1) / (2 * gamma1)) + (gamma1 - 1) / (2 * gamma1)
        )
        M1 = u1 / a1
        M1_rel = M1 - (us / a1)
        if np.abs(us) >= 600 or M1_rel <= 1.3:
            return None

        # Compute skin friction
        T1 = self.physics.get_temperature(state)
        T_rat = T1 / self.wall_temperature
        mu1 = self.physics.get_mu(state)
        r1 = state.density

        Re0 = r1 * u1 * Dh_func(0) / mu1
        cf0 = float(self.skin_friction_coefficient(Re0, M1, T_rat))

        # Pre-compute Fievet model constants
        q1 = gamma1 * M1_rel**2 * p1 / 2
        kappa, k_c, cf_model = self.get_constants(M1_rel, cf0)

        # Prepare initial value and args for spatial integration
        y0 = [float(M1_rel**2), 1.000, float(p1)]

        args = (gamma1, q1, kappa, k_c, cf_model, Dh_func)

        # Update shock position, speed, and time trackers
        self.sf_array.append(shock_idx)
        self.us.append(us)
        self.t_ps.append(time)
        return args, y0, shock_idx, us

    def get_constants(
        self, M1: float, cf0: float, sigma: float | None = None
    ) -> tuple[float, float, float]:
        if sigma is None:
            sigma = self.sigma
        kappa = self.Bp * (1 - np.tanh(self.Cp * (M1 - self.M_ref)))
        norm_int = ((self.Ap + 1) ** (kappa + 1) - self.Ap ** (kappa + 1)) / (kappa + 1)
        k_c = (self.k_ref * cf0**self.Alpha_p * sigma**self.Beta_p) / norm_int
        cf_model = self.Ep + (self.Dp * cf0)
        return kappa, k_c, cf_model

    def M_Ar_Derivatives(self, y, x, gamma, q1, kappa, k_c, cf_model, Dh_func):
        Dh = Dh_func(x)
        M2, AcA, p = y
        q = gamma * M2 * p / 2
        dp_dx = (q / Dh) * k_c * (self.Ap + (q / q1)) ** kappa

        mult = -M2 * (1 + ((gamma - 1) / 2) * M2)
        term1M2 = (2 / (gamma * M2)) * (dp_dx / p)
        term2M2 = 4 * cf_model / Dh
        dM2_dx = mult * ((term1M2 + term2M2) / AcA)

        term1AcA = (1 - M2 * (1 - gamma * (1 - AcA))) / (gamma * M2 * AcA)
        term2AcA = ((1 + (gamma - 1) * M2) / (2 * AcA)) * (4 * cf_model / Dh)
        dAcA_dx = ((term1AcA) * (dp_dx / p) + term2AcA) * AcA

        return np.array([dM2_dx, dAcA_dx, dp_dx])

    def pseudoshock_solver(self, x: Array, args, y0):
        y_out = integrate.odeint(self.M_Ar_Derivatives, y0, x, args)
        M_out_temp = np.sqrt(y_out[:, 0])
        AcA_temp = y_out[:, 1]
        P_out_temp = y_out[:, 2]

        search_inds = np.where(np.gradient(AcA_temp) > 0)[0]
        if search_inds.size > 0:
            abs_diff = np.abs(AcA_temp[search_inds] - 1.0)
            end_ind = search_inds[np.argmin(abs_diff)] + 1
            M_out = M_out_temp[:end_ind]
            AcA = AcA_temp[:end_ind]
            P_out = P_out_temp[:end_ind]
        else:
            M_out, AcA, P_out = M_out_temp, AcA_temp, P_out_temp
        return M_out, AcA, P_out

    def update_indices(self, time: float, state: FluidState | None) -> None:
        """Scan domain for shocks."""
        assert state is not None
        shock_properties = self.get_shock_properties(time, state)
        if shock_properties is None:
            return

        args, y0, s_idx, us = shock_properties

        # Spatially integrate Fievet model
        M_out, AcA, P_out = self.pseudoshock_solver(self.x, args, y0)

        n_ps = len(P_out)
        if n_ps < 2:
            return

        r_idx = np.clip(s_idx + n_ps, 0, len(self.x) - 1)
        self.idx_output_explicit = np.arange(s_idx, r_idx)

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
        rhs: Array = np.zeros_like(state_array_local)
        idx = self.idx_output_explicit

        l_idx = r_idx - s_idx
        x_ps = self.x[idx]
        M_out = M_out[:l_idx]
        P_out = P_out[:l_idx]
        AcA = AcA[:l_idx]

        Dh = self.geometry.hydraulic_diameter(time, x_ps)
        gamma, q1, kappa, k_c, cf_model, Dh_func = args
        q = gamma * M_out**2 * P_out / 2
        K_PS = (k_c / Dh) * (self.Ap + (q / q1)) ** kappa

        P_error = (state.pressure[idx] - P_out) / P_out
        gain = np.clip(P_error, -5.0, 5.0)

        dp_dx_PS = q * K_PS
        rhs[idx, 0] += dp_dx_PS * gain

        sigma_ps = (state.pressure[r_idx]) / P_out[-1]
        self.sigma_ss.append(sigma_ps)
        return rhs
