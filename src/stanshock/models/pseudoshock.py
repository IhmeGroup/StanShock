from __future__ import annotations

import itertools
from collections.abc import Callable

from stanshock.models.boundary_layer import BoundaryLayer
from stanshock.models.wall_models import WallState
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array, Index, TypeAlias, Unpack, np
from stanshock.system.base import PrecomputeSteps, RightHandSide

_Accept: TypeAlias = tuple[float, FluidState, float, float, Array, FluidState, Array]
_Trial: TypeAlias = tuple[
    float, FluidState, float, float, Array, FluidState, Array, float
]


def rk2_step(
    rhs: Callable[[Array, float], Array], y: Array, x: float, dx: float
) -> Array:
    k1 = rhs(y, x)
    return y + dx * rhs(y + 0.5 * dx * k1, x + 0.5 * dx)


def interpolate_fluid_state(
    x: Array,
    state: FluidState,
    x_q: Array | float,
    physics: FluidPhysics,
) -> FluidState:
    x_q = np.atleast_1d(x_q)
    state_array = physics.primitive_to_conservative(state)
    state_array_q = np.column_stack(
        [np.interp(x_q, x, state_array[:, i]) for i in range(state_array.shape[1])]
    )
    return physics.conservative_to_primitive(state_array_q)


class Pseudoshock(RightHandSide):
    state_0: FluidState

    def __init__(
        self,
        boundary_layer: BoundaryLayer,
        parameters: dict[str, float] | None = None,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        super().__init__(**precompute_steps)
        geometry = self.geometry
        physics = self.physics
        assert geometry is not None
        assert physics is not None

        self.boundary_layer = boundary_layer

        self.x = geometry.xc[geometry.idx_cells]
        dx = geometry.dx
        if isinstance(dx, np.ndarray):
            self.dx_cells = dx[geometry.idx_cells]
        else:
            self.dx_cells = np.full(self.x.shape, dx, dtype=np.float64)

        self.all_idx: Index = np.arange(len(self.x), dtype=np.int64)

        self.Tw = boundary_layer.wall_temperature
        self.skin_friction_coefficient = boundary_layer.skin_friction
        self.heat_flux = boundary_layer.heat_flux

        self.x_sf: list[float] = []
        self.us: list[float] = []
        self.t_ps: list[float] = []
        self.sigma_ss: list[float] = []
        self.p2p1: list[float] = []
        self.L_ps: list[float] = []
        self.AcA_ps: Array = np.empty(0, dtype=np.float64)
        self.Dh_fxn: Callable[[Array | float], Array | float] | None = None
        self.A_fxn: Callable[[Array | float], Array | float] | None = None
        self.Pwet_fxn: Callable[[Array | float], Array | float] | None = None
        self.dlnA_dx_fxn: Callable[[Array | float], Array | float] | None = None
        self.A0: float | None = None
        self.cf0: float | None = None
        self._accepted_profile: tuple[Array, FluidState, Array] | None = None

        self.p_res: list[float] = []
        self.p_res_max = 0.05

        if parameters is None:
            parameters = {}

        self.alpha_p = parameters.get("alpha_p", 0.01)  # skin friction scaling
        self.beta_p = parameters.get("beta_p", 0.11)  # obliqueness, user-defined
        self.Gamma_p = parameters.get("gamma_p", 0.0575)  # Default
        self.Omega_p = parameters.get("omega_p", 0.75)  # Default

        self.S_p = parameters.get("S_p", parameters.get("S", 0.90))  # Symmetry
        self.sigma_p = parameters.get("sigma_p", parameters.get("sigma", 0.76))
        self.Lambda_p = self.beta_p * self.S_p * self.sigma_p
        self.p_res_max = parameters.get("p_res_max", self.p_res_max)

        self.n_shock_root_samples = int(parameters.get("n_shock_root_samples", 6))
        self.n_shock_root_iterations = int(parameters.get("n_shock_root_iterations", 4))
        self.shock_search_width_factor = parameters.get(
            "shock_search_width_factor", 2.0
        )
        self.shock_search_min_cells = int(parameters.get("shock_search_min_cells", 2))

    def get_planar_shock_foot(
        self, p: Array, u: Array, a: Array, max_half: int = 6, edge_tol: float = 1.005
    ) -> tuple[float, float, bool]:
        """Returns (s1_float, pi_local, armed) or (None, None, False).
        s1_float is a real-valued cell-centre coordinate (index units)."""

        pR_pL = p[1:] / p[:-1]  # >1 iff compressive
        compressive = pR_pL >= 1.03
        entropy = (u[:-1] - a[:-1]) > (u[1:] - a[1:])
        cand = np.nonzero(compressive & entropy)[0]
        if cand.size == 0:
            return -1e30, -1e30, False
        f = int(cand[0])  # most upstream candidate

        lo = f
        while lo > 0 and pR_pL[lo - 1] >= edge_tol and (f - lo) < max_half:
            lo -= 1
        hi = f + 1
        while hi < len(pR_pL) and pR_pL[hi] >= edge_tol and (hi - f) < max_half:
            hi += 1
        p_lo, p_hi = p[lo], p[hi]
        pi_local = p_hi / p_lo

        pi_crit = 1.50
        if pi_local < pi_crit:
            return -1e30, pi_local, False
        p_t = p_lo * pi_crit
        seg = p[lo : hi + 1]
        k = int(np.searchsorted(seg, p_t))
        if k == 0:
            return float(lo), pi_local, True
        k = min(k, len(seg) - 1)

        lp0, lp1 = np.log(seg[k - 1]), np.log(seg[k])
        frac = 0.0 if lp1 == lp0 else (np.log(p_t) - lp0) / (lp1 - lp0)
        s1 = (lo + k - 1) + np.clip(frac, 0.0, 1.0)
        return float(s1), pi_local, True

    def clear_pseudoshock_setup(self) -> None:
        self.cf0 = None
        self.Dh_fxn = None
        self.A_fxn = None
        self.Pwet_fxn = None
        self.dlnA_dx_fxn = None
        self.A0 = None

    def set_pseudoshock_start(
        self,
        time: float,
        x_sf: float,
        state_0: FluidState,
        us: float,
    ) -> None:
        assert self.geometry is not None
        assert self.physics is not None
        self.Dh_fxn = lambda x_shift: self.geometry.hydraulic_diameter(
            time, x_shift + x_sf
        )
        self.A_fxn = lambda x_shift: self.geometry.area(time, x_shift + x_sf)
        self.Pwet_fxn = lambda x_shift: self.geometry.perimeter(time, x_shift + x_sf)
        if self.geometry.dlnA_dx is None:
            self.dlnA_dx_fxn = lambda _x_shift: 0.0
        else:
            self.dlnA_dx_fxn = lambda x_shift: self.geometry.dlnA_dx(
                time, x_shift + x_sf
            )
        self.A0 = float(self.A_fxn(0.0))
        Dh_sf = self.geometry.hydraulic_diameter(time, x_sf)
        wall = WallState.from_state(state_0, self.physics, Lc=Dh_sf, Tw=self.Tw)
        self.state_0 = state_0

        self.j0 = self.physics.get_density(state_0)[0] * (
            self.physics.get_velocity(state_0)[0] - us
        )
        self.cf0 = self.skin_friction_coefficient(wall)[0]
        self.us.append(us)

    def get_shock_properties(self, time: float, state: FluidState) -> bool:
        assert self.physics is not None
        prediction_active = bool(self.t_ps)

        self.clear_pseudoshock_setup()
        self._accepted_profile = None
        p = self.physics.get_pressure(state)
        u = self.physics.get_velocity(state)
        a = self.physics.get_sound_speed(state)

        if prediction_active:
            candidate = self.get_pseudoshock_shock_foot(time, state)
        else:
            s1, p_ratio, armed = self.get_planar_shock_foot(p, u, a)
            if not armed:
                return False
            x_sf = np.interp(s1, np.arange(len(self.x), dtype=np.float64), self.x)
            state_0 = interpolate_fluid_state(self.x, state, x_sf, self.physics)
            gamma1 = self.physics.get_gamma(state_0)[0]
            a1 = self.physics.get_sound_speed(state_0)[0]
            u1 = self.physics.get_velocity(state_0)[0]
            M1_rel = np.sqrt(
                p_ratio * (gamma1 + 1.0) / (2.0 * gamma1)
                + (gamma1 - 1.0) / (2.0 * gamma1)
            )
            us = u1 - a1 * M1_rel
            candidate = (x_sf, state_0, us, p_ratio)

        if candidate is None:
            return False

        if len(candidate) > 4:
            x_sf, state_0, us, _, x_ps, state_ps, AcA_ps = candidate
            self._accepted_profile = (x_ps, state_ps, AcA_ps)
        else:
            x_sf, state_0, us, _ = candidate
        a1 = self.physics.get_sound_speed(state_0)[0]
        M1_rel = (self.physics.get_velocity(state_0)[0] - us) / a1
        if not np.isfinite(M1_rel) or M1_rel < 1.3:
            self._accepted_profile = None
            return False

        self.set_pseudoshock_start(time, x_sf, state_0, us)
        self.x_sf.append(x_sf)
        return True

    def primitive_profile_state(
        self,
        x: float,
        AcA: float,
        u2: float,
        p: float,
    ) -> FluidState:
        assert self.physics is not None
        assert self.A_fxn is not None
        u = np.sqrt(u2)
        rho = self.j0 * self.A0 / (AcA * self.A_fxn(x) * u)
        composition = self.physics.get_composition(self.state_0)[0].reshape((1, -1))

        state = FluidState(
            shape=(1,),
            density=np.asarray([rho], dtype=np.float64),
            velocity=np.asarray([u], dtype=np.float64),
            pressure=np.asarray([p], dtype=np.float64),
            composition=composition,
        )
        state.temperature = self.physics.get_temperature(state)
        return state

    def get_dlnp_dx(
        self,
        x: float,
        AcA: float,
        state: FluidState,
    ) -> float:
        assert self.physics is not None
        assert self.A_fxn is not None
        rho = self.physics.get_density(state)[0]
        u = self.physics.get_velocity(state)[0]

        mu = self.physics.get_mu(state)[0]
        gamma = self.physics.get_gamma(state)[0]
        a = self.physics.get_sound_speed(state)[0]

        M2 = (u / a) ** 2
        D_c = np.sqrt(4.0 * AcA * self.A_fxn(x) / np.pi)  # streamtube diameter
        Re_Dc = rho * np.abs(u) * D_c / mu

        return float(
            self.Lambda_p * (2.0 * gamma / (gamma + 1.0)) * max(M2 - 1.0, 0.0)
            + (self.S_p / self.Gamma_p) ** 4
            * gamma
            * M2
            * Re_Dc ** (-4.0 * (1.0 - self.Omega_p))
        )

    def dydx(
        self,
        y: Array,
        x: float,
    ) -> Array:
        assert self.physics is not None
        assert self.Dh_fxn is not None
        assert self.Pwet_fxn is not None
        assert self.dlnA_dx_fxn is not None
        assert self.cf0 is not None
        p, u2, AcA = y

        Dh = self.Dh_fxn(x)
        Pwet = self.Pwet_fxn(x)
        dlnA_dx = self.dlnA_dx_fxn(x)
        state = self.primitive_profile_state(x, AcA, u2, p)
        wall = WallState.from_state(state, self.physics, Lc=Dh, Tw=self.Tw)
        wall.Cf = np.full(state.shape, self.cf0 * self.alpha_p)

        rho = self.physics.get_density(state)[0]
        T = self.physics.get_temperature(state)[0]
        cp = self.physics.get_cp(state)[0]
        G_p = self.get_dlnp_dx(x, AcA, state) / Dh

        Cf = wall.Cf[0]
        q_wall = 0.0
        if self.heat_flux is not None:
            q_wall = self.heat_flux(wall)[0]

        dp_dx = p * G_p
        du2_dx = -2.0 * p * G_p / (AcA * rho)
        du2_dx -= 4.0 * Cf * u2 / (AcA * Dh)

        dAcA_dx = AcA * (
            -G_p
            - dlnA_dx
            - q_wall * Pwet / (self.j0 * self.A0 * cp * T)
            - 0.5 * (1.0 + u2 / (cp * T)) * du2_dx / u2
        )

        return np.asarray(
            [
                dp_dx,
                du2_dx,
                dAcA_dx,
            ],
            dtype=np.float64,
        )

    def pseudoshock_solver(
        self,
        x: Array,
        record: bool = True,
    ) -> tuple[Array, FluidState, bool]:
        assert self.physics is not None
        assert self.A_fxn is not None
        state_0 = self.state_0

        y0 = np.asarray(
            [
                self.physics.get_pressure(state_0)[0],
                (self.physics.get_velocity(state_0)[0] - self.us[-1]) ** 2,
                1.0,
            ],
            dtype=np.float64,
        )

        x_out = [x[0]]
        y_out = [y0]
        separated = False
        reattached = False
        for x0, x1 in itertools.pairwise(x):
            y_prev = y_out[-1]
            y_next = rk2_step(self.dydx, y_prev, x0, x1 - x0)
            separated = separated or y_next[2] < 1.0
            if separated and y_prev[2] < 1.0 <= y_next[2]:
                f = (1.0 - y_prev[2]) / (y_next[2] - y_prev[2])
                x_out.append(x0 + f * (x1 - x0))
                y_out.append(y_prev + f * (y_next - y_prev))
                y_out[-1][2] = 1.0
                reattached = True
                break
            x_out.append(x1)
            y_out.append(y_next)

        x_out = np.asarray(x_out, dtype=np.float64)
        y_out = np.vstack(y_out)

        P_out = y_out[:, 0]
        u2_out = y_out[:, 1]
        AcA_out = y_out[:, 2]
        self.AcA_ps = AcA_out
        u_rel = np.sqrt(u2_out)

        rho = self.j0 * self.A0 / (AcA_out * self.A_fxn(x_out) * u_rel)
        u = u_rel + self.us[-1]
        composition = np.tile(self.physics.get_composition(state_0)[0], (x_out.size, 1))

        state_ps = FluidState(
            shape=(x_out.size,),
            density=rho,
            velocity=u,
            pressure=P_out,
            composition=composition,
        )
        state_ps.temperature = self.physics.get_temperature(state_ps)
        state_ps.internal_energy = self.physics.get_internal_energy(state_ps)

        if record:
            self.p2p1.append(P_out[-1] / P_out[0])
            self.L_ps.append(x_out[-1])
        return x_out, state_ps, reattached

    def estimate_dt_max(self, state: FluidState) -> float:
        assert self.physics is not None
        ld_max = np.max(
            np.abs(self.physics.get_velocity(state))
            + self.physics.get_sound_speed(state)
        )
        return float(np.min(self.dx_cells) / ld_max)

    def get_backpressure(
        self, time: float, state_ss: FluidState, x_tail: float
    ) -> float:
        assert self.physics is not None
        assert self.geometry is not None
        x_tail += float(self.geometry.hydraulic_diameter(time, x_tail) / 4.0)
        M_ss = self.physics.get_velocity(state_ss) / self.physics.get_sound_speed(
            state_ss
        )
        subsonic_post_ps = (M_ss <= 1.0) & (self.x >= x_tail)

        idx = np.nonzero(subsonic_post_ps)[0]
        if idx.size == 0:
            idx = np.nonzero(self.x >= x_tail)[0]
        if idx.size == 0:
            idx = np.asarray([len(self.x) - 1], dtype=np.int64)

        return np.max(self.physics.get_pressure(state_ss)[idx])

    def get_pseudoshock_shock_foot(
        self, time: float, state: FluidState
    ) -> _Accept | None:
        assert self.physics is not None
        x_tail = self.L_ps[-1] + self.x_sf[-1]
        p_b = self.get_backpressure(time, state, x_tail)
        dt = time - self.t_ps[-1]
        x_min = float(self.x[0])
        x_max = float(self.x[-2])
        x_sf_guess = float(np.clip(self.x_sf[-1] + self.us[-1] * dt, x_min, x_max))
        dx_min = float(np.min(self.dx_cells))
        travel = abs(self.us[-1] * dt)
        search_width = max(
            self.shock_search_width_factor * travel,
            self.shock_search_min_cells * dx_min,
        )
        x_lo = max(x_min, x_sf_guess - search_width)
        x_hi = min(x_max, x_sf_guess + search_width)

        eval_cache: dict[float, _Trial | None] = {}

        def evaluate(x_sf: float) -> _Trial | None:
            key = round(x_sf, 12)
            if key in eval_cache:
                return eval_cache[key]
            trial: _Trial | None = None
            if x_sf < x_min or x_sf > x_max:
                eval_cache[key] = trial
                return trial
            state_0 = interpolate_fluid_state(self.x, state, x_sf, self.physics)
            gamma1 = self.physics.get_gamma(state_0)[0]
            a1 = self.physics.get_sound_speed(state_0)[0]
            u1 = self.physics.get_velocity(state_0)[0]
            p_ratio = p_b / self.physics.get_pressure(state_0)[0]
            if p_ratio <= 1.0:
                eval_cache[key] = trial
                return trial
            M1_rel = np.sqrt(
                p_ratio * (gamma1 + 1.0) / (2.0 * gamma1)
                + (gamma1 - 1.0) / (2.0 * gamma1)
            )
            us = u1 - a1 * M1_rel
            x_after = self.x[self.x > x_sf] - x_sf
            x_local = np.concatenate((np.asarray([0.0], dtype=np.float64), x_after))
            if x_local.size < 2:
                eval_cache[key] = trial
                return trial
            n_us = len(self.us)
            self.set_pseudoshock_start(time, x_sf, state_0, us)
            x_ps, state_ps, _ = self.pseudoshock_solver(x_local, record=False)
            AcA_ps = self.AcA_ps.copy()
            del self.us[n_us:]
            self.clear_pseudoshock_setup()
            if x_sf + x_ps[-1] > self.x[-1]:
                eval_cache[key] = trial
                return trial
            err = self.physics.get_pressure(state_ps)[-1] / p_b - 1.0
            trial = x_sf, state_0, us, p_ratio, x_ps, state_ps, AcA_ps, err
            eval_cache[key] = trial
            return trial

        def accept(trial: _Trial) -> _Accept:
            self.p_res.append(trial[-1])
            return trial[:-1]

        best = evaluate(x_sf_guess)
        if best is not None and abs(best[-1]) <= self.p_res_max:
            return accept(best)

        x_samples = np.unique(
            np.concatenate(
                (
                    np.asarray([x_sf_guess], dtype=np.float64),
                    np.linspace(x_lo, x_hi, self.n_shock_root_samples),
                )
            )
        )
        prev = None
        for x_sample in x_samples:
            trial = evaluate(float(x_sample))
            if trial is None:
                continue
            if best is None or abs(trial[-1]) < abs(best[-1]):
                best = trial
            if abs(trial[-1]) <= self.p_res_max:
                return accept(trial)
            if prev is not None and prev[-1] * trial[-1] <= 0.0:
                lo = trial
                hi = prev
                for _ in range(self.n_shock_root_iterations):
                    mid = evaluate(0.5 * (lo[0] + hi[0]))
                    if mid is None:
                        break
                    if best is None or abs(mid[-1]) < abs(best[-1]):
                        best = mid
                    if abs(mid[-1]) <= self.p_res_max:
                        return accept(mid)
                    if lo[-1] * mid[-1] <= 0.0:
                        hi = mid
                    else:
                        lo = mid
                break
            prev = trial

        if best is not None and abs(best[-1]) <= self.p_res_max:
            return accept(best)
        return None

    def assemble_corrective_pseudoshock_source(
        self,
        state_array_local: Array,
        state: FluidState,
        x_cells: Array,
        x_ps: Array,
        state_ps: FluidState,
    ) -> Array:
        assert self.physics is not None
        target_state = interpolate_fluid_state(x_ps, state_ps, x_cells, self.physics)
        target_array = self.physics.primitive_to_conservative(target_state)
        tau_ps = 2.0 * self.estimate_dt_max(state)

        return (target_array - state_array_local) / tau_ps

    def source_implementation(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        if state_array_local is None:
            return np.zeros(self.shape_input)
        assert state is not None
        assert self.physics is not None

        # Default to underlying boundary layer model everywhere
        state_array_local = np.reshape(state_array_local, self.shape_input)
        rhs = np.zeros_like(state_array_local)

        rhs_bl = self.boundary_layer.source_implementation(
            time,
            state_array_local,
            state,
            face_states,
            avg_face_states,
            face_gradients,
        ).reshape(len(self.x), 2)
        rhs = np.reshape(self.boundary_layer.add_source(rhs, rhs_bl), self.shape_input)

        # Check for presence of shock
        if not self.get_shock_properties(time, state):
            return rhs

        x_sf = self.x_sf[-1]
        s_idx = int(np.searchsorted(self.x, x_sf, side="left"))
        x_cells = self.x[s_idx:] - x_sf
        if x_cells.size < 1:
            self._accepted_profile = None
            return rhs
        x_local = x_cells
        if x_local[0] > 0.0:
            x_local = np.concatenate((np.asarray([0.0], dtype=np.float64), x_local))
        if x_local.size < 2:
            self._accepted_profile = None
            return rhs

        accepted_profile = self._accepted_profile
        self._accepted_profile = None
        if accepted_profile is None:
            x_ps, state_ps, _ = self.pseudoshock_solver(x_local)
        else:
            x_ps, state_ps, AcA_ps = accepted_profile
            p = self.physics.get_pressure(state_ps)
            self.AcA_ps = AcA_ps
            self.p2p1.append(p[-1] / p[0])
            self.L_ps.append(x_ps[-1])
        if x_ps.size < 2:
            return rhs

        idx_local = np.nonzero(x_cells <= x_ps[-1])[0]
        idx_ps = s_idx + idx_local
        x_cells = x_cells[idx_local]
        if idx_ps.size < 2:
            return rhs

        self.t_ps.append(time)

        rhs[idx_ps, :] = self.assemble_corrective_pseudoshock_source(
            state_array_local[idx_ps], state, x_cells, x_ps, state_ps
        )
        return rhs
