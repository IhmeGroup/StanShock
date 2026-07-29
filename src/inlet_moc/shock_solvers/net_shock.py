from __future__ import annotations

from pathlib import Path

import numpy as np

from inlet_moc.char_net import CharNet, _wall_clearance
from inlet_moc.planar_inlet import PlanarInlet
from inlet_moc.rotational_solvers import field_point_rot, wall_point_rot
from inlet_moc.shock_solvers.char_shock import (
    InsufficientCharPtsLError,
    ShockPoint,
    shock_field,
    shock_from_wall,
    shock_origin,
    shock_to_wall,
)
from inlet_moc.shock_solvers.net_logic_helpers import (
    _active_free_indices,
    clip_coalescing_same_family_chars,
    enforce_net_bounds,
    get_char_interp_segment,
    remove_spurious_preshock_pts,
    restore_resolution,
)
from inlet_moc.solver_events import SolverEvent, check_shock_reflection
from inlet_moc.utils_moc import get_tang, intersect_line_polyline


class NetShockSolver:
    def __init__(
        self,
        net: CharNet,
        inlet: PlanarInlet,
        event: SolverEvent,
        max_iters: int = 5,
        tol: float = 1e-6,
        figdir: str | Path | None = None,
        plot: bool = False,
    ) -> None:
        self.net_L = net
        self.inlet = inlet
        self.event = event
        self.max_iters = max_iters
        self.tol = tol
        self.plot = bool(plot)
        self.output_figdir = None if figdir is None else Path(figdir)
        self.figdir = None if (figdir is None or not self.plot) else self.output_figdir

        self.wall_from = event.wall_from
        self.wall_to = event.wall_to

        self.L_stencil_max = 10
        self.R_stencil_max = 10

        self.net_R: CharNet | None = None
        self.net_R_test: CharNet | None = None
        self.gamma = self.net_L.gamma

        self.shock_pairs: list[ShockPoint] = []
        self.xy_shock_state = np.empty((0, 6), dtype=float)
        self.xy_mesh_state = np.empty((0, 6), dtype=float)
        self.ij_shock_L = np.empty((0, 2), dtype=int)
        self.ij_shock_R = np.empty((0, 2), dtype=int)
        self.ij_mesh_R = np.empty((0, 2), dtype=int)

        self.shock_plotter: object | None = None
        self.shock_resample_N = self.net_L.N

    def _reset_common_state(self) -> None:
        self.net_R = CharNet(
            self.net_L.N,
            self.inlet,
            self.net_L.gamma,
            max_iters=self.max_iters,
            tol=self.tol,
        )
        self.net_R.wall = self.wall_to
        self.net_R_test = None

        self.shock_pairs = []
        self.xy_shock_state = np.empty((0, 6), dtype=float)
        self.ij_shock_L = np.empty((0, 2), dtype=int)
        self.ij_shock_R = np.empty((0, 2), dtype=int)

        self.xy_mesh_state = np.empty((0, 6), dtype=float)
        self.ij_mesh_R = np.empty((0, 2), dtype=int)
        self.gamma = self.net_L.gamma

    def clip_and_clean_nets(self, family: str) -> None:
        downstream_family = "cplus" if family == "cminus" else "cminus"
        enforce_net_bounds(
            self.net_L,
            family=family,
            wall=self.wall_to,
            xy_shock=self.xy_shock_state[:, :2],
            ij_shock=self.ij_shock_L,
            tol=self.tol,
            **self._cleanup_debug_kwargs(),
        )
        clip_coalescing_same_family_chars(
            self.net_L,
            family=family,
            tol=self.tol,
            **self._cleanup_debug_kwargs(),
        )
        clip_coalescing_same_family_chars(
            self.net_R,
            family=downstream_family,
            tol=self.tol,
            **self._cleanup_debug_kwargs(),
        )

    def _downstream_active_idx(self, idx: int, family: str) -> np.ndarray:
        if (idx < 0) or (idx >= self.net_R.N):
            return np.array([], dtype=int)
        if family == "cminus":
            return np.flatnonzero(self.net_R.state_mask()[idx, :])
        return np.flatnonzero(self.net_R.state_mask()[:, idx])

    def _get_from_wall_solve_idx(
        self,
        event_idx: int,
        fixed_idx: int,
        family: str,
    ) -> int:
        if family == "cminus":
            for i_step in range(int(event_idx) + 1, self.net_L.N):
                if self.net_L.has_point(i_step, fixed_idx - 1) and self.net_L.has_point(
                    i_step, fixed_idx
                ):
                    return i_step
            msg = (
                f"No valid from-wall C- solve index exists beyond event i={event_idx} "
                f"at j={fixed_idx}."
            )
            raise InsufficientCharPtsLError(msg)

        for j_step in range(int(event_idx) + 1, self.net_L.N):
            if self.net_L.has_point(fixed_idx - 1, j_step) and self.net_L.has_point(
                fixed_idx, j_step
            ):
                return j_step
        msg = (
            f"No valid from-wall C+ solve index exists beyond event j={event_idx} "
            f"at i={fixed_idx}."
        )
        raise InsufficientCharPtsLError(msg)

    def _get_from_wall_solve_indices(
        self,
        event_idx: int,
        fixed_idx: int,
        family: str,
        shock_im1: ShockPoint,
    ) -> tuple[int, int, bool]:
        if family == "cminus":
            for i_step in range(int(event_idx) + 1, self.net_L.N):
                if self.net_L.has_point(i_step, fixed_idx - 1) and self.net_L.has_point(
                    i_step, fixed_idx
                ):
                    return i_step, fixed_idx, False

                active_j = _active_free_indices(self.net_L, i_step, "cplus")
                if active_j is None or active_j.size == 0:
                    continue
                j_step = int(active_j[-1])
                if j_step >= fixed_idx or j_step < 1:
                    continue
                char_pts_L, _ = get_char_interp_segment(
                    self.net_L,
                    i_step,
                    j_step,
                    "cplus",
                    self.L_stencil_max,
                    "upstream",
                )
                if self._field_left_intersects(char_pts_L, shock_im1):
                    return i_step, j_step, True

            msg = (
                f"No valid from-wall C- solve index exists beyond event i={event_idx} "
                f"near j={fixed_idx}."
            )
            raise InsufficientCharPtsLError(msg)

        for j_step in range(int(event_idx) + 1, self.net_L.N):
            if self.net_L.has_point(fixed_idx - 1, j_step) and self.net_L.has_point(
                fixed_idx, j_step
            ):
                return fixed_idx, j_step, False

            active_i = _active_free_indices(self.net_L, j_step, "cminus")
            if active_i is None or active_i.size == 0:
                continue
            i_step = int(active_i[-1])
            if i_step >= fixed_idx or i_step < 1:
                continue
            char_pts_L, _ = get_char_interp_segment(
                self.net_L,
                j_step,
                i_step,
                "cminus",
                self.L_stencil_max,
                "upstream",
            )
            if self._field_left_intersects(char_pts_L, shock_im1):
                return i_step, j_step, True

        msg = (
            f"No valid from-wall C+ solve index exists beyond event j={event_idx} "
            f"near i={fixed_idx}."
        )
        raise InsufficientCharPtsLError(msg)

    def _field_stencil_length(
        self,
        net: CharNet,
        idx_fixed: int,
        pt_idx: int,
        family: str,
        mode: str,
        stencil_max: int,
    ) -> int:
        active_idx = _active_free_indices(net, int(idx_fixed), family)
        if active_idx is None or active_idx.size == 0:
            return 1

        active_idx = np.asarray(active_idx, dtype=int)
        pt_idx = int(pt_idx)

        if mode == "upstream":
            pos = np.searchsorted(active_idx, pt_idx)
            if pos >= active_idx.size or active_idx[pos] != pt_idx:
                return 1
            available = pos + 1
        else:
            pos = np.searchsorted(active_idx, pt_idx, side="left")
            if pos >= active_idx.size:
                return 1
            available = active_idx.size - pos
        return max(1, min(int(stencil_max), int(available)))

    def _field_left_intersects(
        self,
        char_pts_L: np.ndarray,
        shock_im1: ShockPoint,
    ) -> bool:
        if char_pts_L.shape[0] == 0:
            return False
        eqn_shock = get_tang(shock_im1.pt_pre, np.tan(shock_im1.beta))
        return intersect_line_polyline(eqn_shock, char_pts_L, self.tol / 10) is not None

    def _build_downstream_cplus(
        self,
        i_fixed: int,
        start_idx_min: int = 1,
    ) -> int:  # consider relocating/merging with the functions in CharNet?
        """
        Extend one post-shock downstream row as a C+ characteristic returning
        to ``wall_from``.
        """

        j_active = self._downstream_active_idx(i_fixed, "cminus")
        j_active = j_active[j_active >= int(start_idx_min)]

        if (i_fixed < 1) or (j_active.size == 0):
            return 1

        j_m1 = int(j_active[-1])
        j_start = j_m1 + 1

        for j in range(j_start, self.net_R.N):
            has_plus = self.net_R.has_point(i_fixed, j - 1)
            has_minus = self.net_R.has_point(i_fixed - 1, j)
            if not has_plus:
                break

            pt_plus = self.net_R.get_point(i_fixed, j - 1)

            if has_minus and has_plus:
                pt_minus = self.net_R.get_point(i_fixed - 1, j)
                pt_out = field_point_rot(
                    pt_plus,
                    pt_minus,
                    self.gamma,
                    max_iters=self.max_iters,
                    tol=self.tol,
                )

                if _wall_clearance(pt_out, self.wall_from) <= self.tol:
                    if self.net_R.has_point(i_fixed - 1, j) and self.net_R.has_point(
                        i_fixed - 1, j - 1
                    ):
                        pt_char = self.net_R.get_point(i_fixed - 1, j)
                        pt_wall_anchor = self.net_R.get_point(i_fixed - 1, j - 1)
                    elif self.net_R.has_point(i_fixed - 1, j - 1):
                        pt_char = pt_plus
                        pt_wall_anchor = self.net_R.get_point(i_fixed - 1, j - 1)
                    else:
                        break
                    pt_out = wall_point_rot(
                        pt_char,
                        pt_wall_anchor,
                        self.wall_from,
                        "cplus",
                        self.gamma,
                        max_iters=self.max_iters,
                        tol=self.tol,
                    )
                    self.net_R.edit_point(i_fixed, j, pt_out, 1)
                    j_m1 = j
                    break

                self.net_R.edit_point(i_fixed, j, pt_out, 0)
                j_m1 = j
                continue

            if _wall_clearance(pt_plus, self.wall_from) <= self.tol:
                break

            if self.net_R.has_point(i_fixed - 1, j) and self.net_R.has_point(
                i_fixed - 1, j - 1
            ):
                pt_char = self.net_R.get_point(i_fixed - 1, j)
                pt_wall_anchor = self.net_R.get_point(i_fixed - 1, j - 1)
            elif self.net_R.has_point(i_fixed - 1, j - 1):
                pt_char = pt_plus
                pt_wall_anchor = self.net_R.get_point(i_fixed - 1, j - 1)
            else:
                break

            pt_wall = wall_point_rot(
                pt_char,
                pt_wall_anchor,
                self.wall_from,
                "cplus",
                self.gamma,
                max_iters=self.max_iters,
                tol=self.tol,
            )
            self.net_R.edit_point(i_fixed, j, pt_wall, 1)
            j_m1 = j
            break

        return j_m1

    def _build_downstream_cminus(
        self,
        j_fixed: int,
        start_idx_min: int = 1,
    ) -> int:
        """
        Build cminus char
        """
        i_active = self._downstream_active_idx(j_fixed, "cplus")
        i_active = i_active[i_active >= int(start_idx_min)]

        if (j_fixed < 1) or i_active.size == 0:
            return 1

        i_m1 = int(i_active[-1])
        i_start = i_m1 + 1

        for i in range(i_start, self.net_R.N):
            has_minus = self.net_R.has_point(i - 1, j_fixed)
            has_plus = self.net_R.has_point(i, j_fixed - 1)
            if not has_minus:
                break

            pt_minus = self.net_R.get_point(i - 1, j_fixed)

            if has_plus:
                pt_plus = self.net_R.get_point(i, j_fixed - 1)
                pt_out = field_point_rot(
                    pt_plus,
                    pt_minus,
                    self.gamma,
                    max_iters=self.max_iters,
                    tol=self.tol,
                )
                if _wall_clearance(pt_out, self.wall_from) <= self.tol:
                    if self.net_R.has_point(i, j_fixed - 1) and self.net_R.has_point(
                        i - 1, j_fixed - 1
                    ):
                        pt_char = self.net_R.get_point(i, j_fixed - 1)
                        pt_wall_anchor = self.net_R.get_point(i - 1, j_fixed - 1)
                    elif self.net_R.has_point(i - 1, j_fixed - 1):
                        pt_char = pt_minus
                        pt_wall_anchor = self.net_R.get_point(i - 1, j_fixed - 1)
                    else:
                        break
                    pt_out = wall_point_rot(
                        pt_char,
                        pt_wall_anchor,
                        self.wall_from,
                        "cminus",
                        self.gamma,
                        max_iters=self.max_iters,
                        tol=self.tol,
                    )
                    self.net_R.edit_point(i, j_fixed, pt_out, 1)
                    i_m1 = i
                    break

                self.net_R.edit_point(i, j_fixed, pt_out, 0)
                i_m1 = i
                continue

            if _wall_clearance(pt_minus, self.wall_from) <= self.tol:
                break

            if self.net_R.has_point(i, j_fixed - 1) and self.net_R.has_point(
                i - 1, j_fixed - 1
            ):
                pt_char = self.net_R.get_point(i, j_fixed - 1)
                pt_wall_anchor = self.net_R.get_point(i - 1, j_fixed - 1)
            elif self.net_R.has_point(i - 1, j_fixed - 1):
                pt_char = pt_minus
                pt_wall_anchor = self.net_R.get_point(i - 1, j_fixed - 1)
            else:
                break

            pt_wall = wall_point_rot(
                pt_char,
                pt_wall_anchor,
                self.wall_from,
                "cminus",
                self.gamma,
                max_iters=self.max_iters,
                tol=self.tol,
            )
            self.net_R.edit_point(i, j_fixed, pt_wall, 1)
            i_m1 = i
            break

        return i_m1

    def record_cminus_step(
        self,
        i_step: int,
        j_L_s: int,  # Pre-shock idx net_L
        j_R_s: int,  # Post-shock idx net_R (usually 0)
        shock_pair: ShockPoint,
        pt_type_s: int,
        j_R_m: int | None = None,  # Post-shock idx mesh net_R (>= 1)
        mesh_pt: np.ndarray | None = None,
        pt_type_m: None | int = 0,
    ) -> None:
        self.net_L.edit_point(
            i_step,
            j_L_s,
            shock_pair.pt_pre,
            pt_type_s,
        )
        self.net_R.edit_point(
            i_step,
            j_R_s,
            shock_pair.pt_post,
            pt_type_s,
        )

        self._append_shock_record([i_step, j_L_s], [i_step, j_R_s], shock_pair)
        if mesh_pt is not None:
            self.net_R.edit_point(
                i_step,
                j_R_m,
                mesh_pt,
                pt_type_m,
            )
            if self.shock_plotter is not None:
                self.shock_plotter.add_step(
                    shock_pt=shock_pair.pt_pre,
                    shock_idx=(int(i_step), int(j_L_s)),
                    mesh_pt=mesh_pt,
                    mesh_idx=(int(i_step), int(j_R_m)),
                )

    def record_cplus_step(
        self,
        j_step: int,
        i_L_s: int,
        i_R_s: int,
        shock_pair: ShockPoint,
        pt_type_s: int,
        i_R_m: int | None = None,
        mesh_pt: np.ndarray | None = None,
        pt_type_m: int | None = 0,
    ) -> None:
        self.net_L.edit_point(
            i_L_s,
            j_step,
            shock_pair.pt_pre,
            pt_type_s,
        )
        self.net_R.edit_point(
            i_R_s,
            j_step,
            shock_pair.pt_post,
            pt_type_s,
        )
        self._append_shock_record([i_L_s, j_step], [i_R_s, j_step], shock_pair)
        if mesh_pt is not None:
            self.net_R.edit_point(
                i_R_m,
                j_step,
                mesh_pt,
                pt_type_m,
            )
            if self.shock_plotter is not None:
                self.shock_plotter.add_step(
                    shock_pt=shock_pair.pt_pre,
                    shock_idx=(int(i_L_s), int(j_step)),
                    mesh_pt=mesh_pt,
                    mesh_idx=(int(i_R_m), int(j_step)),
                )

    def _append_shock_record(
        self,
        ij_L: tuple[int, int],
        ij_s_R: tuple[int, int],
        shock_pair: ShockPoint,
        ij_m_R: tuple[int, int] | None = None,
        mesh_pt: np.ndarray | None = None,
    ) -> None:
        self.shock_pairs.append(shock_pair)
        self.xy_shock_state = np.vstack(
            (self.xy_shock_state, np.asarray(shock_pair.pt_post)[None, :])
        )
        self.ij_shock_L = np.vstack((self.ij_shock_L, ij_L))

        self.ij_shock_R = np.vstack((self.ij_shock_R, ij_s_R))
        if mesh_pt is not None:
            self.ij_mesh_R = np.vstack((self.ij_mesh_R, ij_m_R))
            self.xy_mesh_state = np.vstack((self.xy_mesh_state, mesh_pt))

    def _start_shock_plot(self, family: str) -> None:
        if not self.plot:
            self.shock_plotter = None
            return
        import matplotlib.pyplot as plt

        from inlet_moc.plot.net_shock import ShockNetPlotter

        plt.close("all")
        self.shock_plotter = ShockNetPlotter(
            self.net_L,
            self.net_R,
            family=family,
            minimal=True,
        )

    def _net_shock_plot_axes(self):
        if self.shock_plotter is None:
            return None, None
        return self.shock_plotter.fig, self.shock_plotter.ax

    def _refresh_shock_plot(self) -> None:
        if self.shock_plotter is None:
            return
        self.shock_plotter.redraw()

    def _restore_and_propagate_downstream(
        self,
        family: str,
    ) -> bool:
        n_shock_pts = int(self.ij_shock_R.shape[0])
        if n_shock_pts >= self.net_L.N:
            return False

        shock_state_q, _, net_R_res, iter_idx = restore_resolution(
            self.net_R,
            self.xy_shock_state,
            self.ij_shock_R,
            self.shock_resample_N,
        )

        net_R_original = self.net_R
        self.net_R = net_R_res
        if self.shock_plotter is not None:
            self.shock_plotter.net_R = net_R_res
            self.shock_plotter.set_resampled_points(shock_state_q)
        try:
            if family == "cminus":
                for i_fixed in iter_idx:
                    self._build_downstream_cplus(
                        int(i_fixed),
                        start_idx_min=0,
                    )
                remove_spurious_preshock_pts(
                    net_R_res,
                    "cplus",
                    shock_state_q,
                    tol=self.tol,
                )
                self.net_R_test = net_R_res
                return True

            if family == "cplus":
                for j_fixed in iter_idx:
                    self._build_downstream_cminus(
                        int(j_fixed),
                        start_idx_min=0,
                    )
                remove_spurious_preshock_pts(
                    net_R_res,
                    "cminus",
                    shock_state_q,
                    tol=self.tol,
                )
                self.net_R_test = net_R_res
                return True
        except Exception:
            self.net_R = net_R_original
            raise

        msg = f"Unsupported downstream family '{family}'."
        raise ValueError(msg)

    def _restored_reflection_idx(self, family: str) -> tuple[int, int]:
        if family == "cminus":
            return self.net_R.N - 1, 0
        if family == "cplus":
            return 0, self.net_R.N - 1
        msg = f"Unsupported downstream family '{family}'."
        raise ValueError(msg)

    def solve_cminus(self) -> tuple[CharNet, CharNet, SolverEvent | None]:
        # j_L = ind on left net.
        family = "cminus"
        i_L, j_L_s = self.event.point_idx

        j_R_s = 0
        j_R_m = 1

        L_stencil = self.L_stencil_max

        self._reset_common_state()
        self.net_R.idl_kind = family
        self._start_shock_plot(family)

        shock_pt = shock_origin(
            self.net_L.get_point(i_L, j_L_s),
            self.wall_from,
            self.gamma,
        )

        self.record_cminus_step(i_L, j_L_s, j_R_s, shock_pt, 3)

        i_L, j_L_s, used_edge_stencil = self._get_from_wall_solve_indices(
            event_idx=self.event.point_idx[0],
            fixed_idx=j_L_s,
            family=family,
            shock_im1=shock_pt,
        )

        # NOTE: right net shock always gets j = 0
        char_pts_L, idx_L = get_char_interp_segment(
            self.net_L, i_L, j_L_s, "cplus", L_stencil, "upstream"
        )

        shock_pair, mesh_pt = shock_from_wall(
            self.shock_pairs[-1],
            family,
            char_pts_L,
            self.wall_from,
            self.gamma,
            **self._cminus_solver_kwargs(),
        )

        self.record_cminus_step(i_L, j_L_s, j_R_s, shock_pair, 2, j_R_m, mesh_pt, 1)

        pt_type_s = 2
        pt_type_m = 0

        active_i = _active_free_indices(self.net_L, j_L_s, family)
        if used_edge_stencil:
            i_stop = self.net_L.N - 1
        else:
            terminal_offset = min(i_L + 1, int(active_i.size))
            i_stop = int(active_i[-terminal_offset])

        i_L += 1

        for i in range(i_L, i_stop + 1):
            L_stencil = self._field_stencil_length(
                self.net_L,
                i,
                j_L_s,
                "cplus",
                "upstream",
                self.L_stencil_max,
            )
            char_pts_L, idx_L = get_char_interp_segment(
                self.net_L, i, j_L_s, "cplus", L_stencil, "upstream"
            )
            if used_edge_stencil and char_pts_L.shape[0] == 0:
                active_j = _active_free_indices(self.net_L, i, "cplus")
                if active_j is not None and active_j.size > 0:
                    j_edge = int(active_j[-1])
                    if 0 < j_edge < j_L_s:
                        j_L_s = j_edge
                        L_stencil = self._field_stencil_length(
                            self.net_L,
                            i,
                            j_L_s,
                            "cplus",
                            "upstream",
                            self.L_stencil_max,
                        )
                        char_pts_L, idx_L = get_char_interp_segment(
                            self.net_L,
                            i,
                            j_L_s,
                            "cplus",
                            L_stencil,
                            "upstream",
                        )

            R_stencil = self._field_stencil_length(
                self.net_R,
                i - 1,
                j_R_m,
                "cplus",
                "downstream",
                self.R_stencil_max,
            )
            char_pts_R, idx_R = get_char_interp_segment(
                self.net_R, i - 1, j_R_m, "cplus", R_stencil, "downstream"
            )

            shock_im1 = self.shock_pairs[-1]

            if char_pts_L.shape[0] == 0:
                msg = f"No active C+ left-net stencil exists at i={i}, j={j_L_s}."
                raise InsufficientCharPtsLError(msg)

            left_intersects = self._field_left_intersects(char_pts_L, shock_im1)
            terminal_step = (
                (char_pts_L.shape[0] <= 1)
                or (not left_intersects)
                or ((not used_edge_stencil) and (i == i_stop))
            )
            terminal_wall = self.wall_from if used_edge_stencil else self.wall_to

            if terminal_step:
                shock_pair, mesh_pt, j_R_m = shock_to_wall(
                    shock_im1,
                    family,
                    char_pts_L[0, :],
                    char_pts_R,
                    idx_R,
                    terminal_wall,
                    self.gamma,
                    idx_R_min=1,
                    **self._cminus_solver_kwargs(),
                )
                pt_type_s = 3
            else:
                shock_pair, j_L_s, mesh_pt, j_R_m = shock_field(
                    shock_im1,
                    family,
                    char_pts_L,
                    idx_L,
                    char_pts_R,
                    idx_R,
                    self.gamma,
                    self.wall_from.normal_sign,
                    idx_R_min=1,
                    **self._cminus_solver_kwargs(),
                )

            self.record_cminus_step(
                i, j_L_s, j_R_s, shock_pair, pt_type_s, j_R_m, mesh_pt, pt_type_m
            )

            self._build_downstream_cplus(i)

            if terminal_step:
                break

        restored = self._restore_and_propagate_downstream(family)
        self.clip_and_clean_nets(family=family)
        self._refresh_shock_plot()
        reflected_event = check_shock_reflection(
            shock_pair=shock_pair,
            point_idx=self._restored_reflection_idx(family) if restored else (i, j_R_s),
            wall_from=terminal_wall,
            inlet=self.inlet,
            next_family="cplus",
            tol=self.tol,
        )
        return self.net_L, self.net_R, reflected_event

    def solve_cplus(self) -> tuple[CharNet, CharNet, SolverEvent | None]:
        family = "cplus"
        i_L_s, j_L = self.event.point_idx

        i_R_s = 0
        i_R_m = 1

        L_stencil = self.L_stencil_max

        self._reset_common_state()
        self.net_R.idl_kind = family
        self._start_shock_plot(family)

        shock_pt = shock_origin(
            self.net_L.get_point(i_L_s, j_L),
            self.wall_from,
            self.gamma,
        )

        self.record_cplus_step(j_L, i_L_s, i_R_s, shock_pt, 3)

        i_L_s, j_L, used_edge_stencil = self._get_from_wall_solve_indices(
            event_idx=self.event.point_idx[1],
            fixed_idx=i_L_s,
            family=family,
            shock_im1=shock_pt,
        )

        char_pts_L, idx_L = get_char_interp_segment(
            self.net_L, j_L, i_L_s, "cminus", L_stencil, "upstream"
        )

        shock_pair, mesh_pt = shock_from_wall(
            self.shock_pairs[-1],
            family,
            char_pts_L,
            self.wall_from,
            self.gamma,
            **self._cplus_solver_kwargs(),
        )

        self.record_cplus_step(j_L, i_L_s, i_R_s, shock_pair, 2, i_R_m, mesh_pt, 1)

        pt_type_s = 2
        pt_type_m = 0

        active_j = _active_free_indices(self.net_L, i_L_s, family)
        if used_edge_stencil:
            j_stop = self.net_L.N - 1
        else:
            terminal_offset = min(j_L + 1, int(active_j.size))
            j_stop = int(active_j[-terminal_offset])

        j_L += 1

        for j in range(j_L, j_stop + 1):
            L_stencil = self._field_stencil_length(
                self.net_L,
                j,
                i_L_s,
                "cminus",
                "upstream",
                self.L_stencil_max,
            )
            char_pts_L, idx_L = get_char_interp_segment(
                self.net_L, j, i_L_s, "cminus", L_stencil, "upstream"
            )
            if used_edge_stencil and char_pts_L.shape[0] == 0:
                active_i = _active_free_indices(self.net_L, j, "cminus")
                if active_i is not None and active_i.size > 0:
                    i_edge = int(active_i[-1])
                    if 0 < i_edge < i_L_s:
                        i_L_s = i_edge
                        L_stencil = self._field_stencil_length(
                            self.net_L,
                            j,
                            i_L_s,
                            "cminus",
                            "upstream",
                            self.L_stencil_max,
                        )
                        char_pts_L, idx_L = get_char_interp_segment(
                            self.net_L,
                            j,
                            i_L_s,
                            "cminus",
                            L_stencil,
                            "upstream",
                        )

            R_stencil = self._field_stencil_length(
                self.net_R,
                j - 1,
                i_R_m,
                "cminus",
                "downstream",
                self.R_stencil_max,
            )
            char_pts_R, idx_R = get_char_interp_segment(
                self.net_R, j - 1, i_R_m, "cminus", R_stencil, "downstream"
            )

            shock_im1 = self.shock_pairs[-1]

            if char_pts_L.shape[0] == 0:
                msg = f"No active C- left-net stencil exists at i={i_L_s}, j={j}."
                raise InsufficientCharPtsLError(msg)

            left_intersects = self._field_left_intersects(char_pts_L, shock_im1)
            terminal_step = (
                (char_pts_L.shape[0] <= 1)
                or (not left_intersects)
                or ((not used_edge_stencil) and (j == j_stop))
            )
            terminal_wall = self.wall_from if used_edge_stencil else self.wall_to

            if terminal_step:
                shock_pair, mesh_pt, i_R_m = shock_to_wall(
                    shock_im1,
                    family,
                    char_pts_L[0, :],
                    char_pts_R,
                    idx_R,
                    terminal_wall,
                    self.gamma,
                    idx_R_min=1,
                    **self._cplus_solver_kwargs(),
                )
                pt_type_s = 3
            else:
                shock_pair, i_L_s, mesh_pt, i_R_m = shock_field(
                    shock_im1,
                    family,
                    char_pts_L,
                    idx_L,
                    char_pts_R,
                    idx_R,
                    self.gamma,
                    self.wall_from.normal_sign,
                    idx_R_min=1,
                    **self._cplus_solver_kwargs(),
                )

            self.record_cplus_step(
                j, i_L_s, i_R_s, shock_pair, pt_type_s, i_R_m, mesh_pt, pt_type_m
            )

            self._build_downstream_cminus(j)

            if terminal_step:
                break

        restored = self._restore_and_propagate_downstream(family)
        self.clip_and_clean_nets(family=family)
        self._refresh_shock_plot()
        reflected_event = check_shock_reflection(
            shock_pair=shock_pair,
            point_idx=self._restored_reflection_idx(family) if restored else (i_R_s, j),
            wall_from=terminal_wall,
            inlet=self.inlet,
            next_family="cminus",
            tol=self.tol,
        )
        return self.net_L, self.net_R, reflected_event

    def _cleanup_debug_kwargs(self) -> dict[str, object]:
        return {
            "debug_plots": False,
            "debug_plotter": self._net_shock_plot_axes,
        }

    def _cminus_solver_kwargs(self) -> dict[str, object]:
        return {
            "max_iters": self.max_iters,
            "tol": self.tol,
        }

    def _cplus_solver_kwargs(self) -> dict[str, object]:
        return {
            "max_iters": self.max_iters,
            "tol": self.tol,
        }
