from __future__ import annotations

import math
from typing import Self

import numpy as np

from inlet_moc.planar_inlet import PlanarInlet, get_opposite_wall
from inlet_moc.rotational_solvers import (
    NoWallIntersectionError,
    RotationalSolveError,
    field_point_rot,
    wall_point_rot,
)


def _wall_clearance(pt: np.ndarray, wall) -> float:
    wall_y = wall.get_y(float(pt[0]))
    if wall_y is None:
        return np.inf
    return float((pt[1] - wall_y) * wall.normal_sign)


class CharNet:
    def __init__(
        self,
        N_idl: int,
        inlet: PlanarInlet,
        gamma: float,
        max_iters: int = 10,
        tol: float = 1e-6,
    ) -> None:
        self.N = N_idl
        self.inlet = inlet
        self.gamma = float(gamma)
        self.max_iters = int(max_iters)
        self.tol = float(tol)

        self.x = np.full((self.N, self.N), np.nan)
        self.y = np.full_like(self.x, np.nan)
        self.V = np.full_like(self.x, np.nan)
        self.theta = np.full_like(self.x, np.nan)
        self.p = np.full_like(self.x, np.nan)
        self.rho = np.full_like(self.x, np.nan)

        self.active = np.zeros((self.N, self.N), dtype=bool)
        self.point_type = np.full_like(self.x, np.nan)

        self.wall = None
        self.idl_kind = None
        self.prepared_j_stop: int | None = None

    def apply_idl(
        self,
        idl_pts: np.ndarray,
        V: float | np.ndarray,
        theta: float | np.ndarray,
        p: float | np.ndarray,
        rho: float | np.ndarray,
    ) -> None:
        if idl_pts.shape[0] != self.N:
            msg = "IDL not consistent with initialized resolution!"
            raise ValueError(msg)

        if np.isscalar(V):
            V = np.full(self.N, V)
        if np.isscalar(theta):
            theta = np.full(self.N, theta)
        if np.isscalar(p):
            p = np.full(self.N, p)
        if np.isscalar(rho):
            rho = np.full(self.N, rho)

        dY_tot = idl_pts[-1, 1] - idl_pts[0, 1]
        dX_tot = idl_pts[-1, 0] - idl_pts[0, 0]

        if np.isclose(dX_tot, 0.0):
            diag = np.arange(self.N)
            self.x[diag, diag] = idl_pts[:, 0]
            self.y[diag, diag] = idl_pts[:, 1]
            self.V[diag, diag] = V
            self.theta[diag, diag] = theta
            self.p[diag, diag] = p
            self.rho[diag, diag] = rho
            self.active[diag, diag] = True
            self.idl_kind = "vertical"
            self.wall = None
            self.point_type[diag, diag] = 2

        else:
            m_idl = np.sign(dY_tot / dX_tot)

            if m_idl == +1:
                self.x[0, :] = idl_pts[:, 0]
                self.y[0, :] = idl_pts[:, 1]
                self.V[0, :] = V
                self.theta[0, :] = theta
                self.p[0, :] = p
                self.rho[0, :] = rho
                self.active[0, :] = True
                self.wall = self.inlet.centerbody
                self.idl_kind = "cplus"
                self.point_type[0, :] = 2
                self.point_type[0, 0] = 3
                self.point_type[0, self.N - 1] = 3

            elif m_idl == -1:
                self.x[:, 0] = idl_pts[:, 0]
                self.y[:, 0] = idl_pts[:, 1]
                self.V[:, 0] = V
                self.theta[:, 0] = theta
                self.p[:, 0] = p
                self.rho[:, 0] = rho
                self.active[:, 0] = True
                self.wall = self.inlet.cowl
                self.idl_kind = "cminus"
                self.point_type[:, 0] = 2
                self.point_type[0, 0] = 3
                self.point_type[self.N - 1, 0] = 3

            else:
                msg = "IDL cannot have 0 slope."
                raise ValueError(msg)

    @property
    def is_vertical(self):
        return self.idl_kind == "vertical"

    @property
    def is_Cplus(self):
        return self.idl_kind == "cplus"

    def _resolve_wall_pair(self):
        if self.is_vertical:
            msg = "Vertical-IDL nets do not use a single marching wall pair."
            raise ValueError(msg)
        if self.wall is None:
            msg = "Characteristic net has no assigned marching wall."
            raise ValueError(msg)
        xy_wall = None
        if (self.idl_kind == "cplus" and self.has_point(0, 0)) or (
            self.idl_kind == "cminus" and self.has_point(0, 0)
        ):
            xy_wall = self.get_point(0, 0)[:2]
        if xy_wall is None:
            msg = "Could not determine the current wall anchor point for this net."
            raise RuntimeError(msg)
        return self.wall, get_opposite_wall(xy_wall, self.inlet)

    def _wall_point_from_cminus_char(
        self, i: int, j: int, wall, tol: float
    ) -> np.ndarray:
        return wall_point_rot(
            self.get_point(i - 1, j),
            self.get_point(i - 1, j - 1),
            wall,
            "cminus",
            self.gamma,
            max_iters=self.max_iters,
            tol=tol,
        )

    def _wall_point_from_cplus_char(
        self, i: int, j: int, wall, tol: float
    ) -> np.ndarray:
        return wall_point_rot(
            self.get_point(i, j - 1),
            self.get_point(i - 1, j - 1),
            wall,
            "cplus",
            self.gamma,
            max_iters=self.max_iters,
            tol=tol,
        )

    def _wall_point_from_cminus(self, i: int, j: int, wall, tol: float) -> np.ndarray:
        return wall_point_rot(
            self.get_point(i - 1, j),
            self.get_point(i, j - 1),
            wall,
            "cminus",
            self.gamma,
            max_iters=self.max_iters,
            tol=tol,
        )

    def build_cminus_char(
        self,
        fixed_i: int,
        wall_from=None,
        wall_to=None,
        tol: float | None = None,
    ) -> bool:
        tol_val = self.tol if tol is None else float(tol)
        if self.is_vertical:
            msg = (
                "Vertical-IDL nets do not support ordinary C- characteristic marching."
            )
            raise ValueError(msg)

        j_left = fixed_i
        if not (
            self.has_point(fixed_i - 1, j_left)
            and self.has_point(fixed_i - 1, j_left - 1)
        ):
            return False

        pt_left = self._wall_point_from_cminus_char(fixed_i, j_left, wall_from, tol_val)
        pt_left_type = 3 if fixed_i == (self.N - 1) else 1
        self.edit_point(fixed_i, j_left, pt_left, pt_left_type)

        for j in range(fixed_i + 1, self.N):
            has_plus = self.has_point(fixed_i, j - 1)
            has_minus = self.has_point(fixed_i - 1, j)

            if has_plus and has_minus:
                pt_plus = self.get_point(fixed_i, j - 1)
                pt_minus = self.get_point(fixed_i - 1, j)
                pt_out = field_point_rot(
                    pt_plus,
                    pt_minus,
                    self.gamma,
                    max_iters=self.max_iters,
                    tol=tol_val,
                )
                if _wall_clearance(pt_out, wall_to) <= tol_val:
                    break
                self.edit_point(fixed_i, j, pt_out, 0)
                continue

            if has_plus and not has_minus:
                break

            break

        return True

    def build_cplus_char(
        self,
        fixed_j: int,
        wall_from=None,
        wall_to=None,
        tol: float | None = None,
    ) -> bool:
        tol_val = self.tol if tol is None else float(tol)
        if self.is_vertical:
            msg = (
                "Vertical-IDL nets do not support ordinary C+ characteristic marching."
            )
            raise ValueError(msg)

        i_top = fixed_j
        if not (
            self.has_point(i_top, fixed_j - 1)
            and self.has_point(i_top - 1, fixed_j - 1)
        ):
            return False

        pt_top = self._wall_point_from_cplus_char(i_top, fixed_j, wall_from, tol_val)
        pt_top_type = 3 if fixed_j == (self.N - 1) else 1
        self.edit_point(i_top, fixed_j, pt_top, pt_top_type)

        for i in range(fixed_j + 1, self.N):
            has_plus = self.has_point(i, fixed_j - 1)
            has_minus = self.has_point(i - 1, fixed_j)

            if has_plus and has_minus:
                pt_plus = self.get_point(i, fixed_j - 1)
                pt_minus = self.get_point(i - 1, fixed_j)
                pt_out = field_point_rot(
                    pt_plus,
                    pt_minus,
                    self.gamma,
                    max_iters=self.max_iters,
                    tol=tol_val,
                )
                if _wall_clearance(pt_out, wall_to) <= tol_val:
                    try:
                        pt_out = self._wall_point_from_cminus(
                            i, fixed_j, wall_to, tol_val
                        )
                    except (NoWallIntersectionError, RotationalSolveError):
                        pt_out = None
                    if pt_out is not None:
                        self.edit_point(i, fixed_j, pt_out, 2)
                    break
                self.edit_point(i, fixed_j, pt_out, 0)
                continue

            if has_plus and not has_minus:
                break

            break

        return True

    def _solve_vertical_net(self) -> None:
        top_wall = self.inlet.cowl
        bottom_wall = self.inlet.centerbody

        for band in range(self.N - 1):
            front_size = self.N - band
            if front_size == 2:
                pt_upper = self.get_point(0, band)
                pt_lower = self.get_point(1, band + 1)
                pt_out = field_point_rot(
                    pt_lower,
                    pt_upper,
                    self.gamma,
                    max_iters=self.max_iters,
                    tol=self.tol,
                )
                self.edit_point(0, band + 1, pt_out, 0)
                break

            if front_size < 3:
                break

            top_src = self.get_point(1, band + 1)
            top_pt = wall_point_rot(
                top_src,
                top_wall,
                self.gamma,
                max_iters=self.max_iters,
                tol=self.tol,
            )
            self.edit_point(0, band + 1, top_pt, 1)

            for i in range(1, front_size - 2):
                pt_upper = self.get_point(i, i + band)
                pt_lower = self.get_point(i + 1, i + band + 1)
                pt_out = field_point_rot(
                    pt_lower,
                    pt_upper,
                    self.gamma,
                    max_iters=self.max_iters,
                    tol=self.tol,
                )
                j = i + band + 1
                self.edit_point(i, j, pt_out, 0)

            bot_i = front_size - 2
            bot_src = self.get_point(bot_i, bot_i + band)
            bot_pt = wall_point_rot(
                bottom_wall,
                bot_src,
                self.gamma,
                max_iters=self.max_iters,
                tol=self.tol,
            )
            self.edit_point(bot_i, self.N - 1, bot_pt, 1)

    def solve_net(self) -> None:
        if self.is_vertical:
            self._solve_vertical_net()
            return

        wall_from, wall_to = self._resolve_wall_pair()
        if self.is_Cplus:
            for i in range(1, self.N):
                if not self.build_cminus_char(i, wall_from=wall_from, wall_to=wall_to):
                    break
            return

        for j in range(1, self.N):
            if not self.build_cplus_char(j, wall_from=wall_from, wall_to=wall_to):
                break

    def xy_mask(self) -> np.ndarray:
        return self.active & np.isfinite(self.x) & np.isfinite(self.y)

    def state_mask(self) -> np.ndarray:
        return (
            self.xy_mask()
            & np.isfinite(self.V)
            & np.isfinite(self.theta)
            & np.isfinite(self.p)
            & np.isfinite(self.rho)
        )

    def point_mask(self, point_type: str) -> np.ndarray:
        xy_mask = self.xy_mask()
        match point_type:
            case "field":
                return xy_mask & (self.point_type == 0)
            case "wall":
                return xy_mask & (self.point_type == 1)
            case "boundary":
                return xy_mask & ((self.point_type == 2) | (self.point_type == 3))
            case "fluid":
                return xy_mask & (self.point_type == 2)
            case "corner":
                return xy_mask & (self.point_type == 3)
            case _:
                msg = f"Unsupported point type: {point_type}"
                raise ValueError(msg)

    def point_ij_xy(self, point_type: str) -> tuple[np.ndarray, np.ndarray]:
        mask = self.point_mask(point_type)
        rows, cols = np.nonzero(mask)
        ij = np.column_stack((rows, cols))
        xy = np.column_stack((self.x[rows, cols], self.y[rows, cols]))
        return ij, xy

    def get_active_points(self) -> np.ndarray:
        mask = self.state_mask()
        return np.column_stack(
            (
                self.x[mask],
                self.y[mask],
                self.V[mask],
                self.theta[mask],
                self.p[mask],
                self.rho[mask],
            )
        )

    def has_point(self, i: int, j: int) -> bool:
        if (i < 0) or (j < 0) or (i >= self.N) or (j >= self.N):
            return False
        return bool(
            self.active[i, j]
            and math.isfinite(self.x[i, j])
            and math.isfinite(self.y[i, j])
            and math.isfinite(self.V[i, j])
            and math.isfinite(self.theta[i, j])
            and math.isfinite(self.p[i, j])
            and math.isfinite(self.rho[i, j])
        )

    def deactivate_points(self, rows: np.ndarray, cols: np.ndarray) -> None:
        self.active[rows, cols] = False
        self.point_type[rows, cols] = np.nan

    def get_point(self, i: int, j: int) -> np.ndarray:
        return np.array(
            [
                self.x[i, j],
                self.y[i, j],
                self.V[i, j],
                self.theta[i, j],
                self.p[i, j],
                self.rho[i, j],
            ]
        )

    @staticmethod
    def _merge_point_type(existing_type: float, new_type: int) -> int:
        if np.isnan(existing_type):
            return int(new_type)

        existing = int(existing_type)
        new = int(new_type)

        if new == 0:
            return existing
        if existing == 0:
            return new
        return existing | new

    def edit_point(
        self,
        i: int,
        j: int,
        new_data: np.ndarray,
        point_type: int,
    ) -> Self:
        data = new_data.copy()
        self.x[i, j] = data[0]
        self.y[i, j] = data[1]
        self.V[i, j] = data[2]
        self.theta[i, j] = data[3]
        self.p[i, j] = data[4]
        self.rho[i, j] = data[5]
        self.active[i, j] = bool(np.all(np.isfinite(data[:2])))
        self.point_type[i, j] = self._merge_point_type(
            self.point_type[i, j], point_type
        )
        return self
