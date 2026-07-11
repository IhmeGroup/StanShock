from __future__ import annotations

import numpy as np

from inlet_moc.char_solvers import field_point, wall_point
from inlet_moc.flow_physics import RegionState
from inlet_moc.planar_inlet import PlanarInlet, get_opposite_wall


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
        region: RegionState,
        max_iters: int = 10,
        tol: float = 1e-6,
    ) -> None:
        self.N = N_idl
        self.inlet = inlet

        self.region = region
        self.a_fxn = self.region.ref.a_fxn
        self.max_iters = int(max_iters)
        self.tol = float(tol)

        self.x = np.full((self.N, self.N), np.nan)
        self.y = np.full_like(self.x, np.nan)
        self.u = np.full_like(self.x, np.nan)
        self.v = np.full_like(self.x, np.nan)
        self.active = np.zeros((self.N, self.N), dtype=bool)

        self.point_type = np.full_like(self.x, np.nan)  # 0 if field, 1 wall, 2 fluid boundary (shock or IDL)


        self.wall = None
        self.idl_kind = None
        self.prepared_j_stop: int | None = None

    def apply_idl(
        self, idl_pts: np.ndarray, u: float | np.ndarray, v: float | np.ndarray
    ) -> None:
        if idl_pts.shape[0] != self.N:
            msg = "IDL not consistent with initialized resolution!"
            raise ValueError(msg)

        if np.isscalar(u):
            u = np.full(self.N, u)
        if np.isscalar(v):
            v = np.full(self.N, v)

        dY_tot = idl_pts[-1, 1] - idl_pts[0, 1]
        dX_tot = idl_pts[-1, 0] - idl_pts[0, 0]

        if np.isclose(dX_tot, 0.0):
            diag = np.arange(self.N)
            self.x[diag, diag] = idl_pts[:, 0]
            self.y[diag, diag] = idl_pts[:, 1]
            self.u[diag, diag] = u
            self.v[diag, diag] = v
            self.active[diag, diag] = True
            self.idl_kind = "vertical"
            self.wall = None
            self.point_type[diag, diag] = 2

        else:
            m_idl = np.sign(dY_tot / dX_tot)

            if m_idl == +1:
                self.x[0, :] = idl_pts[:, 0]
                self.y[0, :] = idl_pts[:, 1]
                self.u[0, :] = u
                self.v[0, :] = v
                self.active[0, :] = True
                self.wall = self.inlet.centerbody
                self.idl_kind = "cplus"
                self.point_type[0, :] = 2
                self.point_type[0, 0] = 3
                self.point_type[0, self.N - 1] = 3

            elif m_idl == -1:
                self.x[:, 0] = idl_pts[:, 0]
                self.y[:, 0] = idl_pts[:, 1]
                self.u[:, 0] = u
                self.v[:, 0] = v
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



        
        




    def build_cminus_char(
        self,
        fixed_j: int,
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

        i_top = fixed_j
        if not self.has_point(i_top, fixed_j - 1):
            return False

        pt_top = wall_point(
            self.get_point(i_top, fixed_j - 1),
            wall_from,
            self.a_fxn,
            max_iters=self.max_iters,
            tol=tol_val,
        )
        pt_top_type = 3 if fixed_j == (self.N - 1) else 1
        self.edit_point(i_top, fixed_j, pt_top, pt_top_type)

        for i in range(fixed_j + 1, self.N):
            has_plus = self.has_point(i, fixed_j - 1)
            has_minus = self.has_point(i - 1, fixed_j)

            if has_plus and has_minus:
                pt_plus = self.get_point(i, fixed_j - 1)
                pt_minus = self.get_point(i - 1, fixed_j)
                pt_out = field_point(
                    pt_plus,
                    pt_minus,
                    self.a_fxn,
                    max_iters=self.max_iters,
                    tol=tol_val,
                )
                if _wall_clearance(pt_out, wall_to) <= tol_val:
                    pt_out = wall_point(
                        wall_to,
                        pt_plus,
                        self.a_fxn,
                        max_iters=self.max_iters,
                        tol=tol_val,
                    )
                    self.edit_point(i, fixed_j, pt_out, 2)
                    break
                self.edit_point(i, fixed_j, pt_out, 0)
                continue

            if has_plus and not has_minus:
                pt_wall = wall_point(
                    wall_to,
                    self.get_point(i, fixed_j - 1),
                    self.a_fxn,
                    max_iters=self.max_iters,
                    tol=tol_val,
                )
                self.edit_point(i, fixed_j, pt_wall, 2)
                break

            break

        return True

    def build_cplus_char(
        self,
        fixed_i: int,
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
        
        j_left = fixed_i
        if not self.has_point(fixed_i - 1, j_left):
            return False

        pt_left = wall_point(
            wall_from,
            self.get_point(fixed_i - 1, j_left),
            self.a_fxn,
            max_iters=self.max_iters,
            tol=tol_val,
        )
        pt_left_type = 3 if fixed_i == (self.N - 1) else 1
        self.edit_point(fixed_i, j_left, pt_left, pt_left_type)

        for j in range(fixed_i + 1, self.N):
            has_plus = self.has_point(fixed_i, j - 1)
            has_minus = self.has_point(fixed_i - 1, j)

            if has_plus and has_minus:
                pt_plus = self.get_point(fixed_i, j - 1)
                pt_minus = self.get_point(fixed_i - 1, j)
                pt_out = field_point(
                    pt_plus,
                    pt_minus,
                    self.a_fxn,
                    max_iters=self.max_iters,
                    tol=tol_val,
                )
                if _wall_clearance(pt_out, wall_to) <= tol_val:
                    pt_out = wall_point(
                        pt_plus,
                        wall_to,
                        self.a_fxn,
                        max_iters=self.max_iters,
                        tol=tol_val,
                    )
                    self.edit_point(fixed_i, j, pt_out, 2)
                    break
                self.edit_point(fixed_i, j, pt_out, 0)
                continue

            if has_plus and not has_minus:
                pt_wall = wall_point(
                    self.get_point(fixed_i, j - 1),
                    wall_to,
                    self.a_fxn,
                    max_iters=self.max_iters,
                    tol=tol_val,
                )
                self.edit_point(fixed_i, j, pt_wall, 2)
                break

            break

        return True

    def _solve_vertical_net(self):
        top_wall = self.inlet.cowl
        bottom_wall = self.inlet.centerbody

        for band in range(self.N - 1):
            front_size = self.N - band
            if front_size == 2:
                pt_upper = self.get_point(0, band)
                pt_lower = self.get_point(1, band + 1)
                pt_out = field_point(
                    pt_lower,
                    pt_upper,
                    self.a_fxn,
                    max_iters=self.max_iters,
                    tol=self.tol,
                )
                self.edit_point(0, band + 1, pt_out, 0)
                break

            if front_size < 3:
                break

            top_src = self.get_point(1, band + 1)
            top_pt = wall_point(
                top_src,
                top_wall,
                self.a_fxn,
                max_iters=self.max_iters,
                tol=self.tol,
            )
            self.edit_point(0, band + 1, top_pt, 1)

            for i in range(1, front_size - 2):
                pt_upper = self.get_point(i, i + band)
                pt_lower = self.get_point(i + 1, i + band + 1)
                pt_out = field_point(
                    pt_lower,
                    pt_upper,
                    self.a_fxn,
                    max_iters=self.max_iters,
                    tol=self.tol,
                )
                j = i + band + 1
                self.edit_point(i, j, pt_out, 0)

            bot_i = front_size - 2
            bot_src = self.get_point(bot_i, bot_i + band)
            bot_pt = wall_point(
                bottom_wall,
                bot_src,
                self.a_fxn,
                max_iters=self.max_iters,
                tol=self.tol,
            )
            self.edit_point(bot_i, self.N - 1, bot_pt, 1)

    def solve_net(self):
        if self.is_vertical:
            self._solve_vertical_net()
            return

        wall_from, wall_to = self._resolve_wall_pair()
        if self.is_Cplus:
            for i in range(1, self.N):
                if not self.build_cplus_char(i, wall_from=wall_from, wall_to=wall_to):
                    break
            return

        for j in range(1, self.N):
            if not self.build_cminus_char(j, wall_from=wall_from, wall_to=wall_to):
                break

    def xy_mask(self) -> np.ndarray:
        return self.active & np.isfinite(self.x) & np.isfinite(self.y)

    def state_mask(self) -> np.ndarray:
        return self.xy_mask() & np.isfinite(self.u) & np.isfinite(self.v)

    def point_mask(self, point_type: str) -> np.ndarray:
        """Return an xy-based mask for a named point-type grouping."""
        xy_mask = self.xy_mask()
        if point_type == "field":
            return xy_mask & (self.point_type == 0)
        if point_type == "wall":
            return xy_mask & (self.point_type == 1)
        if point_type == "boundary":
            return xy_mask & ((self.point_type == 2) | (self.point_type == 3))
        if point_type == "fluid":
            return xy_mask & (self.point_type == 2)
        if point_type == "corner":
            return xy_mask & (self.point_type == 3)
        return None

    def point_ij_xy(self, point_type: str) -> tuple[np.ndarray, np.ndarray]:
        """Return matching point indices and xy coordinates for a point type."""
        mask = self.point_mask(point_type)
        if mask is None:
            return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=float)

        rows, cols = np.nonzero(mask)
        ij = np.column_stack((rows, cols))
        xy = np.column_stack((self.x[rows, cols], self.y[rows, cols]))
        return ij, xy
    
    def get_active_points(self) -> np.ndarray:
        """Return active finite-state points as an ``(N, 4)`` [x, y, u, v] array."""
        mask = self.state_mask()
        return np.column_stack((self.x[mask], self.y[mask], self.u[mask], self.v[mask]))



    def has_point(self, i: int, j: int) -> bool:
        if (i < 0) or (j < 0) or (i >= self.N) or (j >= self.N):
            return False
        return bool(self.state_mask()[i, j])

    def deactivate_points(self, rows: np.ndarray, cols: np.ndarray) -> None:
        self.active[rows, cols] = False
        self.point_type[rows, cols] = np.nan


    def get_point(self, i: int, j: int):
        return np.array([self.x[i, j], self.y[i, j], self.u[i, j], self.v[i, j]])


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

    def edit_point(self, i: int, j: int, new_data: np.ndarray, point_type: int):
        data = new_data.copy()
        self.x[i, j] = data[0]
        self.y[i, j] = data[1]
        self.u[i, j] = data[2]
        self.v[i, j] = data[3]
        self.active[i, j] = bool(np.all(np.isfinite(data[:2])))
        self.point_type[i, j] = self._merge_point_type(self.point_type[i, j], point_type)
        return self
