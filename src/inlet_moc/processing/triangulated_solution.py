from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import Delaunay

if TYPE_CHECKING:
    from inlet_moc.moc_solution import MOCSolution


class TriangulatedSolution:
    def __init__(self, soln: MOCSolution) -> None:
        self.tol = soln.tol
        tris, primitives = build_tris(soln)
        self.points, self.triangles, self.primitives, self.region_id = merge_tris(
            tris,
            primitives,
        )
        self.edge_a = np.array([0, 1, 2])
        self.edge_b = np.array([1, 2, 0])

        self.process_tris()

    def process_tris(self) -> None:
        if self.triangles.shape[0] == 0:
            self.tri_xy = np.empty((0, 3, 2), dtype=float)
            self.tri_primitives = np.empty(
                (0, 3, self.primitives.shape[1]),
                dtype=float,
            )
            self.tri_xmin = np.empty(0, dtype=float)
            self.tri_xmax = np.empty(0, dtype=float)
            self.tri_ymin = np.empty(0, dtype=float)
            self.tri_ymax = np.empty(0, dtype=float)
            self.y_min_global = np.nan
            self.y_max_global = np.nan
            self.x_min_global = np.nan
            self.x_max_global = np.nan
            self.tri_coeffs = self.triangle_plane_coefficients()
            return

        self.tri_xy = self.points[self.triangles]  # (Nt, 3, 2)
        self.tri_primitives = self.primitives[self.triangles]  # (Nt, 3, Nq)

        self.tri_xmin = self.tri_xy[:, :, 0].min(axis=1)
        self.tri_xmax = self.tri_xy[:, :, 0].max(axis=1)
        self.tri_ymin = self.tri_xy[:, :, 1].min(axis=1)
        self.tri_ymax = self.tri_xy[:, :, 1].max(axis=1)

        self.y_min_global = float(np.min(self.points[:, 1]))
        self.y_max_global = float(np.max(self.points[:, 1]))
        self.x_min_global = float(np.min(self.points[:, 0]))
        self.x_max_global = float(np.max(self.points[:, 0]))

        self.tri_coeffs = self.triangle_plane_coefficients()

    def triangle_plane_coefficients(self):
        if self.triangles.shape[0] == 0:
            return np.empty((0, 3, self.primitives.shape[1]), dtype=float)

        x0 = self.tri_xy[:, 0, 0]
        y0 = self.tri_xy[:, 0, 1]

        dx1 = self.tri_xy[:, 1, 0] - x0
        dy1 = self.tri_xy[:, 1, 1] - y0
        dx2 = self.tri_xy[:, 2, 0] - x0
        dy2 = self.tri_xy[:, 2, 1] - y0

        dq1 = self.tri_primitives[:, 1] - self.tri_primitives[:, 0]
        dq2 = self.tri_primitives[:, 2] - self.tri_primitives[:, 0]

        det = dx1 * dy2 - dx2 * dy1
        det_scale = np.abs(dx1 * dy2) + np.abs(dx2 * dy1)
        valid = np.abs(det) > np.finfo(float).eps * np.maximum(det_scale, 1.0)

        a = np.divide(
            dq1 * dy2[:, None] - dq2 * dy1[:, None],
            det[:, None],
            out=np.full_like(dq1, np.nan),
            where=valid[:, None],
        )
        b = np.divide(
            dx1[:, None] * dq2 - dx2[:, None] * dq1,
            det[:, None],
            out=np.full_like(dq1, np.nan),
            where=valid[:, None],
        )

        c = self.tri_primitives[:, 0] - a * x0[:, None] - b * y0[:, None]

        return np.stack((a, b, c), axis=1)

    def get_primitives(self, x: float, y1: float, y2: float):
        y_lo = min(y1, y2)
        y_hi = max(y1, y2)
        n_primitives = self.primitives.shape[1]

        mask = (
            (self.tri_xmin <= x + self.tol)
            & (self.tri_xmax >= x - self.tol)
            & (self.tri_ymin <= y_hi + self.tol)
            & (self.tri_ymax >= y_lo - self.tol)
        )

        tri_inds = np.flatnonzero(mask)
        if tri_inds.size == 0:
            return (
                x,
                np.empty((0, 2)),
                np.empty((0, 2, n_primitives)),
                tri_inds,
            )

        xy = self.tri_xy[tri_inds]

        xa = xy[:, self.edge_a, 0]  # (Nc, 3)
        xb = xy[:, self.edge_b, 0]
        ya = xy[:, self.edge_a, 1]
        yb = xy[:, self.edge_b, 1]

        dx = xb - xa
        non_vert = np.abs(dx) > self.tol

        t = np.divide(
            x - xa,
            dx,
            out=np.full_like(dx, np.nan),
            where=non_vert,
        )

        hit = non_vert & (t >= -self.tol) & (t <= 1.0 + self.tol)
        t = np.clip(t, 0.0, 1.0)

        y_hit = ya + t * (yb - ya)
        y_hit = np.where(hit, y_hit, np.inf)

        edge_order = np.argsort(y_hit, axis=1)
        selected_edges = edge_order[:, :2]
        y_end = np.take_along_axis(y_hit, selected_edges, axis=1)

        good = np.isfinite(y_end).all(axis=1)

        tri_inds = tri_inds[good]
        y_end = y_end[good]

        if tri_inds.size == 0:
            return (
                x,
                np.empty((0, 2)),
                np.empty((0, 2, n_primitives)),
                tri_inds,
            )

        y_end[:, 0] = np.maximum(y_end[:, 0], y_lo)
        y_end[:, 1] = np.minimum(y_end[:, 1], y_hi)

        keep = y_end[:, 1] > y_end[:, 0] + self.tol

        tri_inds = tri_inds[keep]
        y_end = y_end[keep]

        if tri_inds.size == 0:
            return (
                x,
                np.empty((0, 2)),
                np.empty((0, 2, n_primitives)),
                tri_inds,
            )

        y_breaks = np.sort(y_end.reshape(-1))
        y_merged = [y_breaks[0]]
        for y_val in y_breaks[1:]:
            if y_val > y_merged[-1] + self.tol:
                y_merged.append(y_val)
            elif y_val > y_merged[-1]:
                y_merged[-1] = y_val

        y_breaks = np.asarray(y_merged)
        y0 = y_breaks[:-1]
        y1 = y_breaks[1:]
        span_keep = y1 > y0 + self.tol
        y0 = y0[span_keep]
        y1 = y1[span_keep]

        owner_tri_inds = []
        y_nonoverlap = []
        for y_lower, y_upper in zip(y0, y1, strict=False):
            y_mid = 0.5 * (y_lower + y_upper)
            cover = (y_end[:, 0] <= y_mid + self.tol) & (
                y_end[:, 1] >= y_mid - self.tol
            )
            if not np.any(cover):
                continue

            cover_idx = np.flatnonzero(cover)
            cover_region_id = self.region_id[tri_inds[cover_idx]]
            owner_idx = cover_idx[np.argmax(cover_region_id)]
            owner_tri_inds.append(tri_inds[owner_idx])
            y_nonoverlap.append((y_lower, y_upper))

        if not owner_tri_inds:
            return (
                x,
                np.empty((0, 2)),
                np.empty((0, 2, n_primitives)),
                np.empty(0, dtype=int),
            )

        tri_inds = np.asarray(owner_tri_inds, dtype=int)
        y_end = np.asarray(y_nonoverlap)
        q_end = (
            self.tri_coeffs[tri_inds, 0, None, :] * x
            + self.tri_coeffs[tri_inds, 1, None, :] * y_end[:, :, None]
            + self.tri_coeffs[tri_inds, 2, None, :]
        )

        order = np.argsort(y_end[:, 0], kind="stable")
        return (
            x,
            y_end[order],  # (Ns, 2)
            q_end[order],  # (Ns, 2, 5)
            tri_inds[order],  # (Ns,)
        )


def build_tris(soln: MOCSolution) -> tuple[list[Delaunay], list[np.ndarray]]:
    tris: list[Delaunay] = []
    primitives: list[np.ndarray] = []

    for cell in soln.cells:
        pts = cell.state_points()
        if pts.shape[0] < 3:
            continue

        tri = Delaunay(pts[:, :2])
        V = pts[:, 2]
        theta = pts[:, 3]
        p = pts[:, 4]
        rho = pts[:, 5]
        u = V * np.cos(theta)
        v = V * np.sin(theta)
        a = np.sqrt(soln.gamma * p / rho)

        tris.append(tri)
        # Stored states are [x, y, V, theta, p, rho]; derived primitives are
        # [rho, u, v, p, a] for plotting and stream-thrust integration.
        primitives.append(np.column_stack((rho, u, v, p, a)))

    for net in soln.nets:
        pts = net.get_active_points()
        if pts.shape[0] < 3:
            continue

        tri = Delaunay(pts[:, :2])
        V = pts[:, 2]
        theta = pts[:, 3]
        p = pts[:, 4]
        rho = pts[:, 5]
        u = V * np.cos(theta)
        v = V * np.sin(theta)
        a = np.sqrt(soln.gamma * p / rho)

        tris.append(tri)
        # Stored states are [x, y, V, theta, p, rho]; derived primitives are
        # [rho, u, v, p, a] for plotting and stream-thrust integration.
        primitives.append(np.column_stack((rho, u, v, p, a)))

    return tris, primitives


def merge_tris(
    tris: list[Delaunay],
    primitives_in: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(tris) != len(primitives_in):
        msg = "tris and primitives must have the same length."
        raise ValueError(msg)
    if not tris:
        return (
            np.empty((0, 2), dtype=float),
            np.empty((0, 3), dtype=int),
            np.empty((0, 5), dtype=float),
            np.empty(0, dtype=int),
        )

    point_blocks: list[np.ndarray] = []
    triangle_blocks: list[np.ndarray] = []
    primitive_blocks: list[np.ndarray] = []
    region_blocks: list[np.ndarray] = []

    offset = 0
    for region, (tri, prim) in enumerate(zip(tris, primitives_in, strict=False)):
        points = np.asarray(tri.points, dtype=float)
        simplices = np.asarray(tri.simplices, dtype=int)
        if prim.shape[0] != points.shape[0]:
            msg = "Each primitive block must have one row per triangulation point."
            raise ValueError(msg)
        point_blocks.append(points)
        triangle_blocks.append(simplices + offset)
        primitive_blocks.append(prim)
        region_blocks.append(np.full(simplices.shape[0], region, dtype=int))

        offset += points.shape[0]

    points = np.concatenate(point_blocks, axis=0)
    triangles = np.concatenate(triangle_blocks, axis=0)
    primitives = np.concatenate(primitive_blocks, axis=0)
    region_id = np.concatenate(region_blocks, axis=0)

    return points, triangles, primitives, region_id
