from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import numpy as np
from matplotlib.path import Path

from inlet_moc.charnet import CharNet
from inlet_moc.planar_inlet import PiecewiseLinearCurve
from inlet_moc.plot_net_shock import _plot_cleanup_candidates, _plot_coalescing_cleanup
from inlet_moc.utils_moc import (
    get_intersection_vectorized,
    get_tang_from_pts_vectorized,
)


def _segment_parameter_array(
    xy: np.ndarray,
    pt0: np.ndarray,
    pt1: np.ndarray,
    tol: float,
) -> np.ndarray:
    pt0_xy = np.asarray(pt0, dtype=float)[..., :2]
    pt1_xy = np.asarray(pt1, dtype=float)[..., :2]
    xy = np.asarray(xy, dtype=float)[..., :2]

    delta = pt1_xy - pt0_xy
    seg_len2 = np.sum(delta * delta, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        numer = np.sum((xy - pt0_xy) * delta, axis=-1)
        t = numer / seg_len2
    return np.where(seg_len2 > tol, t, np.nan)


def _char_index_arrays(
    net: CharNet,
    fixed_idx: int,
    free_inds: np.ndarray,
    family: str = "cminus",
) -> tuple[np.ndarray, np.ndarray]:
    """Map a characteristic's fixed/free indices to valid row/col arrays."""
    free_inds = np.asarray(free_inds, dtype=int).reshape(-1)
    if free_inds.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    free_inds = np.unique(free_inds)
    if family == "cminus":
        rows = free_inds
        cols = np.full_like(rows, fixed_idx)
    else:
        rows = np.full_like(free_inds, fixed_idx)
        cols = free_inds

    in_bounds = (rows >= 0) & (rows < net.N) & (cols >= 0) & (cols < net.N)
    rows = rows[in_bounds]
    cols = cols[in_bounds]
    if rows.size == 0:
        return rows, cols

    finite = net.xy_mask()[rows, cols]
    return rows[finite], cols[finite]


def get_char_interp_segment(
    net: CharNet,
    idx_fixed: int,
    pt_idx: int,
    family: str = "cplus",
    stencil_length: int = 10,
    mode: str = "upstream",
):
    """
    Return an interpolation stencil from one characteristic row/col.

    ``idx_fixed`` is the fixed row/column index and ``pt_idx`` sets the
    starting point of the returned stencil.

    In ``mode="upstream"``, returned points are ordered monotonically along
    the characteristic, include ``pt_idx``, and extend upstream from there.

    In ``mode="downstream"``, returned points are ordered monotonically along
    the characteristic, begin at the first active point with free index
    ``>= pt_idx``, and extend downstream from there.
    """
    free_inds = _active_free_indices(net, int(idx_fixed), family)
    if free_inds is None or free_inds.size == 0:
        empty_pts = np.empty((0, 4), dtype=float)
        return empty_pts, np.empty((0,), dtype=int)

    free_inds = np.asarray(free_inds, dtype=int)
    pt_idx = int(pt_idx)
    count = max(1, int(stencil_length))

    if mode == "upstream":
        end_pos = np.searchsorted(free_inds, pt_idx)
        if end_pos >= free_inds.size or free_inds[end_pos] != pt_idx:
            empty_pts = np.empty((0, 4), dtype=float)
            return empty_pts, np.empty((0,), dtype=int)
        start_pos = max(0, end_pos - count + 1)
        free_tail = np.asarray(free_inds[start_pos : end_pos + 1], dtype=int)
    else:
        start_pos = np.searchsorted(free_inds, pt_idx, side="left")
        if start_pos >= free_inds.size:
            empty_pts = np.empty((0, 4), dtype=float)
            return empty_pts, np.empty((0,), dtype=int)
        free_tail = np.asarray(free_inds[start_pos : start_pos + count], dtype=int)

    pts = np.asarray(
        [
            _get_family_point(net, int(idx_fixed), int(free_idx), family)
            for free_idx in free_tail
        ],
        dtype=float,
    )
    return pts, free_tail


def clip_char(
    net: CharNet,
    fixed_idx: int,
    free_inds: np.ndarray,
    family: str = "cminus",
) -> int:
    """
    Vectorized clip of a same-family characteristic using one fixed index and a
    1D array of varying indices.
    """
    rows, cols = _char_index_arrays(net, fixed_idx, free_inds, family=family)
    if rows.size == 0:
        return 0
    net.deactivate_points(rows, cols)
    return int(rows.size)


def remove_spurious_preshock_pts(
    net: CharNet,
    family: str,
    xyuv_shock: np.ndarray,
    tol: float = 1.0e-10,
) -> int:
    """
    Remove restored-net points on the pre-shock side of a shock polyline.

    The test is strictly geometric. Each active net point is projected onto
    the nearest segment of ``xyuv_shock[:, :2]``. For a restored C+ family,
    points with ``dx < 0`` and ``dy > 0`` relative to that nearest shock point
    are removed. For C-, points with ``dx < 0`` and ``dy < 0`` are removed.
    """
    family = str(family).lower()
    if family not in {"cplus", "cminus"}:
        msg = "family must be either 'cplus' or 'cminus'."
        raise ValueError(msg)

    xyuv_shock = np.asarray(xyuv_shock, dtype=float)
    if xyuv_shock.ndim != 2 or xyuv_shock.shape[1] < 2:
        return 0
    if xyuv_shock.shape[0] < 2:
        return 0

    shock_xy = np.asarray(xyuv_shock[:, :2], dtype=float)
    finite_shock = np.all(np.isfinite(shock_xy), axis=1)
    shock_xy = shock_xy[finite_shock]
    if shock_xy.shape[0] < 2:
        return 0
    shock_ds = np.hypot(
        np.diff(shock_xy[:, 0]),
        np.diff(shock_xy[:, 1]),
    )
    keep_shock = np.concatenate(([True], shock_ds > tol))
    shock_xy = shock_xy[keep_shock]
    if shock_xy.shape[0] < 2:
        return 0

    rows, cols = np.nonzero(net.xy_mask())
    if rows.size == 0:
        return 0

    pts_xy = np.column_stack((net.x[rows, cols], net.y[rows, cols]))
    finite_pts = np.all(np.isfinite(pts_xy), axis=1)
    if not np.any(finite_pts):
        return 0

    rows = rows[finite_pts]
    cols = cols[finite_pts]
    pts_xy = pts_xy[finite_pts]

    shock_seg0 = shock_xy[:-1]
    shock_seg1 = shock_xy[1:]
    shock_vec = shock_seg1 - shock_seg0
    shock_len2 = np.sum(shock_vec * shock_vec, axis=1)
    valid_seg = shock_len2 > (tol * tol)
    if not np.any(valid_seg):
        return 0

    shock_seg0 = shock_seg0[valid_seg]
    shock_vec = shock_vec[valid_seg]
    shock_len2 = shock_len2[valid_seg]

    rel = pts_xy[:, None, :] - shock_seg0[None, :, :]
    t_proj = np.sum(rel * shock_vec[None, :, :], axis=2) / shock_len2[None, :]
    t_proj = np.clip(t_proj, 0.0, 1.0)
    xy_proj = shock_seg0[None, :, :] + t_proj[:, :, None] * shock_vec[None, :, :]
    dxy = pts_xy[:, None, :] - xy_proj
    dist2 = np.sum(dxy * dxy, axis=2)
    nearest_seg = np.argmin(dist2, axis=1)
    nearest_xy = xy_proj[np.arange(pts_xy.shape[0]), nearest_seg]

    dx = nearest_xy[:, 0] - pts_xy[:, 0]
    dy = pts_xy[:, 1] - nearest_xy[:, 1]
    tol_xy = max(float(tol), 1.0e-12)

    if family == "cplus":
        remove = (dx > tol_xy) & (dy < -tol_xy)
    else:
        remove = (dx > tol_xy) & (dy > tol_xy)

    if not np.any(remove):
        return 0

    net.deactivate_points(rows[remove], cols[remove])
    return int(np.count_nonzero(remove))


def restore_resolution(
    net: CharNet,
    xyuv_shock: np.ndarray,
    ij_shock_R: np.ndarray,
    N_res: int,
) -> tuple[np.ndarray, np.ndarray, CharNet, np.ndarray]:
    """
    Resample ordered shock points and seed a fresh downstream net.

    ``xyuv_shock`` contains post-shock states ``[x, y, u, v]``. ``ij_shock_R``
    identifies whether those states live on a fixed downstream ``i`` or ``j``
    boundary. The restored net is seeded by direct point edits; no IDL machinery
    is used.
    """
    N_res = int(N_res)
    if N_res < 2:
        msg = "N_res must be at least 2."
        raise ValueError(msg)

    xyuv_q, ij_q, _, _ = resample_shock(
        xyuv_shock,
        ij_shock_R,
        N_res,
    )
    net_R = _make_restored_net(net, xyuv_q, ij_q)
    iter_idx = np.arange(1, N_res, dtype=int)
    return xyuv_q, ij_q, net_R, iter_idx


def resample_shock(
    xyuv_shock: np.ndarray,
    ij_shock_R: np.ndarray,
    N_res: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    xyuv_shock = np.asarray(xyuv_shock, dtype=float)
    if xyuv_shock.ndim != 2 or xyuv_shock.shape[1] < 4:
        msg = "xyuv_shock must be a 2D array with at least four columns: [x, y, u, v]."
        raise ValueError(msg)
    if xyuv_shock.shape[0] == 0:
        msg = "xyuv_shock must contain at least one point."
        raise ValueError(msg)

    ij_shock_R = np.asarray(ij_shock_R, dtype=int)
    if ij_shock_R.ndim != 2 or ij_shock_R.shape[1] != 2:
        msg = "ij_shock_R must be a 2D integer array with columns [i, j]."
        raise ValueError(msg)
    if ij_shock_R.shape[0] != xyuv_shock.shape[0]:
        msg = "ij_shock_R must have one row for each shock point."
        raise ValueError(msg)

    xyuv_shock = xyuv_shock[:, :4]
    finite_mask = np.all(np.isfinite(xyuv_shock), axis=1)
    finite_mask &= np.all(ij_shock_R >= 0, axis=1)
    xyuv_shock = xyuv_shock[finite_mask]
    ij_shock_R = ij_shock_R[finite_mask]
    if xyuv_shock.shape[0] == 0:
        msg = "xyuv_shock has no finite points after filtering."
        raise ValueError(msg)

    fixed_axis, changing_axis = _shock_index_axes(ij_shock_R)
    order = np.argsort(ij_shock_R[:, changing_axis])
    xyuv_shock = xyuv_shock[order]
    ij_shock_R = ij_shock_R[order]

    _, unique_idx = np.unique(ij_shock_R[:, changing_axis], return_index=True)
    unique_idx = np.sort(unique_idx)
    xyuv_shock = xyuv_shock[unique_idx]
    ij_shock_R = ij_shock_R[unique_idx]

    fixed_idx = int(ij_shock_R[0, fixed_axis])
    s_pts = [0.0]
    xyuv_pts = [xyuv_shock[0]]

    s_total = 0.0
    for idx in range(1, xyuv_shock.shape[0]):
        pt_prev = xyuv_pts[-1]
        pt_next = xyuv_shock[idx]
        ds = float(np.hypot(pt_next[0] - pt_prev[0], pt_next[1] - pt_prev[1]))
        if ds <= 0.0:
            continue
        s_total += ds
        s_pts.append(s_total)
        xyuv_pts.append(pt_next)

    if s_total <= 0.0:
        xyuv_q = np.repeat(xyuv_shock[:1], int(N_res), axis=0)
        ij_q = _resampled_shock_indices(
            int(N_res), fixed_axis, changing_axis, fixed_idx
        )
        return xyuv_q, ij_q, fixed_axis, changing_axis

    r_pts = [s_val / s_total for s_val in s_pts]
    r_q = np.linspace(0.0, 1.0, int(N_res))
    xyuv_q = np.empty((int(N_res), 4), dtype=float)

    seg_idx = 0
    for q_idx, r_val in enumerate(r_q):
        if r_val <= 0.0:
            xyuv_q[q_idx] = xyuv_pts[0]
            continue
        if r_val >= 1.0:
            xyuv_q[q_idx] = xyuv_pts[-1]
            continue

        while seg_idx < (len(r_pts) - 2) and r_val > r_pts[seg_idx + 1]:
            seg_idx += 1

        r0 = r_pts[seg_idx]
        r1 = r_pts[seg_idx + 1]
        frac = (r_val - r0) / (r1 - r0)
        xyuv_q[q_idx] = xyuv_pts[seg_idx] + frac * (
            xyuv_pts[seg_idx + 1] - xyuv_pts[seg_idx]
        )

    ij_q = _resampled_shock_indices(int(N_res), fixed_axis, changing_axis, fixed_idx)
    return xyuv_q, ij_q, fixed_axis, changing_axis


def _shock_index_axes(ij_shock_R: np.ndarray) -> tuple[int, int]:
    i_is_fixed = np.all(ij_shock_R[:, 0] == ij_shock_R[0, 0])
    j_is_fixed = np.all(ij_shock_R[:, 1] == ij_shock_R[0, 1])

    if i_is_fixed and not j_is_fixed:
        return 0, 1
    if j_is_fixed and not i_is_fixed:
        return 1, 0
    if i_is_fixed and j_is_fixed:
        msg = "Shock index record cannot be resampled because both indices are fixed."
        raise ValueError(msg)

    i_span = int(np.max(ij_shock_R[:, 0]) - np.min(ij_shock_R[:, 0]))
    j_span = int(np.max(ij_shock_R[:, 1]) - np.min(ij_shock_R[:, 1]))
    if i_span == 0 or j_span == 0:
        fixed_axis = 0 if i_span == 0 else 1
        return fixed_axis, 1 - fixed_axis

    msg = "Shock index record must have one fixed index and one changing index."
    raise ValueError(msg)


def _resampled_shock_indices(
    N_res: int,
    fixed_axis: int,
    changing_axis: int,
    fixed_idx: int,
) -> np.ndarray:
    if fixed_idx < 0 or fixed_idx >= N_res:
        msg = (
            "Fixed shock index is outside the restored net resolution: "
            f"fixed_idx={fixed_idx}, N_res={N_res}."
        )
        raise ValueError(msg)

    ij_q = np.zeros((N_res, 2), dtype=int)
    ij_q[:, fixed_axis] = fixed_idx
    ij_q[:, changing_axis] = np.arange(N_res, dtype=int)
    return ij_q


def _make_restored_net(
    net: CharNet,
    xyuv_q: np.ndarray,
    ij_q: np.ndarray,
) -> CharNet:
    net_R = CharNet(
        xyuv_q.shape[0],
        net.inlet,
        net.region,
        max_iters=net.max_iters,
        tol=net.tol,
    )
    net_R.wall = net.wall
    net_R.idl_kind = net.idl_kind
    net_R.prepared_j_stop = net.prepared_j_stop

    for q_idx, pt in enumerate(xyuv_q):
        i_q, j_q = ij_q[q_idx]
        point_type = 3 if q_idx in (0, xyuv_q.shape[0] - 1) else 2
        net_R.edit_point(int(i_q), int(j_q), pt, point_type)

    return net_R


def _point_on_wall(
    pt: np.ndarray,
    wall: PiecewiseLinearCurve,
    tol: float,
) -> np.ndarray | bool:
    pt = np.asarray(pt, dtype=float)
    single_pt = pt.ndim == 1
    xy = pt.reshape(1, 2) if single_pt else pt.reshape(-1, 2)
    wall_y = np.array([wall.get_y(x_val) for x_val in xy[:, 0]], dtype=float)
    wall_tol = max(1.0e-8, 100.0 * tol)
    on_wall = np.isfinite(wall_y) & (np.abs(xy[:, 1] - wall_y) <= wall_tol)
    if single_pt:
        return bool(on_wall[0])
    return on_wall


def build_net_boundary(
    net: CharNet,
    family: str,  # downstream/shock family
    main_wall: PiecewiseLinearCurve,
    xy_shock: np.ndarray,
    ij_shock: np.ndarray,
    tol: float = 1.0e-10,
):
    _, xy_wall = net.point_ij_xy("wall")
    ij_fluid, xy_fluid = net.point_ij_xy("fluid")
    ij_corner, xy_corner = net.point_ij_xy("corner")

    shock_corner_mask = (ij_corner[:, None] == ij_shock).all(axis=2).any(axis=1)
    corner_wall_mask = _point_on_wall(xy_corner, main_wall, tol)

    if family == "cminus":  # find first cplus char (i=0)
        upstream_mask = ij_fluid[:, 0] == 0
        corner_le_mask = (
            ij_corner[:, 0] == 0
        ) & corner_wall_mask  # leading edge corner
        corner_te_mask = shock_corner_mask & corner_wall_mask
        shock_order = np.argsort(ij_shock[:, 0])[::-1]
    else:
        upstream_mask = ij_fluid[:, 1] == 0
        corner_le_mask = (ij_corner[:, 1] == 0) & corner_wall_mask
        corner_te_mask = shock_corner_mask & corner_wall_mask
        shock_order = np.argsort(ij_shock[:, 1])[::-1]

    xy_upstream = xy_fluid[upstream_mask][
        np.argsort(xy_fluid[upstream_mask][:, 0])[::-1]
    ]  # Assemble points on first characteristic (EXCLUDES CORNERS!)

    xy_corner_le = xy_corner[corner_le_mask][:1]
    xy_corner_te = xy_corner[corner_te_mask][:1]

    x_wall_mask = (xy_wall[:, 0] > xy_corner_le[0, 0]) & (
        xy_wall[:, 0] < xy_corner_te[0, 0]
    )
    xy_wall = xy_wall[x_wall_mask]

    xy_wall = xy_wall[np.argsort(xy_wall[:, 0])]
    xy_shock = xy_shock[shock_order]

    return np.vstack(
        (xy_corner_le, xy_wall, xy_shock, xy_upstream, xy_corner_le)
    )  # this should form closed loop, counterclockwise


def _points_on_polyline(
    pts: np.ndarray, polyline: np.ndarray, tol: float
) -> np.ndarray:
    on_boundary = np.zeros(pts.shape[0], dtype=bool)
    finite_pts = np.all(np.isfinite(pts), axis=1)
    if polyline.shape[0] < 2 or not np.any(finite_pts):
        return on_boundary

    seg0 = polyline[:-1]
    seg1 = polyline[1:]
    delta = seg1 - seg0
    seg_len2 = np.sum(delta * delta, axis=1)

    pts_finite = pts[finite_pts]
    diff = pts_finite[:, None, :] - seg0[None, :, :]
    numer = np.sum(diff * delta[None, :, :], axis=2)

    t = np.divide(
        numer,
        seg_len2[None, :],
        out=np.full(numer.shape, np.nan),
        where=seg_len2[None, :] > tol,
    )
    t = np.clip(t, 0.0, 1.0)

    proj = seg0[None, :, :] + t[:, :, None] * delta[None, :, :]
    dist2 = np.sum((pts_finite[:, None, :] - proj) ** 2, axis=2)
    boundary_hits = np.where(seg_len2[None, :] > tol, dist2 <= tol * tol, False)

    if np.any(seg_len2 <= tol):
        degenerate = np.flatnonzero(seg_len2 <= tol)
        dist2_deg = np.sum(
            (pts_finite[:, None, :] - seg0[degenerate][None, :, :]) ** 2,
            axis=2,
        )
        boundary_hits[:, degenerate] = dist2_deg <= tol * tol

    on_boundary[finite_pts] = np.any(boundary_hits, axis=1)
    return on_boundary


def enforce_net_bounds(
    net: CharNet,
    family: str,
    wall: PiecewiseLinearCurve,
    xy_shock: np.ndarray,
    ij_shock: np.ndarray | None = None,
    tol: float = 1.0e-10,
    *,
    debug_plots: bool = False,
    debug_plotter: Callable[[], tuple[object, object]] | None = None,
) -> None:
    """
    Constructs net boundaries from point types and shock coordinates
    Deactivates downstream, non-physical points after shock-thru-net solve
    """
    polygon_xy = build_net_boundary(
        net,
        family=family,
        main_wall=wall,
        xy_shock=xy_shock,
        ij_shock=ij_shock,
        tol=tol,
    )
    active_rows, active_cols = np.nonzero(net.xy_mask())
    pts = np.column_stack(
        (net.x[active_rows, active_cols], net.y[active_rows, active_cols])
    )
    boundary_xy = polygon_xy
    polygon_path = Path(boundary_xy, closed=True)
    inside = polygon_path.contains_points(pts)
    on_boundary = _points_on_polyline(
        pts,
        boundary_xy,
        tol=max(1.0e-8, 10.0 * tol),
    )
    keep = inside | on_boundary
    if np.all(keep):
        return

    rows_delete = active_rows[~keep]
    cols_delete = active_cols[~keep]
    _plot_cleanup_candidates(
        net,
        rows_delete,
        cols_delete,
        debug_plots=debug_plots,
        debug_plotter=debug_plotter,
        boundary_xy=boundary_xy,
    )

    net.deactivate_points(rows_delete, cols_delete)
    return


def _active_fixed_indices(net: CharNet, family: str) -> np.ndarray:
    mask = net.state_mask()
    if family == "cminus":
        counts = np.count_nonzero(mask, axis=0)
    else:
        counts = np.count_nonzero(mask, axis=1)
    return np.flatnonzero(counts >= 2)


def _active_free_indices(
    net: CharNet, fixed_idx: int, family: str
) -> np.ndarray | None:
    mask = net.state_mask()
    if family == "cminus":
        return np.flatnonzero(mask[:, fixed_idx])
    return np.flatnonzero(mask[fixed_idx, :])


def _get_family_point(
    net: CharNet, fixed_idx: int, free_idx: int, family: str
) -> np.ndarray:
    if family == "cminus":
        return net.get_point(free_idx, fixed_idx)
    return net.get_point(fixed_idx, free_idx)


def _family_segments(
    net: CharNet,
    fixed_idx: int,
    family: str,
    free_idx_max: int | None = None,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    free_inds = _active_free_indices(net, fixed_idx, family)
    if free_idx_max is not None:
        free_inds = free_inds[free_inds <= free_idx_max]
    if free_inds.size < 2:
        return []

    segments = []
    for free_lo, free_hi in pairwise(free_inds):
        pt0 = _get_family_point(net, fixed_idx, int(free_lo), family)
        pt1 = _get_family_point(net, fixed_idx, int(free_hi), family)
        segments.append((int(free_lo), int(free_hi), pt0, pt1))
    return segments


def _segment_arrays(
    segments: list[tuple[int, int, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    free_hi = np.array([seg[1] for seg in segments], dtype=int)
    pt0 = np.array([seg[2] for seg in segments], dtype=float)
    pt1 = np.array([seg[3] for seg in segments], dtype=float)
    eqns = get_tang_from_pts_vectorized(pt0, pt1)
    return free_hi, pt0, pt1, eqns


def clip_coalescing_same_family_chars(
    net: CharNet,
    family: str,
    tol: float = 1e-10,
    *,
    debug_plots: bool = False,
    debug_plotter: Callable[[], tuple[object, object]] | None = None,
) -> None:
    """
    Post-process a completed net and clip any same-family characteristic that
    characteristic that crosses its adjacent higher-index neighbor.

    The higher-index characteristic is preserved, while the offending
    lower-index characteristic is deactivated from the first intersecting
    segment endpoint onward.
    """
    while True:
        fixed_inds = _active_fixed_indices(net, family)
        changed_pass = False

        for clip_idx, keep_idx in pairwise(fixed_inds):
            clip_segments = _family_segments(net, int(clip_idx), family)
            keep_segments = _family_segments(net, int(keep_idx), family)
            if not clip_segments or not keep_segments:
                continue

            clip_free_hi, clip_pt0, clip_pt1, clip_eqns = _segment_arrays(clip_segments)
            _, keep_pt0, keep_pt1, keep_eqns = _segment_arrays(keep_segments)

            xy_hit = get_intersection_vectorized(clip_eqns, keep_eqns)
            t_clip = _segment_parameter_array(
                xy_hit,
                clip_pt0[:, None, :],
                clip_pt1[:, None, :],
                tol,
            )
            t_keep = _segment_parameter_array(
                xy_hit,
                keep_pt0[None, :, :],
                keep_pt1[None, :, :],
                tol,
            )
            valid = (
                np.all(np.isfinite(xy_hit), axis=-1)
                & np.isfinite(t_clip)
                & np.isfinite(t_keep)
                & (tol < t_clip)
                & (t_clip < 1.0 - tol)
                & (tol < t_keep)
                & (t_keep < 1.0 - tol)
            )
            row_hits = np.any(valid, axis=1)
            if not np.any(row_hits):
                continue

            clip_start = int(clip_free_hi[np.flatnonzero(row_hits)[0]])
            _plot_coalescing_cleanup(
                net,
                family=family,
                clip_idx=int(clip_idx),
                keep_idx=int(keep_idx),
                clip_start=clip_start,
                intersection_xy=xy_hit[valid],
                debug_plots=debug_plots,
                debug_plotter=debug_plotter,
            )

            clip_char(
                net,
                fixed_idx=int(clip_idx),
                free_inds=np.arange(int(clip_start), net.N, dtype=int),
                family=family,
            )
            changed_pass = True
            break

        if not changed_pass:
            return
