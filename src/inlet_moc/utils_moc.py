from __future__ import annotations

import numpy as np
import pandas as pd


def cleaner(array: np.ndarray[tuple[int]]):
    return array[~np.isnan(array)]


def get_tang(pt0: np.ndarray, m0: float) -> np.ndarray:
    """
    converts point coordinates (pt0, up to 4, first two args are x and y)
    and slope of line to coefficient form
    """
    return np.array([m0, -1, (pt0[1] - m0 * pt0[0])])


def get_norm(pt0: np.ndarray, m0: float) -> np.ndarray:
    m_n = -1 / m0
    return np.array([m_n, -1, (pt0[1] - m_n * pt0[0])])


def get_slope(pt0: np.ndarray, pt1: np.ndarray):
    return (pt1[1] - pt0[1]) / (pt1[0] - pt0[0])


def get_intersection(coeffs1, coeffs2) -> np.ndarray | None:
    """
    coeffs = [a, b, c] s.t. a * x + b * y + c = 0
    returns (2,) array, with (x,y)_intersect
    """
    a1, b1, c1 = coeffs1
    a2, b2, c2 = coeffs2

    denom = (a1 * b2) - (a2 * b1)
    if np.isclose(denom, 0.0):
        return None

    return np.array([(b1 * c2 - b2 * c1) / denom, (a2 * c1 - a1 * c2) / denom])


def get_theta(pt: np.ndarray):
    return pt[3]


def get_tang_from_pts(pt0: np.ndarray, pt1: np.ndarray) -> np.ndarray:
    m = get_slope(pt0, pt1)
    return get_tang(pt0, m)


def get_tang_from_pts_vectorized(point_set0, point_set1) -> np.ndarray:
    """
    Vectorized tangent-line coefficients for paired point sets.

    Parameters
    ----------
    point_set0, point_set1:
        Arrays of shape ``(N, >=2)`` whose leading columns are ``x`` and ``y``.

    Returns
    -------
    ndarray
        Coefficients of shape ``(N, 3)`` for lines ``a*x + b*y + c = 0``.
    """
    point_set0 = np.asarray(point_set0, dtype=float)
    point_set1 = np.asarray(point_set1, dtype=float)
    if point_set0.ndim != 2 or point_set1.ndim != 2:
        msg = "Expected 2D point arrays for vectorized tangent construction."
        raise ValueError(msg)
    if point_set0.shape != point_set1.shape:
        msg = "Point arrays must have matching shapes."
        raise ValueError(msg)
    if point_set0.shape[1] < 2:
        msg = "Point arrays must include at least x and y columns."
        raise ValueError(msg)

    y1 = point_set1[:, 1]
    y0 = point_set0[:, 1]
    x1 = point_set1[:, 0]
    x0 = point_set0[:, 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        a = (y1 - y0) / (x1 - x0)
    b = np.full_like(a, -1.0)
    c = y0 - a * x0

    return np.column_stack((a, b, c))


def get_intersection_vectorized(coeffs1, coeffs2):
    """
    Vectorized line intersections for coefficient arrays.

    Parameters
    ----------
    coeffs1 : array_like
        Shape ``(N, 3)`` or ``(3,)``.
    coeffs2 : array_like
        Shape ``(M, 3)`` or ``(3,)``.

    Returns
    -------
    ndarray
        Intersections of shape ``(N, M, 2)``. Parallel pairs are returned as
        ``NaN`` rows.
    """
    coeffs1 = np.asarray(coeffs1, dtype=float)
    coeffs2 = np.asarray(coeffs2, dtype=float)
    if coeffs1.ndim == 1:
        coeffs1 = coeffs1[None, :]
    if coeffs2.ndim == 1:
        coeffs2 = coeffs2[None, :]
    if coeffs1.shape[-1] != 3 or coeffs2.shape[-1] != 3:
        msg = "Line coefficient arrays must end with length 3."
        raise ValueError(msg)

    a1 = coeffs1[:, None, 0]
    b1 = coeffs1[:, None, 1]
    c1 = coeffs1[:, None, 2]
    a2 = coeffs2[None, :, 0]
    b2 = coeffs2[None, :, 1]
    c2 = coeffs2[None, :, 2]

    denom = (a1 * b2) - (a2 * b1)
    xy = np.full((*denom.shape, 2), np.nan, dtype=float)
    valid = ~np.isclose(denom, 0.0)

    np.divide(
        (b1 * c2 - b2 * c1),
        denom,
        out=xy[..., 0],
        where=valid,
    )
    np.divide(
        (a2 * c1 - a1 * c2),
        denom,
        out=xy[..., 1],
        where=valid,
    )
    return xy


def pt_to_plotvec(x0, y0, beta, L: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    dx = np.cos(beta)
    dy = np.sin(beta)
    mag = np.hypot(dx, dy)
    dx /= mag
    dy /= mag

    x = np.array([x0, x0 + L * dx])
    y = np.array([y0, y0 + L * dy])
    return x, y


def pts_to_plotseg(p1, p2):
    x_pts = [p1[0], p2[0]]
    y_pts = [p1[1], p2[1]]
    return x_pts, y_pts


def geo_reader(filepath) -> list[np.ndarray]:
    first_row = pd.read_csv(filepath, nrows=1, header=None).iloc[0]

    skip_first = any(pd.to_numeric(first_row, errors="coerce").isna())

    walls = pd.read_csv(filepath, header=None, skiprows=1 if skip_first else 0)

    ncols = walls.shape[1]
    if ncols % 2 != 0:
        msg = "CSV must have an even number of columns (x,y pairs)."
        raise ValueError(msg)
    nbodies = ncols // 2

    bodies = []
    for i in range(nbodies):
        x = cleaner(walls.iloc[:, 2 * i].to_numpy())
        y = cleaner(walls.iloc[:, 2 * i + 1].to_numpy())
        bodies.append(np.column_stack((x, y)))

    return bodies


def order_cell_vertices(verts: np.ndarray) -> np.ndarray:
    verts = np.asarray(verts, dtype=float)
    centroid = np.mean(verts, axis=0)
    angles = np.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0])
    return verts[np.argsort(angles)]


def interp_pts(
    pt1: np.ndarray, pt2: np.ndarray, xy_q, clip: bool = False
) -> np.ndarray:
    pt1 = np.asarray(pt1, dtype=float).reshape(-1)
    pt2 = np.asarray(pt2, dtype=float).reshape(-1)
    xy_q = np.asarray(xy_q, dtype=float)
    if pt1.shape != pt2.shape:
        msg = "Interpolation endpoints must have matching state shapes."
        raise ValueError(msg)
    if pt1.size < 2:
        msg = "Interpolation endpoints must include at least x and y."
        raise ValueError(msg)
    dxy = pt2[:2] - pt1[:2]
    seg_len2 = float(np.dot(dxy, dxy))

    r_frac = 0.0 if seg_len2 <= 0.0 else float(np.dot(xy_q - pt1[:2], dxy) / seg_len2)
    if clip:
        r_frac = float(np.clip(r_frac, 0.0, 1.0))

    state_q = pt1[2:] + r_frac * (pt2[2:] - pt1[2:])
    return np.concatenate((xy_q[:2], state_q))


def intersect_line_polyline(coeffs: np.ndarray, pts: np.ndarray, tol: float = 1e-12):
    """
    coeffs: shape (3,), [a, b, c] for a*x + b*y + c = 0
    pts:    shape (N, 6), rows are [x, y, V, theta, p, rho]

    returns:
        ``(pt, idx)`` when an intersection is found, otherwise ``None``.
    """
    a, b, c = coeffs
    pts = np.asarray(pts, dtype=float)

    if pts.shape[0] == 0:
        return None
    if pts.shape[0] == 1:
        scale = np.sqrt(a * a + b * b)
        a, b, c = a / scale, b / scale, c / scale
        f0 = a * pts[0, 0] + b * pts[0, 1] + c
        if abs(f0) <= tol:
            return pts[0].copy(), 0
        return None

    scale = np.sqrt(a * a + b * b)
    a, b, c = a / scale, b / scale, c / scale

    pts0 = pts[:-1]
    p1 = pts[1:]
    dp = p1 - pts0

    f0 = a * pts0[:, 0] + b * pts0[:, 1] + c
    f1 = a * p1[:, 0] + b * p1[:, 1] + c

    z0 = np.abs(f0) <= tol
    z1 = np.abs(f1) <= tol

    hit = z0 | ((f0 * f1) < 0.0)

    if not np.any(hit):
        if z1[-1]:
            return pts[-1].copy(), int(pts.shape[0] - 1)
        return None

    i = np.argmax(hit)
    if z0[i]:
        return pts0[i].copy(), int(i)
    r = f0[i] / (f0[i] - f1[i])
    return pts0[i] + r * dp[i], int(i + 1)
