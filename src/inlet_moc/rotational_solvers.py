from __future__ import annotations

import math

import numpy as np

from inlet_moc.planar_inlet import PiecewiseLinearCurve
from inlet_moc.utils_moc import interp_pts

WALL_ROOT_TOL = 1.0e-10
ROT_XY_TOL = 1.0e-3
ROT_STATE_TOL = 1.0e-2


class RotationalSolveError(ValueError):
    """Raised when the rotational MOC solve cannot form a physical point."""


class NoWallIntersectionError(ValueError):
    """Raised when no valid wall/root intersection can be found."""


def get_qrs(
    pt: tuple[float, float, float, float, float, float] | np.ndarray,
    gamma: float,
    sign: int = 1,
    delta: float = 0.0,
) -> tuple[float, float, float]:
    _, y, V, theta, p, rho = pt

    R = rho * V
    M2 = R * V / (gamma * p)
    if not math.isfinite(M2) or M2 <= 1.0:
        msg = (
            "Subsonic or invalid rotational characteristic state: "
            f"V={V:.6g}, theta={theta:.6g}, p={p:.6g}, "
            f"rho={rho:.6g}, M2={M2:.6g}."
        )
        raise RotationalSolveError(msg)

    M = math.sqrt(M2)
    alpha = math.asin(1.0 / M)

    Q = math.sqrt(M2 - 1.0) / (R * V)
    if delta == 0.0:
        S = 0.0
    else:
        S = delta * math.sin(theta) / (y * M * math.cos(theta + sign * alpha))

    return Q, R, S


def _get_lqrs_values(
    pt: tuple[float, float, float, float, float, float] | np.ndarray,
    gamma: float,
    sign: int,
    delta: float = 0.0,
) -> tuple[float, float, float, float]:
    _, _, V, theta, p, rho = pt
    M2 = rho * V * V / (gamma * p)
    if not math.isfinite(M2) or M2 <= 1.0:
        msg = (
            "Subsonic or invalid rotational characteristic state: "
            f"V={V:.6g}, theta={theta:.6g}, p={p:.6g}, "
            f"rho={rho:.6g}, M2={M2:.6g}."
        )
        raise RotationalSolveError(msg)

    M = math.sqrt(M2)
    alpha = math.asin(1.0 / M)
    L = math.tan(theta + sign * alpha)
    Q, R, S = get_qrs(pt, gamma, sign=sign, delta=delta)
    return L, Q, R, S


def _point_values_converged(
    pt_new: np.ndarray,
    pt_old: np.ndarray,
    xy_tol: float = ROT_XY_TOL,
    state_tol: float = ROT_STATE_TOL,
) -> bool:
    tol_vec = np.array(
        [
            float(xy_tol),
            float(xy_tol),
            float(state_tol),
            float(state_tol),
            float(state_tol),
            float(state_tol),
        ]
    )
    return bool(
        np.all(np.isfinite(pt_new))
        and np.all(np.isfinite(pt_old))
        and np.all(np.abs(pt_new - pt_old) <= tol_vec)
    )


def _segment_line_intersection_x(
    x0: float,
    y0: float,
    m_seg: float,
    x1: float,
    x_fix: float,
    y_fix: float,
    m_char: float,
    tol: float,
) -> float | None:
    if math.isinf(m_char):
        x_hit = x_fix
    else:
        denom = m_seg - m_char
        if abs(denom) <= tol:
            return None
        x_hit = (y_fix - (m_char * x_fix) - y0 + (m_seg * x0)) / denom

    if (x_hit < (x0 - tol)) or (x_hit > (x1 + tol)):
        return None
    return min(max(x_hit, x0), x1)


def _choose_wall_root_candidate(
    candidates: list[tuple[float, int]],
    x_fix: float,
    tol: float,
) -> tuple[float, int]:
    if not candidates:
        msg = "No wall intersection found on any local wall segment."
        raise NoWallIntersectionError(msg)

    def key(cand):
        x_hit, seg_idx = cand
        dx = x_hit - x_fix
        if dx >= -tol:
            return (0, max(dx, 0.0), seg_idx)
        return (1, abs(dx), -seg_idx)

    return min(candidates, key=key)


def _solve_wall_intersection(
    x_fix: float,
    y_fix: float,
    char_slope: float,
    wall: PiecewiseLinearCurve,
    tol: float = WALL_ROOT_TOL,
) -> tuple[float, float, float]:
    candidates: list[tuple[float, int]] = []

    for seg_idx, m_seg in enumerate(wall.slopes):
        x0 = wall.x[seg_idx]
        x1 = wall.x[seg_idx + 1]
        y0 = wall.y[seg_idx]
        x_hit = _segment_line_intersection_x(
            x0,
            y0,
            m_seg,
            x1,
            x_fix,
            y_fix,
            char_slope,
            tol,
        )
        if x_hit is None:
            continue
        candidates.append((x_hit, seg_idx))

    x_new, seg_idx = _choose_wall_root_candidate(candidates, x_fix, tol)
    m_wall = wall.slopes[seg_idx]
    y_new = wall.y[seg_idx] + m_wall * (x_new - wall.x[seg_idx])
    return x_new, y_new, m_wall


def _interp_segment_point(
    x4: float,
    y4: float,
    theta3: float,
    theta4: float,
    pt_minus: np.ndarray,
    pt_plus: np.ndarray,
    iter_idx: int,
    interp_tol: float,
) -> tuple[float, float, float, float, float, float]:
    x1, y1, V1, theta1, p1, rho1 = pt_minus
    x2, y2, V2, theta2, p2, rho2 = pt_plus
    pt_minus = np.array([x1, y1, V1, theta1, p1, rho1])
    pt_plus = np.array([x2, y2, V2, theta2, p2, rho2])

    vertical_segment = x1 == x2
    if not vertical_segment:
        L12 = (y1 - y2) / (x1 - x2)
    dxy = pt_minus[:2] - pt_plus[:2]
    seg_len2 = float(np.dot(dxy, dxy))
    k = 1
    yc = math.nan

    for _ in range(50):
        L0 = math.tan(0.5 * (theta3 + theta4))
        if vertical_segment:
            x3 = x1
        else:
            denom = L12 - L0
            if abs(denom) <= interp_tol:
                if seg_len2 <= 0.0:
                    x3, y3 = x2, y2
                else:
                    r = np.dot(np.array([x4, y4]) - pt_plus[:2], dxy) / seg_len2
                    r = float(np.clip(r, 0.0, 1.0))
                    x3, y3 = pt_plus[:2] + r * dxy
            else:
                x3 = (y4 - y2 - L0 * x4 + L12 * x2) / denom
                y3 = y4 + L0 * (x3 - x4)
        if not (math.isfinite(x3) and math.isfinite(y3)):
            if seg_len2 <= 0.0:
                x3, y3 = x2, y2
            else:
                r = np.dot(np.array([x4, y4]) - pt_plus[:2], dxy) / seg_len2
                r = float(np.clip(r, 0.0, 1.0))
                x3, y3 = pt_plus[:2] + r * dxy
        pt3 = interp_pts(pt_plus, pt_minus, (x3, y3), clip=True)
        theta3 = pt3[3]
        if iter_idx == 0:
            theta4 = theta3
        if k > 1 and abs(y3 - yc) < interp_tol:
            break
        yc = y3
        k += 1
    else:
        msg = "Rotational interpolation point did not converge."
        raise RotationalSolveError(msg)

    return x3, y3, pt3[2], theta3, pt3[4], pt3[5]


def _family_sign(family: str) -> int:
    if family in ("cplus", "plus", "+"):
        return 1
    if family in ("cminus", "minus", "-"):
        return -1
    msg = "family must be 'cplus' or 'cminus'."
    raise ValueError(msg)


def field_point_rot(
    pt_plus: np.ndarray,
    pt_minus: np.ndarray,
    gamma: float,
    max_iters: int = 10,
    tol: float = 1e-6,
    delta: float = 0.0,
    interp_tol: float = ROT_XY_TOL,
) -> np.ndarray:
    x1, y1, V1, theta1, p1, rho1 = pt_minus
    x2, y2, V2, theta2, p2, rho2 = pt_plus

    pt_plus_iter = np.array([x2, y2, V2, theta2, p2, rho2])
    pt_minus_iter = np.array([x1, y1, V1, theta1, p1, rho1])

    theta3 = 0.5 * (theta1 + theta2)
    theta4 = theta3
    V4 = math.nan
    p4 = math.nan
    rho4 = math.nan

    x_out = math.nan
    y_out = math.nan
    V_out = math.nan
    theta_out = math.nan
    p_out = math.nan
    rho_out = math.nan

    for iter_idx in range(max(0, int(max_iters)) + 1):
        Lp, Qp, _, Sp = _get_lqrs_values(pt_plus_iter, gamma, sign=1, delta=delta)
        Lm, Qm, _, Sm = _get_lqrs_values(pt_minus_iter, gamma, sign=-1, delta=delta)

        x4 = (y1 - y2 - Lm * x1 + Lp * x2) / (Lp - Lm)
        y4 = y1 + Lm * (x4 - x1)
        if y4 < 0.0:
            msg = f"Rotational field point crossed below y=0: y={y4:.6g}."
            raise RotationalSolveError(msg)

        Tp = -Sp * (x4 - x2) + Qp * p2 + theta2
        Tm = -Sm * (x4 - x1) + Qm * p1 - theta1

        _x3, _y3, V3, theta3, p3, rho3 = _interp_segment_point(
            x4,
            y4,
            theta3,
            theta4,
            np.array([x1, y1, V1, theta1, p1, rho1]),
            np.array([x2, y2, V2, theta2, p2, rho2]),
            iter_idx,
            interp_tol,
        )
        if iter_idx == 0:
            theta4 = theta3
            V4 = V3
            p4 = p3
            rho4 = rho3

        p_avg = 0.5 * (p3 + p4)
        rho_avg = 0.5 * (rho3 + rho4)
        V_avg = 0.5 * (V3 + V4)

        R0 = rho_avg * V_avg
        a02 = gamma * p_avg / rho_avg
        T01 = R0 * V3 + p3
        T02 = p3 - a02 * rho3

        p4 = (Tp + Tm) / (Qp + Qm)
        theta4 = Tp - Qp * p4
        V4 = (T01 - p4) / R0
        rho4 = (p4 - T02) / a02

        pt_out = np.array([x4, y4, V4, theta4, p4, rho4])

        if iter_idx > 0 and _point_values_converged(
            pt_out,
            np.array([x_out, y_out, V_out, theta_out, p_out, rho_out]),
            xy_tol=max(float(tol), ROT_XY_TOL),
            state_tol=max(float(tol), ROT_STATE_TOL),
        ):
            return pt_out

        x_out = x4
        y_out = y4
        V_out = V4
        theta_out = theta4
        p_out = p4
        rho_out = rho4

        pt_plus_iter = np.array(
            [
                0.5 * (x2 + x4),
                0.5 * (y2 + y4),
                0.5 * (V2 + V4),
                0.5 * (theta2 + theta4),
                0.5 * (p2 + p4),
                0.5 * (rho2 + rho4),
            ]
        )
        pt_minus_iter = np.array(
            [
                0.5 * (x1 + x4),
                0.5 * (y1 + y4),
                0.5 * (V1 + V4),
                0.5 * (theta1 + theta4),
                0.5 * (p1 + p4),
                0.5 * (rho1 + rho4),
            ]
        )

    return np.array([x_out, y_out, V_out, theta_out, p_out, rho_out])


def wall_point_rot(
    pt_char: np.ndarray,
    pt_wall: np.ndarray,
    wall: PiecewiseLinearCurve,
    family: str,
    gamma: float,
    max_iters: int = 10,
    tol: float = 1e-6,
    delta: float = 0.0,
) -> np.ndarray:
    x_char, y_char, V_char, theta_char, p_char, rho_char = pt_char
    _x_wall, _y_wall, V_wall, theta_wall, p_wall, rho_wall = pt_wall

    sign = _family_sign(family)
    pt_char_iter = np.array([x_char, y_char, V_char, theta_char, p_char, rho_char])

    V4 = V_wall
    theta4 = theta_wall
    p4 = p_wall
    rho4 = rho_wall

    pt_out_prev = np.full(6, np.nan)

    for _ in range(max(0, int(max_iters)) + 1):
        L, Q, _, S = _get_lqrs_values(pt_char_iter, gamma, sign=sign, delta=delta)

        x4, y4, wall_slope = _solve_wall_intersection(
            x_char,
            y_char,
            L,
            wall,
            tol=tol,
        )
        theta4 = math.atan(wall_slope)

        if sign > 0:
            T_char = -S * (x4 - x_char) + Q * p_char + theta_char
            p4 = (T_char - theta4) / Q
        else:
            T_char = -S * (x4 - x_char) + Q * p_char - theta_char
            p4 = (T_char + theta4) / Q

        p_avg = 0.5 * (p_wall + p4)
        rho_avg = 0.5 * (rho_wall + rho4)
        V_avg = 0.5 * (V_wall + V4)

        R0 = rho_avg * V_avg
        a02 = gamma * p_avg / rho_avg
        T01 = R0 * V_wall + p_wall
        T02 = p_wall - a02 * rho_wall

        V4 = (T01 - p4) / R0
        rho4 = (p4 - T02) / a02

        pt_out = np.array([x4, y4, V4, theta4, p4, rho4])
        if _point_values_converged(
            pt_out,
            pt_out_prev,
            xy_tol=max(float(tol), ROT_XY_TOL),
            state_tol=max(float(tol), ROT_STATE_TOL),
        ):
            return pt_out

        pt_out_prev = pt_out
        pt_char_iter = np.array(
            [
                0.5 * (x_char + x4),
                0.5 * (y_char + y4),
                0.5 * (V_char + V4),
                0.5 * (theta_char + theta4),
                0.5 * (p_char + p4),
                0.5 * (rho_char + rho4),
            ]
        )

    return pt_out_prev


field_point = field_point_rot


__all__ = [
    "NoWallIntersectionError",
    "RotationalSolveError",
    "field_point",
    "field_point_rot",
    "get_qrs",
    "wall_point_rot",
]
