from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from inlet_moc.planar_inlet import PiecewiseLinearCurve

WALL_ROOT_TOL = 1.0e-10


class NoWallIntersectionError(ValueError):
    """Raised when no valid wall/root intersection can be found."""


def get_Vmag(u: float, v: float):
    return math.hypot(u, v)


def get_angles(u: float, v: float, Vm: float, a: float):
    if not (math.isfinite(Vm) and math.isfinite(a)) or (Vm <= a):
        from inlet_moc.char_shock_solvers import SubsonicFlowError

        msg = (
            "Subsonic flow encountered in characteristic solver: "
            f"V={Vm:.6g}, a={a:.6g}."
        )
        raise SubsonicFlowError(msg)
    theta = math.atan2(v, u)
    alpha = math.asin(a / Vm)
    return theta, alpha


def get_QRS(u, v, a, y, delta, L):
    Q = u * u - a * a
    R = (2 * u * v) - Q * L
    S = (delta * a * a * v) / y
    return Q, R, S


def get_LQRS_plus(yuv, a_fxn, delta=0):
    y, u, v = yuv
    Lp, Qp, Rp, Sp = _get_LQRS_plus_values(y, u, v, a_fxn, delta)
    return np.array([Lp, Qp, Rp, Sp])


def _get_LQRS_plus_values(y, u, v, a_fxn, delta=0):
    y = float(y)
    u = float(u)
    v = float(v)
    Vm = get_Vmag(u, v)
    a = float(a_fxn(u, v))
    theta, alpha = get_angles(u, v, Vm, a)
    Lp = math.tan(theta + alpha)
    Qp, Rp, Sp = get_QRS(u, v, a, y, delta, Lp)
    return Lp, Qp, Rp, Sp


def get_LQRS_min(yuv, a_fxn, delta=0):
    y, u, v = yuv
    Lm, Qm, Rm, Sm = _get_LQRS_min_values(y, u, v, a_fxn, delta)
    return np.array([Lm, Qm, Rm, Sm])


def _get_LQRS_min_values(y, u, v, a_fxn, delta=0):
    y = float(y)
    u = float(u)
    v = float(v)
    Vm = get_Vmag(u, v)
    a = float(a_fxn(u, v))
    theta, alpha = get_angles(u, v, Vm, a)
    Lm = math.tan(theta - alpha)
    Qm, Rm, Sm = get_QRS(u, v, a, y, delta, Lm)
    return Lm, Qm, Rm, Sm


def _iter_converged(pt_new: np.ndarray, pt_old: np.ndarray, tol: float) -> bool:
    return bool(
        np.all(np.isfinite(pt_new))
        and np.all(np.isfinite(pt_old))
        and np.linalg.norm(pt_new - pt_old, ord=np.inf) <= tol
    )


def _point_values_converged(
    x_new: float,
    y_new: float,
    u_new: float,
    v_new: float,
    x_old: float,
    y_old: float,
    u_old: float,
    v_old: float,
    tol: float,
) -> bool:
    return (
        math.isfinite(x_new)
        and math.isfinite(y_new)
        and math.isfinite(u_new)
        and math.isfinite(v_new)
        and math.isfinite(x_old)
        and math.isfinite(y_old)
        and math.isfinite(u_old)
        and math.isfinite(v_old)
        and max(
            abs(x_new - x_old),
            abs(y_new - y_old),
            abs(u_new - u_old),
            abs(v_new - v_old),
        )
        <= tol
    )


def field_point(
    pt_plus: np.ndarray,
    pt_minus: np.ndarray,
    a_fxn: Callable,
    max_iters: int = 10,
    tol: float = 1e-6,
):
    xp = float(pt_plus[0])
    yp = float(pt_plus[1])
    y_plus = yp
    u_plus = float(pt_plus[2])
    v_plus = float(pt_plus[3])

    xm = float(pt_minus[0])
    ym = float(pt_minus[1])
    y_minus = ym
    u_minus = float(pt_minus[2])
    v_minus = float(pt_minus[3])

    x_out, y_out, u_out, v_out = _iter_field_point_values(
        y_plus,
        u_plus,
        v_plus,
        y_minus,
        u_minus,
        v_minus,
        xp,
        yp,
        xm,
        ym,
        a_fxn,
    )

    for _ in range(max(0, int(max_iters))):
        y_plus = 0.5 * (y_out + y_plus)
        u_plus = 0.5 * (u_out + u_plus)
        v_plus = 0.5 * (v_out + v_plus)
        y_minus = 0.5 * (y_out + y_minus)
        u_minus = 0.5 * (u_out + u_minus)
        v_minus = 0.5 * (v_out + v_minus)
        x_next, y_next, u_next, v_next = _iter_field_point_values(
            y_plus,
            u_plus,
            v_plus,
            y_minus,
            u_minus,
            v_minus,
            xp,
            yp,
            xm,
            ym,
            a_fxn,
        )
        if _point_values_converged(
            x_next,
            y_next,
            u_next,
            v_next,
            x_out,
            y_out,
            u_out,
            v_out,
            tol,
        ):
            x_out = x_next
            y_out = y_next
            u_out = u_next
            v_out = v_next
            break
        x_out = x_next
        y_out = y_next
        u_out = u_next
        v_out = v_next
    return np.array([x_out, y_out, u_out, v_out])


def iter_field_point(yuv_plus, yuv_minus, xyp, xym, a_fxn):
    y_plus, u_plus, v_plus = yuv_plus
    y_minus, u_minus, v_minus = yuv_minus
    xp, yp = xyp
    xm, ym = xym
    return np.array(
        _iter_field_point_values(
            y_plus,
            u_plus,
            v_plus,
            y_minus,
            u_minus,
            v_minus,
            xp,
            yp,
            xm,
            ym,
            a_fxn,
        )
    )


def _iter_field_point_values(
    y_plus,
    u_plus,
    v_plus,
    y_minus,
    u_minus,
    v_minus,
    xp,
    yp,
    xm,
    ym,
    a_fxn,
):
    Lp, Qp, Rp, Sp = _get_LQRS_plus_values(y_plus, u_plus, v_plus, a_fxn)
    Lm, Qm, Rm, Sm = _get_LQRS_min_values(y_minus, u_minus, v_minus, a_fxn)

    xp = float(xp)
    yp = float(yp)
    xm = float(xm)
    ym = float(ym)

    if math.isinf(Lm):
        x_new = xm
        y_new = yp - Lp * (xp - x_new)

    elif math.isinf(Lp):
        x_new = xp
        y_new = ym - Lm * (xm - x_new)
    else:
        x_new = (ym - yp - Lm * xm + Lp * xp) / (Lp - Lm)
        y_new = ym + Lm * (x_new - xm)

    Tp = Sp * (x_new - xp) + Qp * u_plus + Rp * v_plus
    Tm = Sm * (x_new - xm) + Qm * u_minus + Rm * v_minus

    det = Qm * Rp - Qp * Rm
    u_new = (Tm * Rp - Tp * Rm) / det
    v_new = (Qm * Tp - Qp * Tm) / det

    return x_new, y_new, u_new, v_new


def _segment_line_intersection_x(
    x0: float,
    y0: float,
    m_seg: float,
    x1: float,
    x_fix: float,
    y_fix: float,
    m_char: float,
    tol: float,
):
    if math.isinf(m_char):
        x_hit = x_fix
    else:
        denom = m_seg - m_char
        if abs(denom) <= tol:
            return None
        x_hit = (y_fix - (m_char * x_fix) - y0 + (m_seg * x0)) / denom

    if (x_hit < (x0 - tol)) or (x_hit > (x1 + tol)):
        return None
    return float(min(max(x_hit, x0), x1))


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
    """
    Find the piecewise-linear wall intersection using local wall segments.

    This avoids solving across the entire wall at once. If the chosen hit lies on
    a wall corner, the selected segment index supplies the appropriate one-sided
    wall slope for the boundary condition without altering the wall geometry.
    """
    candidates: list[tuple[float, int]] = []

    for seg_idx, m_seg in enumerate(wall.slopes):
        x0 = float(wall.x[seg_idx])
        x1 = float(wall.x[seg_idx + 1])
        y0 = float(wall.y[seg_idx])
        x_hit = _segment_line_intersection_x(
            x0,
            y0,
            float(m_seg),
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
    m_wall = float(wall.slopes[seg_idx])
    y_new = float(wall.y[seg_idx] + m_wall * (x_new - wall.x[seg_idx]))
    return x_new, y_new, m_wall


def wall_point(
    plus_arg: np.ndarray | PiecewiseLinearCurve,
    minus_arg: np.ndarray | PiecewiseLinearCurve,
    a_fxn: Callable,
    delta=0,
    max_iters: int = 10,
    tol: float = 1e-6,
):
    if (isinstance(plus_arg, np.ndarray)) and (
        isinstance(minus_arg, PiecewiseLinearCurve)
    ):
        pt = plus_arg
        wall = minus_arg
        solver = _iter_wall_point_plus_values
    else:
        solver = _iter_wall_point_minus_values
        pt = minus_arg
        wall = plus_arg

    x_fix = float(pt[0])
    y_fix = float(pt[1])
    y_iter = float(pt[1])
    u_iter = float(pt[2])
    v_iter = float(pt[3])

    x_out, y_out, u_out, v_out = solver(
        y_iter,
        u_iter,
        v_iter,
        x_fix,
        y_fix,
        a_fxn,
        wall,
        tol=tol,
    )

    for _ in range(max(0, int(max_iters))):
        y_iter = 0.5 * (y_out + y_iter)
        u_iter = 0.5 * (u_out + u_iter)
        v_iter = 0.5 * (v_out + v_iter)
        x_next, y_next, u_next, v_next = solver(
            y_iter,
            u_iter,
            v_iter,
            x_fix,
            y_fix,
            a_fxn,
            wall,
            tol=tol,
        )
        if _point_values_converged(
            x_next,
            y_next,
            u_next,
            v_next,
            x_out,
            y_out,
            u_out,
            v_out,
            tol,
        ):
            x_out = x_next
            y_out = y_next
            u_out = u_next
            v_out = v_next
            break
        x_out = x_next
        y_out = y_next
        u_out = u_next
        v_out = v_next
    return np.array([x_out, y_out, u_out, v_out])


def iter_wall_point_plus(
    yuv_plus,
    xyp,
    a_fxn,
    wall: PiecewiseLinearCurve,
    tol: float = WALL_ROOT_TOL,
):
    y_plus, u_plus, v_plus = yuv_plus
    xp, yp = xyp
    return np.array(
        _iter_wall_point_plus_values(
            y_plus,
            u_plus,
            v_plus,
            xp,
            yp,
            a_fxn,
            wall,
            tol=tol,
        )
    )


def _iter_wall_point_plus_values(
    y_plus,
    u_plus,
    v_plus,
    xp,
    yp,
    a_fxn,
    wall: PiecewiseLinearCurve,
    tol: float = WALL_ROOT_TOL,
):
    Lp, Qp, Rp, Sp = _get_LQRS_plus_values(y_plus, u_plus, v_plus, a_fxn)
    xp = float(xp)
    yp = float(yp)

    if math.isinf(Lp):
        x_new = xp
        y_new = wall.get_y(x_new)
        dydx_new = wall.get_dydx(x_new)
    else:
        x_new, y_new, dydx_new = _solve_wall_intersection(xp, yp, Lp, wall, tol=tol)

    Tp = Sp * (x_new - xp) + Qp * u_plus + Rp * v_plus
    u_new = Tp / (Qp + Rp * dydx_new)
    v_new = dydx_new * u_new
    return x_new, y_new, u_new, v_new


def iter_wall_point_minus(
    yuv_minus,
    xym,
    a_fxn,
    wall: PiecewiseLinearCurve,
    tol: float = WALL_ROOT_TOL,
):
    y_minus, u_minus, v_minus = yuv_minus
    xm, ym = xym
    return np.array(
        _iter_wall_point_minus_values(
            y_minus,
            u_minus,
            v_minus,
            xm,
            ym,
            a_fxn,
            wall,
            tol=tol,
        )
    )


def _iter_wall_point_minus_values(
    y_minus,
    u_minus,
    v_minus,
    xm,
    ym,
    a_fxn,
    wall: PiecewiseLinearCurve,
    tol: float = WALL_ROOT_TOL,
):
    Lm, Qm, Rm, Sm = _get_LQRS_min_values(y_minus, u_minus, v_minus, a_fxn)
    xm = float(xm)
    ym = float(ym)

    if math.isinf(abs(Lm)):
        x_new = xm
        y_new = wall.get_y(x_new)
        dydx_new = wall.get_dydx(x_new)
    else:
        x_new, y_new, dydx_new = _solve_wall_intersection(xm, ym, Lm, wall, tol=tol)

    Tm = Sm * (x_new - xm) + Qm * u_minus + Rm * v_minus
    u_new = Tm / (Qm + Rm * dydx_new)
    v_new = dydx_new * u_new
    return x_new, y_new, u_new, v_new
