from __future__ import annotations

import math

import numpy as np
from scipy.optimize import root_scalar

from inlet_moc.planar_inlet import PiecewiseLinearCurve
from inlet_moc.utils_moc import interp_pts

WALL_ROOT_TOL = 1.0e-10
ROT_XY_TOL = 1.0e-3
ROT_STATE_TOL = 1.0e-2
INV_WALL_SEG_TOL = 1.0e-6


class RotationalSolveError(ValueError):
    """Raised when the rotational MOC solve cannot form a physical point."""


class NoWallIntersectionError(ValueError):
    """Raised when no valid wall/root intersection can be found."""



def pm_fxn(M, gamma):
    return (
        np.sqrt((gamma + 1) / (gamma - 1)) *
        np.arctan(np.sqrt((gamma - 1)*(M**2 - 1)/(gamma + 1))) -
        np.arctan(np.sqrt(M**2 - 1))
        )


def pm_diff(M, nu, gamma):
    return (nu - pm_fxn(M, gamma))

def pm_mach_solver(M, gamma, delta):
    nu1 = pm_fxn(M, gamma)
    nu2 = nu1 + np.abs(delta)
    nu_max = 0.5 * np.pi * (np.sqrt((gamma + 1) / (gamma - 1)) - 1.0)
    if nu2 >= nu_max:
        msg = (
            "Requested Prandtl-Meyer turn exceeds the maximum expansion angle: "
            f"nu2={nu2:.6g}, nu_max={nu_max:.6g}."
        )
        raise RotationalSolveError(msg)
    M_hi = max(2.0 * M, M + 1.0)
    for _ in range(64):
        if pm_diff(M_hi, nu2, gamma) <= 0.0:
            break
        M_hi *= 2.0
    else:
        msg = "Could not bracket Prandtl-Meyer downstream Mach number."
        raise RotationalSolveError(msg)
    M2 = root_scalar(pm_diff, bracket=(M, M_hi), args=(nu2, gamma)).root
    return M2

def pm_solver(M, gamma, d):
    """
    Wrapper function for Prandtl-Meyer fan.
    Outputs: post-fan gas object, post-fan Mach number, fan angles v1 and v2 relative to horizontal
    """
    M2 = pm_mach_solver(M, gamma, np.abs(d))
    P2_P1 = (H(M, M2, gamma)) ** (gamma / (gamma - 1))
    T2_T1 = (H(M, M2, gamma))
    return M2,  T2_T1, P2_P1

def H(M_in, M_out, gamma): #Returns: post-expansion temperature ratio
    #Inputs: gamma, incoming Mach number, outgoing Mach number
    return (
        (1 + ((gamma - 1) / 2) * M_in**2) / (1 + ((gamma - 1) / 2) * M_out**2)
    )

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


def _raise_unphysical_rot_state(pt: np.ndarray, solve_name: str) -> None:
    msg = (
        f"{solve_name} produced an unphysical state: "
        f"V={pt[2]:.6g}, theta={pt[3]:.6g}, p={pt[4]:.6g}, rho={pt[5]:.6g}."
    )
    raise RotationalSolveError(msg)


def _invalid_state(V: float, p: float, rho: float) -> bool:
    return bool(
        not math.isfinite(V)
        or not math.isfinite(p)
        or not math.isfinite(rho)
        or V <= 0.0
        or rho <= 0.0
        or p <= 0.0
    )


def _pm_wall_fan_state(
    V_in: float,
    theta_in: float,
    p_in: float,
    rho_in: float,
    theta_out: float,
    gamma: float,
    sign: int,
) -> tuple[float, float, float] | None:
    dtheta = theta_out - theta_in
    if not math.isfinite(dtheta) or sign * dtheta <= 0.0:
        return None
    if _invalid_state(V_in, p_in, rho_in):
        return None

    M_in2 = rho_in * V_in * V_in / (gamma * p_in)
    if not math.isfinite(M_in2) or M_in2 <= 1.0:
        return None

    try:
        M_out, T2_T1, P2_P1 = pm_solver(math.sqrt(M_in2), gamma, abs(dtheta))
    except (OverflowError, ValueError, RotationalSolveError):
        return None
    p_out = p_in * P2_P1
    rho_out = rho_in * P2_P1 / T2_T1
    V_out = M_out * math.sqrt(gamma * p_out / rho_out)
    if _invalid_state(V_out, p_out, rho_out):
        return None
    return V_out, p_out, rho_out


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
            y3 = y4 + L0 * (x3 - x4)
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


def _line_segment_intersection_from_slope(
    pt_a: np.ndarray,
    pt_b: np.ndarray,
    x_fix: float,
    y_fix: float,
    slope: float,
    tol: float,
) -> tuple[float, float, float]:
    d13 = pt_b[:2] - pt_a[:2]
    if math.isinf(slope):
        d24 = np.array([0.0, 1.0])
    else:
        d24 = np.array([1.0, slope])

    mat = np.column_stack((d13, -d24))
    rhs = np.array([x_fix, y_fix]) - pt_a[:2]
    det = np.linalg.det(mat)
    if abs(det) <= tol:
        msg = "Inverse wall point characteristic lines are nearly parallel."
        raise RotationalSolveError(msg)

    r13, _r24 = np.linalg.solve(mat, rhs)
    if r13 < -INV_WALL_SEG_TOL or r13 > 1.0 + INV_WALL_SEG_TOL:
        msg = (
            "Inverse wall interpolation point left the known characteristic segment: "
            f"r={r13:.6g}."
        )
        raise RotationalSolveError(msg)

    r13 = float(np.clip(r13, 0.0, 1.0))
    x2, y2 = pt_a[:2] + r13 * d13
    return float(x2), float(y2), r13


def _inverse_wall_interp_point(
    pt_char: np.ndarray,
    pt_wall: np.ndarray,
    x4: float,
    y4: float,
    V4: float,
    theta4: float,
    p4: float,
    rho4: float,
    gamma: float,
    sign: int,
    delta: float,
    max_iters: int,
    tol: float,
) -> np.ndarray:
    pt2 = pt_wall.copy()
    x2_prev = math.nan

    for iter_idx in range(max(0, int(max_iters)) + 1):
        pt24_avg = np.array(
            [
                0.5 * (pt2[0] + x4),
                0.5 * (pt2[1] + y4),
                0.5 * (pt2[2] + V4),
                0.5 * (pt2[3] + theta4),
                0.5 * (pt2[4] + p4),
                0.5 * (pt2[5] + rho4),
            ]
        )
        L, _, _, _ = _get_lqrs_values(
            pt24_avg,
            gamma,
            sign=sign,
            delta=delta,
        )
        x2, y2, _r13 = _line_segment_intersection_from_slope(
            pt_char,
            pt_wall,
            x4,
            y4,
            L,
            tol,
        )
        pt2 = interp_pts(pt_char, pt_wall, (x2, y2), clip=True)

        if iter_idx == 0:
            V4 = pt2[2]
            p4 = pt2[4]
            rho4 = pt2[5]

        if iter_idx > 0 and abs(x2 - x2_prev) <= max(float(tol), ROT_XY_TOL):
            return pt2
        x2_prev = x2

    msg = "Inverse wall interpolation point did not converge."
    raise RotationalSolveError(msg)


def _wall_angle_at_point(
    wall: PiecewiseLinearCurve,
    x4: float,
    y4: float,
    pt_wall: np.ndarray,
) -> float:
    wall_angle = wall.get_angle(x4)
    if wall_angle is not None:
        return wall_angle
    return math.atan2(y4 - pt_wall[1], x4 - pt_wall[0])


def _bisect_wall_target(
    pt_wall: np.ndarray,
    x4: float,
    y4: float,
    wall: PiecewiseLinearCurve,
) -> tuple[float, float]:
    x_mid = 0.5 * (pt_wall[0] + x4)
    y_mid = wall.get_y(x_mid)
    if y_mid is None:
        y_mid = 0.5 * (pt_wall[1] + y4)
    return x_mid, y_mid


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
    x_char, y_char, V_char, theta_char, p_char, rho_char = pt_char #eq of pt 2
    _x_wall, _y_wall, V_wall, theta_wall, p_wall, rho_wall = pt_wall #eq of pt 3 in z&h

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

        # if _invalid_state(V4, p4, rho4):
        #     pm_state = _pm_wall_fan_state(
        #         V_wall,
        #         theta_wall,
        #         p_wall,
        #         rho_wall,
        #         theta4,
        #         gamma,
        #         sign,
        #     )
        #     if pm_state is None:
        #         _raise_unphysical_rot_state(
        #             np.array([x4, y4, V4, theta4, p4, rho4]),
        #             "Wall point",
        #         )
        #     V4, p4, rho4 = pm_state


        p_avg = 0.5 * (p_wall + p4)
        rho_avg = 0.5 * (rho_wall + rho4)
        V_avg = 0.5 * (V_wall + V4)

        # if _invalid_state(V_avg, p_avg, rho_avg):
        #     pm_state = _pm_wall_fan_state(
        #         V_wall,
        #         theta_wall,
        #         p_wall,
        #         rho_wall,
        #         theta4,
        #         gamma,
        #         sign,
        #     )
        #     if pm_state is None:
        #         _raise_unphysical_rot_state(
        #             np.array([x4, y4, V4, theta4, p4, rho4]),
        #             "Wall point",
        #         )
        #     V4, p4, rho4 = pm_state
        #     p_avg = 0.5 * (p_wall + p4)
        #     rho_avg = 0.5 * (rho_wall + rho4)
        #     V_avg = 0.5 * (V_wall + V4)

        # if _invalid_state(V_avg, p_avg, rho_avg):
        #     _raise_unphysical_rot_state(
        #         np.array([x4, y4, V4, theta4, p4, rho4]),
        #         "Wall point",
        #     )

        R0 = rho_avg * V_avg
        a02 = gamma * p_avg / rho_avg
        T01 = R0 * V_wall + p_wall
        T02 = p_wall - a02 * rho_wall

        V4 = (T01 - p4) / R0
        rho4 = (p4 - T02) / a02

        pt_out = np.array([x4, y4, V4, theta4, p4, rho4])
        # if _invalid_state(V4, p4, rho4):
        #     _raise_unphysical_rot_state(pt_out, "Wall point")
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


def inv_wall_point_rot(
    pt_char: np.ndarray,
    pt_wall: np.ndarray,
    wall: PiecewiseLinearCurve,
    family: str,
    gamma: float,
    x4: float,
    y4: float,
    max_iters: int = 10,
    tol: float = 1e-6,
    delta: float = 0.0,
    max_adjustments: int = 8,
) -> np.ndarray:
    x_char, y_char, V_char, theta_char, p_char, rho_char = pt_char
    _x_wall, _y_wall, V_wall, theta_wall, p_wall, rho_wall = pt_wall

    sign = _family_sign(family)
    pt_char = np.array([x_char, y_char, V_char, theta_char, p_char, rho_char])
    pt_wall = np.array([_x_wall, _y_wall, V_wall, theta_wall, p_wall, rho_wall])

    x4_try = x4
    y4_try = y4
    last_err: RotationalSolveError | None = None

    for _adjust_idx in range(max(0, int(max_adjustments)) + 1):
        theta4 = _wall_angle_at_point(wall, x4_try, y4_try, pt_wall)
        V4 = 0.5 * (V_char + V_wall)
        p4 = 0.5 * (p_char + p_wall)
        rho4 = 0.5 * (rho_char + rho_wall)
        pt_out_prev = np.full(6, np.nan)

        try:
            for iter_idx in range(max(0, int(max_iters)) + 1):
                pt2 = _inverse_wall_interp_point(
                    pt_char,
                    pt_wall,
                    x4_try,
                    y4_try,
                    V4,
                    theta4,
                    p4,
                    rho4,
                    gamma,
                    sign,
                    delta,
                    max_iters,
                    tol,
                )

                pt24_avg = np.array(
                    [
                        0.5 * (pt2[0] + x4_try),
                        0.5 * (pt2[1] + y4_try),
                        0.5 * (pt2[2] + V4),
                        0.5 * (pt2[3] + theta4),
                        0.5 * (pt2[4] + p4),
                        0.5 * (pt2[5] + rho4),
                    ]
                )
                _L, Q, _, S = _get_lqrs_values(
                    pt24_avg,
                    gamma,
                    sign=sign,
                    delta=delta,
                )

                if sign > 0:
                    T_char = -S * (x4_try - pt2[0]) + Q * pt2[4] + pt2[3]
                    p4_new = (T_char - theta4) / Q
                else:
                    T_char = -S * (x4_try - pt2[0]) + Q * pt2[4] - pt2[3]
                    p4_new = (T_char + theta4) / Q

                used_pm_fan = False
                if iter_idx == 0:
                    V4_stream = V_wall
                    p4_stream = p_wall
                    rho4_stream = rho_wall
                else:
                    V4_stream = V4
                    p4_stream = p4
                    rho4_stream = rho4

                if not math.isfinite(p4_new) or p4_new <= 0.0:
                    pm_state = _pm_wall_fan_state(
                        V4_stream,
                        theta_wall,
                        p4_stream,
                        rho4_stream,
                        theta4,
                        gamma,
                        sign,
                    )
                    if pm_state is None:
                        _raise_unphysical_rot_state(
                            np.array([x4_try, y4_try, V4, theta4, p4_new, rho4]),
                            "Inverse wall point",
                        )
                    V4, p4, rho4 = pm_state
                    used_pm_fan = True

                p_avg = 0.5 * (p_wall + p4_stream)
                rho_avg = 0.5 * (rho_wall + rho4_stream)
                V_avg = 0.5 * (V_wall + V4_stream)
                if (not used_pm_fan) and _invalid_state(V_avg, p_avg, rho_avg):
                    pm_state = _pm_wall_fan_state(
                        V4_stream,
                        theta_wall,
                        p4_stream,
                        rho4_stream,
                        theta4,
                        gamma,
                        sign,
                    )
                    if pm_state is None:
                        _raise_unphysical_rot_state(
                            np.array([x4_try, y4_try, V4, theta4, p4, rho4]),
                            "Inverse wall point",
                        )
                    V4, p4, rho4 = pm_state
                    used_pm_fan = True

                if not used_pm_fan:
                    R0 = rho_avg * V_avg
                    a02 = gamma * p_avg / rho_avg
                    T01 = R0 * V_wall + p_wall
                    T02 = p_wall - a02 * rho_wall

                    V4 = (T01 - p4_new) / R0
                    p4 = p4_new
                    rho4 = (p4 - T02) / a02

                pt_out = np.array([x4_try, y4_try, V4, theta4, p4, rho4])
                if _invalid_state(V4, p4, rho4):
                    _raise_unphysical_rot_state(pt_out, "Inverse wall point")

                if _point_values_converged(
                    pt_out,
                    pt_out_prev,
                    xy_tol=max(float(tol), ROT_XY_TOL),
                    state_tol=max(float(tol), ROT_STATE_TOL),
                ):
                    return pt_out

                pt_out_prev = pt_out

            return pt_out_prev

        except RotationalSolveError as err:
            last_err = err
            x4_try, y4_try = _bisect_wall_target(pt_wall, x4_try, y4_try, wall)

    if last_err is not None:
        raise last_err
    msg = "Inverse wall point failed before starting an iteration."
    raise RotationalSolveError(msg)


field_point = field_point_rot
wall_point_rot_inverse = inv_wall_point_rot


__all__ = [
    "NoWallIntersectionError",
    "RotationalSolveError",
    "field_point",
    "field_point_rot",
    "get_qrs",
    "inv_wall_point_rot",
    "wall_point_rot",
    "wall_point_rot_inverse",
]
