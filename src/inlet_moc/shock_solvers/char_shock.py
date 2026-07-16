from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from inlet_moc.planar_inlet import PiecewiseLinearCurve
from inlet_moc.rotational_solvers import field_point_rot, wall_point_rot
from inlet_moc.utils_moc import (
    get_intersection,
    get_tang,
    get_tang_from_pts,
    interp_pts,
    intersect_line_polyline,
)


class DetachedShockError(ValueError):
    pass


class SubsonicFlowError(ValueError):
    pass


class InsufficientCharPtsLError(ValueError):
    pass


def obl_shock_angle(M_in: float, gamma: float, d: float, a: int = 1) -> float:
    d = abs(float(d))
    M_in = float(M_in)
    gamma = float(gamma)
    if not math.isfinite(M_in) or (M_in <= 1.0):
        return math.nan

    gp = gamma + 1.0
    gm = gamma - 1.0
    M2 = M_in * M_in
    tan_d = math.tan(d)
    tand2 = tan_d * tan_d
    discr = (M2 - 1.0) ** 2 - 3.0 * (1.0 + (0.5 * gm) * M2) * (
        1.0 + (0.5 * gp) * M2
    ) * tand2
    if discr < 0.0:
        return math.nan

    lamb = math.sqrt(discr)

    chi = (1.0 / (lamb**3)) * (
        (M2 - 1.0) ** 3
        - 9.0
        * (1.0 + (0.5 * gm) * M2)
        * (1.0 + (0.5 * gm) * M2 + (0.25 * gp) * M2 * M2)
        * tand2
    )
    chi = min(max(chi, -1.0), 1.0)
    numerator = (
        M2
        - 1.0
        + 2.0 * lamb * math.cos((4.0 * math.pi * int(a) + math.acos(chi)) / 3.0)
    )
    denominator = 3.0 * (1.0 + (0.5 * gm) * M2) * tan_d
    if not math.isfinite(denominator) or math.isclose(denominator, 0.0):
        return math.nan
    return math.atan(numerator / denominator)


def _obl_shock_downstream_values(
    M_in: float,
    gamma: float,
    beta: float,
) -> tuple[float, float, float]:
    gp = gamma + 1.0
    gm = gamma - 1.0
    sin_beta = math.sin(beta)
    sin_beta2 = sin_beta * sin_beta
    M2 = M_in * M_in
    Mn1_2 = M2 * sin_beta2

    M_out_num = _sqrt_or_nan(
        1.0 + gm * Mn1_2 + (((0.5 * gp) ** 2 - gamma * sin_beta2) * M2 * M2 * sin_beta2)
    )
    M_out_den = _sqrt_or_nan(gamma * Mn1_2 - 0.5 * gm) * _sqrt_or_nan(
        0.5 * gm * Mn1_2 + 1.0
    )

    if not math.isfinite(M_out_den) or math.isclose(M_out_den, 0.0):
        return math.nan, math.nan, math.nan

    M_out = M_out_num / M_out_den

    p2p1 = (2.0 * gamma * Mn1_2 / gp) - (gm / gp)
    rho2rho1_den = gm * Mn1_2 + 2.0

    if math.isclose(rho2rho1_den, 0.0):
        return M_out, p2p1, math.nan
    rho2rho1 = gp * Mn1_2 / rho2rho1_den
    return M_out, p2p1, rho2rho1


def _sqrt_or_nan(value: float) -> float:
    if value < 0.0:
        return math.nan
    return math.sqrt(value)


def obl_shock_post_state(
    M_in: float,
    gamma: float,
    d: float,
    a: int = 1,
) -> tuple[float, float, float, float]:
    beta = obl_shock_angle(M_in, gamma, d, a=a)
    if not math.isfinite(beta):
        return math.nan, math.nan, math.nan, math.nan

    try:
        M_out, p2p1, rho2rho1 = _obl_shock_downstream_values(
            M_in,
            gamma,
            beta,
        )
    except ValueError:
        return beta, math.nan, math.nan, math.nan
    return beta, M_out, p2p1, rho2rho1


@dataclass(frozen=True)
class ShockPoint:
    pt_pre: np.ndarray  # Pre-shock state. (6,) [x, y, V, theta, p, rho]
    pt_post: np.ndarray  # Post-shock state. (6,) [x, y, V, theta, p, rho]
    beta: float  # Shock angle, rad
    gamma: float

    def get_shock_tang(self):
        return get_tang(self.pt_pre[:2], np.tan(self.beta))


def verify_field_soln(
    shock_pair: ShockPoint,
    p3_pt: np.ndarray | None,
    tol: float = 1e-6,
) -> bool:
    if not (np.all(np.isfinite(shock_pair.pt_post)) and np.all(np.isfinite(p3_pt))):
        return False
    return bool(p3_pt[0] >= (shock_pair.pt_post[0] - tol))


def _deflection_converged(delta_new: float, delta_prev: float, tol: float) -> bool:
    return bool(
        np.isfinite(delta_new)
        and np.isfinite(delta_prev)
        and (abs(delta_new - delta_prev) <= tol)
    )


def _position_converged(pt_pre: np.ndarray, pt_post: np.ndarray, xy_tol: float) -> bool:
    return bool(
        np.all(np.isfinite(pt_pre[:2]))
        and np.all(np.isfinite(pt_post[:2]))
        and (np.linalg.norm(pt_post[:2] - pt_pre[:2]) <= xy_tol)
    )


def _shock_point_converged(
    delta_new: float,
    delta_prev: float,
    pt_pre: np.ndarray,
    pt_post: np.ndarray,
    tol: float,
    xy_tol: float,
) -> bool:
    return _deflection_converged(delta_new, delta_prev, tol) and _position_converged(
        pt_pre,
        pt_post,
        xy_tol,
    )


def _mach_from_point(pt: np.ndarray, gamma: float) -> float:
    _, _, V, _, p, rho = pt
    return V / math.sqrt(gamma * p / rho)


def _alpha_from_point(pt: np.ndarray, gamma: float) -> float:
    M = _mach_from_point(pt, gamma)
    if not math.isfinite(M) or (M <= 1.0):
        msg = (
            "Subsonic flow encountered while computing characteristic angle: "
            f"M={M:.6g}."
        )
        raise SubsonicFlowError(msg)
    return math.asin(1.0 / M)


def _lambda_from_point(pt: np.ndarray, gamma: float, family: str) -> float:
    alpha = _alpha_from_point(pt, gamma)
    sign = 1.0 if family == "cplus" else -1.0
    return pt[3] + sign * alpha


def point_thru_shock(pt: np.ndarray, gamma: float, delta: float, normal: float):
    x, y, _, theta, p, rho = pt
    M = _mach_from_point(pt, gamma)
    if not math.isfinite(M) or (M <= 1.0):
        msg = f"Subsonic flow encountered upstream of shock solve: M={M:.6g}."
        raise SubsonicFlowError(msg)
    beta, M_out, p2p1, rho2rho1 = obl_shock_post_state(
        M,
        gamma,
        abs(float(delta)),
    )
    if not math.isfinite(beta):
        msg = (
            "Subsonic flow encountered in shock solve: no attached oblique shock "
            "solution exists for the requested state."
        )
        raise SubsonicFlowError(msg)

    if not math.isfinite(M_out) or (M_out <= 1.0):
        msg = f"Subsonic flow encountered downstream of shock solve: M={M_out:.6g}."
        raise SubsonicFlowError(msg)

    theta_out = theta + delta
    p_out = p * p2p1
    rho_out = rho * rho2rho1
    V_out = M_out * math.sqrt(gamma * p_out / rho_out)
    pt_out = np.array([x, y, V_out, theta_out, p_out, rho_out])

    beta_geom = (normal * beta) + theta
    return pt_out, beta_geom


def _shock_delta(shock_pair: ShockPoint) -> float:
    return shock_pair.pt_post[3] - shock_pair.pt_pre[3]


def shock_origin(
    pt_pre: np.ndarray,
    wall: PiecewiseLinearCurve,
    gamma: float,
    tol: float = 1e-6,
):
    """
    Bookkeeping function to get post-shock state immediately at the solved delta
    """
    n_wall = wall.normal_sign

    wall_angle = wall.get_angle(pt_pre[0])
    theta = pt_pre[3]

    # Deflection at shock origin:
    delta = wall_angle - theta
    if delta * n_wall < 0:
        msg = "No shock here; expansion fan detected. Not implemented yet."
        raise ValueError(msg)
    if abs(delta) < tol:
        return None
    try:
        pt_post, beta = point_thru_shock(
            pt_pre,
            gamma,
            delta,
            n_wall,
        )
    except SubsonicFlowError as err:
        msg = "Detached shock not implemented yet. No attached oblique shock soln here."
        raise DetachedShockError(msg) from err

    return ShockPoint(pt_pre, pt_post, beta, gamma)


def shock_field(
    pt_prior: ShockPoint,
    family: str,
    char_pts_L: np.ndarray,
    idx_L,
    char_pts_R: np.ndarray,
    idx_R,
    gamma: float,
    n_wall,
    max_iters=5,
    tol=1e-6,
    idx_R_min: int = 0,
):
    xy_tol = 10 * tol
    idx_R = np.asarray(idx_R, dtype=int)
    idx_R_keep = idx_R >= int(idx_R_min)
    char_pts_R = np.asarray(char_pts_R, dtype=float)[idx_R_keep]
    idx_R = idx_R[idx_R_keep]

    pt_s_pre = pt_prior.pt_pre  # pre-shock state at prior iter (in marching loop)
    pt_s_post = pt_prior.pt_post  # post-shock state at prior iter

    delta_i = _shock_delta(pt_prior)
    beta_prior = pt_prior.beta
    beta_ip1 = beta_prior

    pt_pair = None
    pt_3pr = None
    best_valid = None
    best_valid_resid = math.inf
    emergency_candidate = None

    iter_limit = max(1, int(max_iters))
    for local_idx in range(char_pts_R.shape[0]):
        pt_mesh = char_pts_R[local_idx]
        idx_R_m = int(idx_R[local_idx])

        delta_iter = delta_i
        eqn_sa = get_tang_from_pts(pt_mesh, pt_s_post)
        shock_converged = False

        for _ in range(iter_limit):
            # Find intersection of prior shock point and LHS char points
            eqn_shock = get_tang(pt_s_pre, np.tan(beta_prior))
            hit = intersect_line_polyline(eqn_shock, char_pts_L, tol / 10)
            if hit is None:
                msg = "Insufficient res. for char_pts_L and shock!"
                raise InsufficientCharPtsLError(msg)
            pt_sc_pre, idx_loc_L = hit
            idx_L_m = int(idx_L[idx_loc_L])

            theta_sc = pt_sc_pre[3]

            # Position unchanged; get post-shock state/angles
            pt_sc_post, beta_ip1 = point_thru_shock(
                pt_sc_pre,
                gamma,
                delta_iter,
                n_wall,
            )

            lamd = _lambda_from_point(pt_sc_post, gamma, family)

            # Get equation of char. from post-shock to constructed char.
            eqn_1p = get_tang(pt_sc_post, np.tan(lamd))

            if family == "cplus":
                pt_3pr = field_point_rot(
                    pt_mesh, pt_sc_post, gamma, max_iters=max_iters, tol=tol
                )
            else:
                pt_3pr = field_point_rot(
                    pt_sc_post, pt_mesh, gamma, max_iters=max_iters, tol=tol
                )

            xy_ref = get_intersection(eqn_sa, eqn_1p)
            pt_ref = interp_pts(pt_s_post, pt_mesh, xy_ref)

            if family == "cplus":
                pt_sc_recomp = field_point_rot(
                    pt_ref, pt_3pr, gamma, max_iters=max_iters, tol=tol
                )
            else:
                pt_sc_recomp = field_point_rot(
                    pt_3pr, pt_ref, gamma, max_iters=max_iters, tol=tol
                )

            delta_ip1 = pt_sc_recomp[3] - theta_sc

            if _shock_point_converged(
                delta_ip1, delta_iter, pt_sc_pre, pt_sc_recomp, tol, xy_tol
            ):
                delta_iter = delta_ip1
                shock_converged = True
                break

            delta_iter = delta_ip1
        pt_pair = ShockPoint(
            pt_pre=pt_sc_pre,
            pt_post=pt_sc_recomp,
            beta=beta_ip1,
            gamma=gamma,
        )
        solution_valid = verify_field_soln(pt_pair, pt_3pr, tol)
        if shock_converged and solution_valid:
            return pt_pair, idx_L_m, pt_3pr, int(idx_R_m)

        if (
            best_valid is None
            and np.all(np.isfinite(pt_pair.pt_post))
            and np.all(np.isfinite(pt_3pr))
        ):
            emergency_candidate = (pt_pair, idx_L_m, pt_3pr, int(idx_R_m))

        if solution_valid:
            resid = np.linalg.norm(pt_pair.pt_post[:2] - pt_pair.pt_pre[:2])
            if resid < best_valid_resid:
                best_valid_resid = resid
                best_valid = (pt_pair, idx_L_m, pt_3pr, int(idx_R_m))

        has_more_rhs = local_idx < (len(char_pts_R) - 1)
        if has_more_rhs and ((not shock_converged) or (not solution_valid)):
            continue

        rhs_exhausted = local_idx == (len(char_pts_R) - 1)
        if (rhs_exhausted or len(char_pts_R) == 1) and solution_valid:
            return best_valid

    if best_valid is None and emergency_candidate is not None:
        pt_pair, idx_L_m, pt_3pr, idx_R_m = emergency_candidate
        pt_3pr_flip = np.array(pt_3pr, dtype=float, copy=True)
        pt_3pr_flip[:2] = 2.0 * pt_pair.pt_post[:2] - pt_3pr_flip[:2]
        if verify_field_soln(pt_pair, pt_3pr_flip, tol):
            return pt_pair, idx_L_m, pt_3pr_flip, int(idx_R_m)

    msg = "No valid downstream field mesh point found in provided stencil."
    raise RuntimeError(msg)


def shock_to_wall(
    pt_prior: ShockPoint,
    family: str,
    pt1: np.ndarray,  # char_pts_L
    char_pts_R: np.ndarray,
    idx_R,
    wall: PiecewiseLinearCurve,
    gamma: float,
    max_iters: int = 5,
    tol: float = 1e-6,
    idx_R_min: int = 0,
):
    pt_s_pre = pt_prior.pt_pre
    x_s, y_s = pt_s_pre[:2]
    beta = pt_prior.beta
    delta = _shock_delta(pt_prior)

    n_wall = wall.normal_sign
    pt1 = np.asarray(pt1, dtype=float)
    if pt1.ndim > 1:
        pt1 = pt1[0]
    idx_R = np.asarray(idx_R, dtype=int)
    idx_R_keep = idx_R >= int(idx_R_min)
    char_pts_R = np.asarray(char_pts_R, dtype=float)[idx_R_keep]
    idx_R = idx_R[idx_R_keep]

    pt3 = wall_point_rot(
        pt_s_pre,
        pt1,
        wall,
        family,
        gamma,
        max_iters=max_iters,
        tol=tol,
    )

    def F(x):
        return wall.get_y(x) - (y_s + np.tan(beta) * (x - x_s))

    x_sw = brentq(F, wall.x_min, wall.x_max)
    y_sw = wall.get_y(x_sw)

    pt4_pre = interp_pts(pt1, pt3, (x_sw, y_sw))
    n_shock = n_wall * -1
    pt4_post, _ = point_thru_shock(
        pt4_pre,
        gamma,
        delta,
        n_shock,
    )
    pt_pair = ShockPoint(pt4_pre, pt4_post, beta, gamma)

    for local_idx in range(char_pts_R.shape[0]):
        pt_a = char_pts_R[local_idx]
        idx_R_m = int(idx_R[local_idx])

        if family == "cplus":
            pt_mesh = field_point_rot(
                pt_a, pt4_post, gamma, max_iters=max_iters, tol=tol
            )
        else:
            pt_mesh = field_point_rot(
                pt4_post, pt_a, gamma, max_iters=max_iters, tol=tol
            )

        solution_valid = verify_field_soln(pt_pair, pt_mesh, tol)
        if solution_valid:
            return pt_pair, pt_mesh, int(idx_R_m)

    msg = "No valid downstream wall mesh point found in provided stencil."
    raise RuntimeError(msg)


def shock_from_wall(
    shock_origin: ShockPoint,
    family: str,
    char_pts_L: np.ndarray,
    wall: PiecewiseLinearCurve,
    gamma: float,
    max_iters: int = 5,
    tol: float = 1e-6,
):
    xy_tol = 10 * tol

    pt_o_pre = shock_origin.pt_pre
    delta_i = _shock_delta(shock_origin)
    beta_origin = shock_origin.beta
    beta_ip1 = beta_origin

    n_wall = wall.normal_sign
    pt_wall_anchor = shock_origin.pt_post

    for _ in range(max(1, int(max_iters))):
        eqn_shock = get_tang(pt_o_pre, np.tan(beta_origin))
        hit = intersect_line_polyline(eqn_shock, char_pts_L, tol)
        if hit is None:
            msg = "Insufficient res. for char_pts_L and shock!"
            raise InsufficientCharPtsLError(msg)
        pt_sc_pre, _ = hit
        theta_sc = pt_sc_pre[3]
        pt_sc_post, beta_ip1 = point_thru_shock(
            pt_sc_pre,
            gamma,
            delta_i,
            n_wall,
        )

        if family == "cplus":
            # pt_wall_anchor is the wall-streamline anchor, not a second C+ point.
            pt_pref = wall_point_rot(
                pt_sc_post,
                pt_wall_anchor,
                wall,
                "cplus",
                gamma,
                max_iters=max_iters,
                tol=tol,
            )
            pt_3pr = wall_point_rot(
                pt_sc_post,
                pt_wall_anchor,
                wall,
                "cminus",
                gamma,
                max_iters=max_iters,
                tol=tol,
            )
            pt_sc_recomp = field_point_rot(
                pt_pref, pt_3pr, gamma, max_iters=max_iters, tol=tol
            )
        else:
            # pt_wall_anchor is the wall-streamline anchor, not a second C- point.
            pt_pref = wall_point_rot(
                pt_sc_post,
                pt_wall_anchor,
                wall,
                "cminus",
                gamma,
                max_iters=max_iters,
                tol=tol,
            )
            pt_3pr = wall_point_rot(
                pt_sc_post,
                pt_wall_anchor,
                wall,
                "cplus",
                gamma,
                max_iters=max_iters,
                tol=tol,
            )
            pt_sc_recomp = field_point_rot(
                pt_3pr, pt_pref, gamma, max_iters=max_iters, tol=tol
            )

        delta_ip1 = pt_sc_recomp[3] - theta_sc

        if _shock_point_converged(
            delta_ip1, delta_i, pt_sc_pre, pt_sc_recomp, tol, xy_tol
        ):
            delta_i = delta_ip1
            break

        delta_i = delta_ip1

    shock_pair = ShockPoint(
        pt_pre=pt_sc_pre,
        pt_post=pt_sc_recomp,
        beta=beta_ip1,
        gamma=gamma,
    )

    return shock_pair, pt_3pr
