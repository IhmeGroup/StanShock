from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

import inlet_moc.shock_fxns as sfx
from inlet_moc.char_solvers import field_point, wall_point
from inlet_moc.flow_physics import RegionState
from inlet_moc.planar_inlet import PiecewiseLinearCurve
from inlet_moc.utils_moc import (
    get_intersection,
    get_tang,
    get_tang_from_pts,
    get_theta,
    interp_pts,
    intersect_line_polyline,
)


class DetachedShockError(ValueError):
    """Raised when the requested shock is detached and unsupported."""


class SubsonicFlowError(ValueError):
    """Raised when the MOC march encounters sonic or subsonic flow."""


class InsufficientCharPtsLError(ValueError):
    pass


@dataclass(frozen=True)
class ShockPoint:
    pt_pre: np.ndarray      #Pre-shock state. (4,) [x, y, u, v]
    pt_post: np.ndarray     #Post-shock state.
    delta: float            #Deflection angle (rad)
    beta: float             #Shock angle (rad)
    reg_pre: RegionState
    reg_post: RegionState

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


def point_thru_shock(pt: np.ndarray, region: RegionState, delta: float, normal: float):
    u = pt[2]
    v = pt[3]
    M, theta, _, _ = region.get_static_props(u, v)
    if not math.isfinite(M) or (M <= 1.0):
        msg = (
            "Subsonic flow encountered upstream of shock solve: "
            f"M={M:.6g}."
        )
        raise SubsonicFlowError(msg)
    beta, M_out, p02p01 = sfx.obl_shock_post_state(
        M,
        region.ref.gamma,
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
    u_out, v_out = region.ref.mach_to_vel(M_out, theta_out)
    pt_out = np.concatenate((pt[:2], [u_out, v_out]))
    region_out = RegionState(ref=region.ref, p0=region.p0 * p02p01)

    beta_geom = (normal * beta) + theta
    return pt_out, region_out, beta_geom


def shock_origin(
    pt_pre: np.ndarray,
    wall: PiecewiseLinearCurve,
    reg_pre: RegionState,
    tol: float = 1e-6,
):
    """
    Bookkeeping function to get post-shock state immediately at the solved delta
    """
    n_wall = wall.normal_sign

    wall_angle = wall.get_angle(pt_pre[0])
    theta = get_theta(pt_pre)

    # Deflection at shock origin:
    delta = wall_angle - theta
    if delta * n_wall < 0:
        msg = "No shock here; expansion fan detected. Not implemented yet."
        raise ValueError(msg)
    if abs(delta) < tol:
        return None
    try:
        pt_post, reg_post, beta = point_thru_shock(
            pt_pre, reg_pre, delta, n_wall
        )
    except SubsonicFlowError as err:
        msg = (
            "Detached shock not implemented yet. No attached oblique shock soln here."
        )
        raise DetachedShockError(msg) from err

    return ShockPoint(pt_pre,
                            pt_post,
                            delta,
                            beta,
                            reg_pre,
                            reg_post)


def shock_field(
    pt_prior: ShockPoint,
    family: str,
    char_pts_L: np.ndarray, #requires the most care. (4,N) np array from prior net
    idx_L,
    char_pts_R: np.ndarray, #this is intended to replace the single mesh_pt. 
    idx_R,
    a_fxn: Callable,
    n_wall,
    max_iters=5,
    tol=1e-6,
):

    xy_tol = 10 * tol
    reg_pre = pt_prior.reg_pre #upstream region

    pt_s_pre= pt_prior.pt_pre #pre-shock state at prior iter (in marching loop)
    pt_s_post = pt_prior.pt_post #post-shock state at prior iter

    delta_i = pt_prior.delta
    beta_i = pt_prior.beta

    pt_pair = None
    pt_3pr = None

    iter_limit = max(1, int(max_iters))
    for local_idx in range(char_pts_R.shape[0]):
        pt_mesh = char_pts_R[local_idx]
        idx_R_m = int(idx_R[local_idx])

        delta_iter = delta_i
        beta_iter = beta_i
        eqn_sa = get_tang_from_pts(pt_mesh, pt_s_post)
        shock_converged = False

        for _ in range(iter_limit):
            # Find intersection of prior shock point and LHS char points
            eqn_shock = get_tang(pt_s_pre, np.tan(beta_iter)) #this can stay
            hit = intersect_line_polyline(eqn_shock, char_pts_L, tol/10)
            if hit is None:
                raise InsufficientCharPtsLError("Insufficient res. for char_pts_L and shock!")
            pt_sc_pre, idx_loc_L = hit
            idx_L_m = int(idx_L[idx_loc_L])

            theta_sc = get_theta(pt_sc_pre)

            # Position unchanged; get post-shock state/angles
            pt_sc_post, reg_post, beta_ip1 = point_thru_shock(
                pt_sc_pre, reg_pre, delta_iter, n_wall
            )


            if family == "cplus":
                lamd = reg_post.get_lamd_plus(pt_sc_post)
            else:
                lamd = reg_post.get_lamd_minus(pt_sc_post)

            #Get equation of char. from post-shock to constructed char.
            eqn_1p = get_tang(pt_sc_post, np.tan(lamd))

            if family == "cplus":
                pt_3pr = field_point(
                    pt_mesh, pt_sc_post, a_fxn, max_iters=max_iters, tol=tol
                )
            else:
                pt_3pr = field_point(
                    pt_sc_post, pt_mesh, a_fxn, max_iters=max_iters, tol=tol
                )

            xy_ref = get_intersection(eqn_sa, eqn_1p)
            pt_ref = interp_pts(pt_s_post, pt_mesh, xy_ref)

            if family == "cplus":
                pt_sc_recomp = field_point(
                    pt_ref, pt_3pr, a_fxn, max_iters=max_iters, tol=tol
                )
            else:
                pt_sc_recomp = field_point(
                    pt_3pr, pt_ref, a_fxn, max_iters=max_iters, tol=tol
                )

            delta_ip1 = get_theta(pt_sc_recomp) - theta_sc

            if _shock_point_converged(
                delta_ip1, delta_iter, pt_sc_pre, pt_sc_recomp, tol, xy_tol
            ):
                delta_iter = delta_ip1
                beta_iter = beta_ip1
                shock_converged = True
                break

            delta_iter = delta_ip1
            beta_iter = beta_ip1
        pt_pair = ShockPoint(
            pt_pre=pt_sc_pre,
            pt_post=pt_sc_recomp,
            delta=delta_iter,
            beta=beta_iter,
            reg_pre=reg_pre,
            reg_post=reg_post,
        )
        solution_valid = verify_field_soln(pt_pair, pt_3pr, tol)
        if shock_converged and solution_valid:
            return pt_pair, idx_L_m, pt_3pr, int(idx_R_m)

        has_more_rhs = local_idx < (len(char_pts_R) - 1)
        if has_more_rhs and ((not shock_converged) or (not solution_valid)):
            continue

        rhs_exhausted = local_idx == (len(char_pts_R) - 1)
        if (rhs_exhausted or len(char_pts_R) == 1) and pt_3pr is not None:
            pt_3pr_fallback = np.array(pt_3pr, copy=True)
            dx = pt_3pr_fallback[0] - pt_pair.pt_post[0]
            dy = pt_3pr_fallback[1] - pt_pair.pt_post[1]
            pt_3pr_fallback[0] = pt_pair.pt_post[0] - dx
            pt_3pr_fallback[1] = pt_pair.pt_post[1] - dy
            return pt_pair, idx_L_m, pt_3pr_fallback, int(idx_R_m)

    msg = "No valid downstream field mesh point found in provided stencil."
    raise RuntimeError(msg)





def shock_to_wall(
        pt_prior: ShockPoint,
        family: str,
        pt1: np.ndarray, #char_pts_L
        char_pts_R: np.ndarray,
        idx_R,
        wall: PiecewiseLinearCurve,
        a_fxn: Callable,
        max_iters: int = 5,
        tol: float = 1e-6,
):

    reg_pre = pt_prior.reg_pre
    pt_s_pre = pt_prior.pt_pre
    x_s, y_s, _, _ = pt_s_pre
    beta = pt_prior.beta
    delta = pt_prior.delta

    n_wall = wall.normal_sign
    pt1 = np.asarray(pt1, dtype=float)
    if pt1.ndim > 1:
        pt1 = pt1[0]
    idx_R = np.asarray(idx_R, dtype=int)

    if family == "cplus":
        pt3 = wall_point(pt_s_pre, wall, a_fxn, max_iters=max_iters, tol=tol)
    else:
        pt3 = wall_point(wall, pt_s_pre, a_fxn, max_iters=max_iters, tol=tol)

    def F(x):
        return wall.get_y(x) - (y_s + np.tan(beta) * (x - x_s))
    
    x_sw = brentq(F, wall.x_min, wall.x_max)
    y_sw = wall.get_y(x_sw)

    pt4_pre = interp_pts(pt1, pt3, (x_sw, y_sw))
    n_shock = n_wall * -1
    pt4_post, reg_post_4, _ = point_thru_shock(pt4_pre, reg_pre, delta, n_shock)
    pt_pair = ShockPoint(pt4_pre, pt4_post, delta, beta, reg_pre, reg_post_4)

    for local_idx in range(char_pts_R.shape[0]):
        pt_a = char_pts_R[local_idx]
        idx_R_m = int(idx_R[local_idx])

        if family == "cplus":
            pt_mesh = field_point(pt_a, pt4_post, a_fxn, max_iters=max_iters, tol=tol)
        else:
            pt_mesh = field_point(pt4_post, pt_a, a_fxn, max_iters=max_iters, tol=tol)

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
    a_fxn: Callable,
    max_iters: int = 5,
    tol: float = 1e-6,
):

# def shock_from_wall(
#     char_pts_L,
#     wall: PiecewiseLinearCurve,
#     a_fxn: Callable,
#     shock_origin: ShockPoint,
#     family: str = "cminus",
#     max_iters=5,
#     debug_plots=False,
#     debug_plotter: Callable[[], tuple[object, object]] | None = None,
#     tol: float = 1e-6,
# ):
    xy_tol = 10 * tol

    pt_o_pre = shock_origin.pt_pre
    delta_i = shock_origin.delta
    beta_i = shock_origin.beta
    reg_pre = shock_origin.reg_pre

    n_wall = wall.normal_sign

    for _ in range(max(1, int(max_iters))):
        eqn_shock = get_tang(pt_o_pre, np.tan(beta_i))
        hit = intersect_line_polyline(eqn_shock, char_pts_L, tol)
        if hit is None:
            raise InsufficientCharPtsLError("Insufficient res. for char_pts_L and shock!")
        pt_sc_pre, _ = hit
        theta_sc = get_theta(pt_sc_pre)
        pt_sc_post, reg_post, beta_ip1 = point_thru_shock(
            pt_sc_pre, reg_pre, delta_i, n_wall
        )

        if family == "cplus":
            pt_pref = wall_point(pt_sc_post, wall, a_fxn, max_iters=max_iters, tol=tol)
            pt_3pr = wall_point(wall, pt_sc_post, a_fxn, max_iters=max_iters, tol=tol)
            pt_sc_recomp = field_point(
                pt_pref, pt_3pr, a_fxn, max_iters=max_iters, tol=tol
            )
        else:
            pt_3pr = wall_point(pt_sc_post, wall, a_fxn, max_iters=max_iters, tol=tol)
            pt_pref = wall_point(wall, pt_sc_post, a_fxn, max_iters=max_iters, tol=tol)
            pt_sc_recomp = field_point(
                pt_3pr, pt_pref, a_fxn, max_iters=max_iters, tol=tol
            )


        delta_ip1 = get_theta(pt_sc_recomp) - theta_sc

        if _shock_point_converged(
            delta_ip1, delta_i, pt_sc_pre, pt_sc_recomp, tol, xy_tol
        ):
            delta_i = delta_ip1
            beta_i = beta_ip1
            break

        delta_i = delta_ip1
        beta_i = beta_ip1

    shock_pair = ShockPoint(
        pt_pre=pt_sc_pre,
        pt_post=pt_sc_recomp,
        delta=delta_i,
        beta=beta_i,
        reg_pre=reg_pre,
        reg_post=reg_post,
    )

    return shock_pair, pt_3pr



def field_plot(
    char_pts_L: np.ndarray,
    char_pts_R: np.ndarray,
    pt_s: np.ndarray,
    pt_sc: np.ndarray,
    pt_mesh: np.ndarray,
    pt_ref: np.ndarray,
    pt_3pr: np.ndarray,
    pt_sc_recomp: np.ndarray,
    *,
    iteration: int,
    debug_plots: bool = False,
    debug_plotter: Callable[[], tuple[object, object]] | None = None,
):
    if not debug_plots or debug_plotter is None:
        return

    fig, ax = debug_plotter()
    if ax is None:
        return

    def char_label(pt0: np.ndarray, pt1: np.ndarray) -> str:
        if pt0[0] <= pt1[0]:
            pt_left = pt0
            pt_right = pt1
        else:
            pt_left = pt1
            pt_right = pt0
        dx = float(pt_right[0] - pt_left[0])
        dy = float(pt_right[1] - pt_left[1])
        if np.isclose(dx, 0.0):
            return "C+" if dy < 0.0 else "C-"
        slope = dy / dx
        return "C+" if slope < 0.0 else "C-"

    def annotate_segment(
        pt0: np.ndarray,
        pt1: np.ndarray,
        *,
        color: str,
        linestyle: str,
        lw: float,
    ) -> None:
        ax.plot(
            [pt0[0], pt1[0]],
            [pt0[1], pt1[1]],
            color=color,
            linestyle=linestyle,
            lw=lw,
            zorder=14,
        )
        xy_mid = 0.5 * (pt0[:2] + pt1[:2])
        ax.annotate(
            char_label(pt0, pt1),
            (float(xy_mid[0]), float(xy_mid[1])),
            color=color,
            fontsize=6,
            xytext=(2.0, 2.0),
            textcoords="offset points",
            zorder=15,
        )

    ax.plot(
        char_pts_L[:, 0],
        char_pts_L[:, 1],
        color="0.5",
        marker="o",
        markersize=3,
        markerfacecolor="0.5",
        markeredgecolor="0.5",
        lw=1.0,
        zorder=10,
    )
    ax.plot(
        char_pts_R[:, 0],
        char_pts_R[:, 1],
        color="k",
        marker="o",
        markersize=3,
        markerfacecolor="k",
        markeredgecolor="k",
        lw=1.0,
        zorder=11,
    )
    ax.scatter(
        float(pt_ref[0]),
        float(pt_ref[1]),
        s=20,
        c="g",
        label="Ref",
        zorder=100,
    )
    ax.scatter(
        float(pt_s[0]),
        float(pt_s[1]),
        s=20,
        facecolors="none",
        edgecolors="red",
        linewidths=1.0,
        zorder=16,
        label=r"$S_{init}$",
    )

    ax.scatter(
        pt_3pr[0],
        pt_3pr[1],
        s=20,
        c='b',
        label="Mesh")
    annotate_segment(
        pt_mesh,
        pt_3pr,
        color="blue",
        linestyle="--",
        lw=0.5,
    )
    annotate_segment(
        pt_s,
        pt_sc_recomp,
        color="red",
        linestyle="--",
        lw=0.5
    )
    annotate_segment(
        pt_sc_recomp,
        pt_ref,
        color="green",
        linestyle="--",
        lw=0.5,
    )
    annotate_segment(
        pt_sc_recomp,
        pt_3pr,
        color="green",
        linestyle="--",
        lw=0.5,
    )
    ax.scatter(
        float(pt_sc_recomp[0]),
        float(pt_sc_recomp[1]),
        s=36,
        c="red",
        zorder=17,
        label=r"$S_{final}$",
    )
    ax.set_title(f"Field iteration {int(iteration)}", fontsize=8)
    # ax.set_xlim(0.225, 0.235)
    ax.legend(fontsize=6)
    fig.show()
    plt.close(fig)
