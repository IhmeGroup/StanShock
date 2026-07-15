from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import inlet_moc.utils_moc
from inlet_moc.char_net import CharNet
from inlet_moc.flow_physics import FlowCell, static_state_from_mach
from inlet_moc.planar_inlet import PlanarInlet, get_opposite_wall
from inlet_moc.shock_solvers.char_shock import ShockPoint, shock_origin


@dataclass(frozen=True)
class InitialDomain:
    net: CharNet
    leading_shock: ShockPoint
    cells: list[FlowCell]
    x_start: float


def build_freestream_cell(inlet: PlanarInlet, leading_shock: ShockPoint):
    xy_bot = leading_shock.pt_pre[:2]
    shock_tang = leading_shock.get_shock_tang()
    wall_opp = get_opposite_wall(xy_bot, inlet)

    xy_top = np.array([xy_bot[0], wall_opp.y_max])
    top_tang = inlet_moc.utils_moc.get_tang(xy_top, 0)
    xy_r = inlet_moc.utils_moc.get_intersection(shock_tang, top_tang)
    verts = inlet_moc.utils_moc.order_cell_vertices(np.vstack((xy_bot, xy_top, xy_r)))
    cell = FlowCell(
        verts=verts,
        V=leading_shock.pt_pre[2],
        theta=leading_shock.pt_pre[3],
        p=leading_shock.pt_pre[4],
        rho=leading_shock.pt_pre[5],
    )
    return cell, xy_r


def build_idl_cell(leading_shock: ShockPoint, xy_r: np.ndarray, idl_pts: np.ndarray):
    verts = inlet_moc.utils_moc.order_cell_vertices(
        np.vstack((leading_shock.pt_pre[:2], xy_r, idl_pts[-1], idl_pts[0]))
    )
    return FlowCell(
        verts=verts,
        V=leading_shock.pt_post[2],
        theta=leading_shock.pt_post[3],
        p=leading_shock.pt_post[4],
        rho=leading_shock.pt_post[5],
    )


def initialize_domain(
    inlet: PlanarInlet,
    M_init: float,
    theta_init: float,
    T_amb: float,
    p_amb: float,
    N_idl: int,
    gamma: float,
    R: float,
    max_iters: int,
    tol: float,
) -> InitialDomain:
    V0, theta0, p_static0, rho0 = static_state_from_mach(
        M_init,
        theta_init,
        T_amb,
        p_amb,
        gamma,
        R,
    )
    wall_leading_edge, xy_0, n_le = inlet.get_infl0()
    pt0_upstream = np.concatenate((xy_0, [V0, theta0, p_static0, rho0]))

    leading_shock = shock_origin(pt0_upstream, wall_leading_edge, gamma, tol)
    if leading_shock is None:
        msg = "Can't proceed without leading shock!"
        raise RuntimeError(msg)

    cell_init, xy_r = build_freestream_cell(inlet, leading_shock)
    postshock = leading_shock.pt_post
    M_idl = postshock[2] / np.sqrt(gamma * postshock[4] / postshock[5])
    if not np.isfinite(M_idl) or M_idl <= 1.0:
        msg = f"Post-shock IDL state is not supersonic: M={M_idl:.6g}."
        raise RuntimeError(msg)
    theta_idl = postshock[3]
    alpha0 = np.arcsin(1.0 / M_idl)
    V_idl, p_idl, rho_idl = postshock[2], postshock[4], postshock[5]

    idl_slope = np.tan(theta_idl + n_le * alpha0)
    idl_pts = inlet.build_idl_pts(idl_slope, N_idl)
    if idl_pts is None:
        msg = "Couldn't get initial data line from inlet geometry."
        raise RuntimeError(msg)

    net = CharNet(
        N_idl,
        inlet,
        gamma,
        max_iters=max_iters,
        tol=tol,
    )
    net.apply_idl(idl_pts, V_idl, theta_idl, p_idl, rho_idl)

    return InitialDomain(
        net=net,
        leading_shock=leading_shock,
        cells=[cell_init, build_idl_cell(leading_shock, xy_r, idl_pts)],
        x_start=float(xy_0[0]),
    )
