from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import inlet_moc.utils_moc
from inlet_moc.char_shock_solvers import ShockPoint, shock_origin
from inlet_moc.charnet import CharNet
from inlet_moc.flow_physics import FlowCell, RegionState
from inlet_moc.planar_inlet import PlanarInlet, get_opposite_wall


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
        u=leading_shock.pt_pre[2],
        v=leading_shock.pt_pre[3],
        region=leading_shock.reg_pre,
    )
    return cell, xy_r


def build_idl_cell(leading_shock: ShockPoint, xy_r: np.ndarray, idl_pts: np.ndarray):
    verts = inlet_moc.utils_moc.order_cell_vertices(
        np.vstack((leading_shock.pt_pre[:2], xy_r, idl_pts[-1], idl_pts[0]))
    )
    return FlowCell(
        verts=verts,
        u=leading_shock.pt_post[2],
        v=leading_shock.pt_post[3],
        region=leading_shock.reg_post,
    )


def initialize_domain(
    inlet: PlanarInlet,
    region_amb: RegionState,
    M_init: float,
    theta_init: float,
    N_idl: int,
    max_iters: int,
    tol: float,
) -> InitialDomain:
    u0, v0 = region_amb.ref.mach_to_vel(M_init, theta_init)
    wall_leading_edge, xy_0, n_le = inlet.get_infl0()
    pt0_upstream = np.concatenate((xy_0, [u0, v0]))

    leading_shock = shock_origin(pt0_upstream, wall_leading_edge, region_amb, tol)
    if leading_shock is None:
        msg = "Can't proceed without leading shock!"
        raise RuntimeError(msg)

    cell_init, xy_r = build_freestream_cell(inlet, leading_shock)
    region_idl = leading_shock.reg_post
    postshock = leading_shock.pt_post
    theta0, alpha0 = region_idl.get_angles(postshock[2], postshock[3])
    u_idl, v_idl = leading_shock.pt_post[2:]

    idl_slope = np.tan(theta0 + n_le * alpha0)
    idl_pts = inlet.build_idl_pts(idl_slope, N_idl)
    if idl_pts is None:
        msg = "Couldn't get initial data line from inlet geometry."
        raise RuntimeError(msg)

    net = CharNet(
        N_idl,
        inlet,
        region_idl,
        max_iters=max_iters,
        tol=tol,
    )
    net.apply_idl(idl_pts, u_idl, v_idl)

    return InitialDomain(
        net=net,
        leading_shock=leading_shock,
        cells=[cell_init, build_idl_cell(leading_shock, xy_r, idl_pts)],
        x_start=float(xy_0[0]),
    )
