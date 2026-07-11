from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from inlet_moc.char_shock_solvers import ShockPoint
from inlet_moc.charnet import CharNet
from inlet_moc.planar_inlet import (
    PiecewiseLinearCurve,
    PlanarInlet,
    get_opposite_wall,
)


@dataclass(frozen=True)
class SolverEvent:
    family: str
    point_idx: tuple[int, int]
    wall_from: PiecewiseLinearCurve
    wall_to: PiecewiseLinearCurve
    delta: float


def _next_event_family(prior_net: CharNet) -> str:
    boundary_mask = prior_net.point_mask("boundary")
    row_max = int(np.count_nonzero(boundary_mask, axis=1).max(initial=0))
    col_max = int(np.count_nonzero(boundary_mask, axis=0).max(initial=0))

    if row_max > col_max:
        return "cminus"
    if col_max > row_max:
        return "cplus"

    msg = "Could not infer next event family from the active boundary geometry."
    raise RuntimeError(msg)


def _wall_event_arrays(
    prior_net: CharNet,
    inlet: PlanarInlet,
    tol: float,
) -> dict[str, np.ndarray]:
    wall_mask = prior_net.state_mask() & (
        prior_net.point_mask("wall") | prior_net.point_mask("corner")
    )
    rows, cols = np.nonzero(wall_mask)
    if rows.size == 0:
        return {}

    x = np.asarray(prior_net.x[rows, cols], dtype=float)
    y = np.asarray(prior_net.y[rows, cols], dtype=float)
    u = np.asarray(prior_net.u[rows, cols], dtype=float)
    v = np.asarray(prior_net.v[rows, cols], dtype=float)

    y_cowl = np.asarray([inlet.cowl.get_y(float(xv)) for xv in x], dtype=float)
    y_cent = np.asarray([inlet.centerbody.get_y(float(xv)) for xv in x], dtype=float)
    d_cowl = np.abs(y - y_cowl)
    d_cent = np.abs(y - y_cent)
    wall_tol = max(1.0e-8, 10.0 * float(tol))

    on_cowl = np.isfinite(y_cowl) & (d_cowl <= wall_tol)
    on_cent = np.isfinite(y_cent) & (d_cent <= wall_tol)
    on_cowl &= ~on_cent | (d_cowl <= d_cent)
    on_cent &= ~on_cowl | (d_cent < d_cowl)

    return {
        "rows": rows,
        "cols": cols,
        "x": x,
        "y": y,
        "u": u,
        "v": v,
        "on_cowl": on_cowl,
        "on_cent": on_cent,
    }


def get_next_event(
    prior_net: CharNet, inlet: PlanarInlet, tol: float = 1e-10
) -> SolverEvent | None:
    if prior_net.is_vertical:
        return None

    family = _next_event_family(prior_net)
    wall_data = _wall_event_arrays(prior_net, inlet, tol)
    if not wall_data:
        return None

    candidate_groups: list[dict[str, object]] = []
    for on_wall, wall in (
        (wall_data["on_cowl"], inlet.cowl),
        (wall_data["on_cent"], inlet.centerbody),
    ):
        if not np.any(on_wall):
            continue

        theta_flow = np.atan2(wall_data["v"][on_wall], wall_data["u"][on_wall])
        theta_wall = np.atan(
            np.asarray(
                [wall.get_dydx(float(xv)) for xv in wall_data["x"][on_wall]],
                dtype=float,
            )
        )
        delta = theta_wall - theta_flow
        valid = ~np.isclose(delta, 0.0, atol=tol)
        valid &= (delta * wall.normal_sign) > tol
        if not np.any(valid):
            continue

        candidate_groups.append(
            {
                "rows": wall_data["rows"][on_wall][valid],
                "cols": wall_data["cols"][on_wall][valid],
                "x": wall_data["x"][on_wall][valid],
                "delta": delta[valid],
                "wall": wall,
            }
        )

    if not candidate_groups:
        return None

    candidates_x = np.concatenate([group["x"] for group in candidate_groups])
    candidates_row = np.concatenate([group["rows"] for group in candidate_groups])
    candidates_col = np.concatenate([group["cols"] for group in candidate_groups])
    candidates_delta = np.concatenate([group["delta"] for group in candidate_groups])
    candidates_wall = np.concatenate(
        [
            np.full(group["x"].shape, group["wall"], dtype=object)
            for group in candidate_groups
        ]
    )
    x_min = float(np.min(candidates_x))
    event_mask = np.isclose(candidates_x, x_min, atol=tol)
    if np.count_nonzero(event_mask) > 1:
        msg = (
            "Detected simultaneous endpoint events at identical x; "
            "shock-shock intersections are not yet supported."
        )
        raise RuntimeError(msg)

    event_idx = int(np.flatnonzero(event_mask)[0])
    wall_from = candidates_wall[event_idx]
    point_idx = (int(candidates_row[event_idx]), int(candidates_col[event_idx]))

    return SolverEvent(
        family=family,
        point_idx=point_idx,
        wall_from=wall_from,
        wall_to=get_opposite_wall(prior_net.get_point(*point_idx)[:2], inlet),
        delta=float(candidates_delta[event_idx]),
    )


def check_shock_reflection(
    shock_pair: ShockPoint,
    point_idx: tuple[int, int],
    wall_from: PiecewiseLinearCurve,
    inlet: PlanarInlet,
    next_family: str,
    tol: float = 1e-10,
):
    x_sw, y_sw, u_sw, v_sw = shock_pair.pt_post

    theta_flow = np.atan2(v_sw, u_sw)
    theta_wall = np.atan(wall_from.get_dydx(x_sw))
    delta = theta_wall - theta_flow
    signed_turn = delta * wall_from.normal_sign

    if np.isclose(signed_turn, 0.0, atol=tol) or signed_turn < -tol:
        return None

    return SolverEvent(
        family=next_family,
        point_idx=point_idx,
        wall_from=wall_from,
        wall_to=get_opposite_wall(np.array([x_sw, y_sw]), inlet),
        delta=delta,
    )


def handle_event(
    inlet: PlanarInlet,
    max_iters: int,
    tol: float,
    plot: bool,
    net_im1: CharNet,
    event_i: SolverEvent | None,
    figdir: str | Path | None = None,
) -> tuple[CharNet, CharNet, SolverEvent | None] | None:
    if event_i is None:
        return None

    from inlet_moc.char_shock_solvers import DetachedShockError, SubsonicFlowError
    from inlet_moc.char_solvers import NoWallIntersectionError
    from inlet_moc.net_shock_solver import NetShockSolver

    solver = NetShockSolver(
        net=net_im1,
        inlet=inlet,
        event=event_i,
        max_iters=max_iters,
        tol=tol,
        figdir=figdir,
        plot=plot,
    )
    try:
        if event_i.family == "cminus":
            net_i, net_ip1, reflected_event = solver.solve_cminus()
        elif event_i.family == "cplus":
            net_i, net_ip1, reflected_event = solver.solve_cplus()
        else:
            msg = f"Unsupported event family '{event_i.family}'."
            raise ValueError(msg)
    except (
        NoWallIntersectionError,
        DetachedShockError,
        SubsonicFlowError,
    ) as err:
        err.current_net = net_im1
        raise

    if net_ip1 is None:
        msg = "Shock event returned no downstream net."
        raise RuntimeError(msg)

    return net_i, net_ip1, reflected_event
