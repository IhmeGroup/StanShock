from __future__ import annotations

from pathlib import Path

import cantera as ct
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import least_squares

from inlet_moc.processing.triangulated_solution import TriangulatedSolution

AVERAGE_COLUMNS = ("x", "rho_st", "u_st", "p_st", "a_st", "T_st", "mach")


def _streamthrust_residual(
    x_st: np.ndarray,
    mass: np.float64,
    momentum: np.float64,
    energy: np.float64,
    gas: ct.Solution,
    Y_st: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    T_st, u_st, p_st = x_st
    gas.TPY = T_st, p_st, Y_st
    rho_st = gas.density_mass
    h_st = gas.enthalpy_mass

    residual = np.array(
        [
            mass - rho_st * u_st,
            momentum - (rho_st * u_st**2 + p_st),
            energy - rho_st * u_st * (h_st + 0.5 * u_st**2),
        ]
    )
    return residual / scale


def _streamthrust_jacobian(
    x_st: np.ndarray,
    mass: np.float64,
    momentum: np.float64,
    energy: np.float64,
    gas: ct.Solution,
    Y_st: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    T_st, u_st, p_st = x_st
    gas.TPY = T_st, p_st, Y_st
    rho_st = gas.density_mass
    h_st = gas.enthalpy_mass
    cp_st = gas.cp_mass

    drho_dT = -rho_st / T_st
    drho_dp = rho_st / p_st
    h_total = h_st + 0.5 * u_st**2

    jac = np.empty((3, 3))
    jac[0, 0] = -drho_dT * u_st
    jac[0, 1] = -rho_st
    jac[0, 2] = -drho_dp * u_st

    jac[1, 0] = -drho_dT * u_st**2
    jac[1, 1] = -2.0 * rho_st * u_st
    jac[1, 2] = -drho_dp * u_st**2 - 1.0

    jac[2, 0] = -drho_dT * u_st * h_total - rho_st * u_st * cp_st
    jac[2, 1] = -rho_st * (h_st + 1.5 * u_st**2)
    jac[2, 2] = -drho_dp * u_st * h_total
    return jac / scale[:, None]


def flux_avg(
    y,
    rho,
    u,
    v,
    p,
    T,
    mech,
    Y: np.ndarray | None = None,
    gas: ct.Solution | None = None,
    x0_st: np.ndarray | None = None,
    verbose: bool = False,
):
    if gas is None:
        gas = ct.Solution(mech)
    nsp = gas.n_species

    y = np.asarray(y)
    rho = np.asarray(rho)
    u = np.asarray(u)
    p = np.asarray(p)
    T = np.asarray(T)

    N = len(y)
    if N < 3:
        msg = "At least three sample points are required."
        raise ValueError(msg)

    order = np.argsort(y, kind="stable")
    y = y[order]
    rho = rho[order]
    u = u[order]
    p = p[order]
    T = T[order]

    if Y is None:
        Y_eval = np.broadcast_to(gas.Y, (N, nsp)).copy()
    else:
        Y_arr = np.asarray(Y)
        if Y_arr.ndim == 1:
            if Y_arr.size != nsp:
                raise ValueError
            Y_eval = np.broadcast_to(Y_arr, (N, nsp)).copy()
        elif Y_arr.shape == (N, nsp):
            Y_eval = Y_arr[order]
        else:
            raise ValueError

    sol = ct.SolutionArray(gas, shape=y.shape)
    sol.TPY = T, p, Y_eval
    h = sol.enthalpy_mass

    A = abs(y[-1] - y[0])
    mass = trapezoid(rho * u, x=y) / A
    momentum = trapezoid(rho * u * u + p, x=y) / A
    energy = trapezoid(rho * u * (h + 0.5 * u**2), x=y) / A

    species = trapezoid(rho[:, None] * Y_eval * u[:, None], x=y, axis=0) / A

    Y_st = species / mass

    if x0_st is None:
        x0 = np.array([np.mean(T), np.mean(u), np.mean(p)])
    else:
        x0 = np.asarray(x0_st)
    lower = np.array(
        [
            max(1.0, 0.2 * np.min(T)),
            0.0,
            max(1.0, 0.05 * np.min(p)),
        ]
    )
    upper = np.array(
        [
            max(5000.0, 5.0 * np.max(T)),
            max(5000.0, 5.0 * np.max(np.abs(u))),
            max(2.0e5, 5.0 * np.max(p)),
        ]
    )
    x0 = np.clip(x0, lower, upper)
    scale = np.maximum(np.abs(np.array([mass, momentum, energy])), 1.0)

    sol = least_squares(
        _streamthrust_residual,
        x0,
        args=(mass, momentum, energy, gas, Y_st, scale),
        jac=_streamthrust_jacobian,
        bounds=(lower, upper),
        x_scale=np.maximum(np.abs(x0), 1.0),
        max_nfev=500,
    )

    if verbose and not sol.success:
        print(sol.message)

    T_st, u_st, p_st = sol.x
    gas.TPY = T_st, p_st, Y_st
    rho_st = gas.density_mass
    a_st = np.sqrt(gas.cp / gas.cv * p_st / rho_st)
    return rho_st, u_st, p_st, a_st, T_st


def streamthrust_average(
    x: np.ndarray,
    y_1: np.ndarray,
    y_2: np.ndarray,
    point_mesh: TriangulatedSolution,
    mech: str = "air.yaml",
    composition: str | None = None,
) -> np.ndarray:
    gas = ct.Solution(mech)
    if composition is not None:
        gas.X = composition
    Y = gas.Y.copy()
    R_mix = ct.gas_constant / gas.mean_molecular_weight
    x0_st = None

    avg = np.full((x.size, len(AVERAGE_COLUMNS)), np.nan)
    avg[:, 0] = x

    for idx, x_q in enumerate(x):
        _, y_seg, prim_seg, _ = point_mesh.get_primitives(x_q, y_1[idx], y_2[idx])
        if y_seg.shape[0] == 0:
            continue

        y = y_seg.reshape(-1)
        prim = prim_seg.reshape(-1, prim_seg.shape[-1])
        good = np.isfinite(y) & np.all(np.isfinite(prim), axis=1)
        y = y[good]
        prim = prim[good]
        if y.size < 2:
            continue

        order = np.argsort(y, kind="stable")
        y = y[order]
        prim = prim[order]

        if y.size == 2:
            y = np.array([y[0], 0.5 * (y[0] + y[1]), y[1]])
            prim = np.vstack((prim[0], 0.5 * (prim[0] + prim[1]), prim[1]))
        if y.size < 3:
            continue

        # prim is [rho, u, v, p, a], derived from [V, theta, p, rho].
        rho = prim[:, 0]
        u = prim[:, 1]
        v = prim[:, 2]
        p = prim[:, 3]
        T = p / (rho * R_mix)

        rho_st, u_st, p_st, a_st, T_st = flux_avg(
            y=y,
            rho=rho,
            u=u,
            v=v,
            p=p,
            T=T,
            mech=mech,
            Y=Y,
            gas=gas,
            x0_st=x0_st,
        )
        x0_st = np.array([T_st, u_st, p_st])
        avg[idx, 1:6] = rho_st, u_st, p_st, a_st, T_st
        if a_st > 0.0:
            avg[idx, 6] = u_st / a_st

    return avg


def compute_stream_thrust_average(
    *,
    point_mesh: TriangulatedSolution,
    bounds: tuple[np.ndarray, np.ndarray, np.ndarray],
    x: np.ndarray | None = None,
    mech: str = "air.yaml",
    composition: str | None = None,
) -> np.ndarray:
    x_b, y_1, y_2 = bounds
    if x is None:
        x = x_b
    else:
        x = np.asarray(x)
        y_1 = np.interp(x, x_b, y_1)
        y_2 = np.interp(x, x_b, y_2)

    return streamthrust_average(
        x=x,
        y_1=y_1,
        y_2=y_2,
        point_mesh=point_mesh,
        mech=mech,
        composition=composition,
    )


def save_stream_thrust_average_csv(
    output_path: str | Path,
    avg: np.ndarray,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        np.asarray(avg),
        delimiter=",",
        header=",".join(AVERAGE_COLUMNS),
        comments="",
        fmt="%.4e",
    )
    return output
