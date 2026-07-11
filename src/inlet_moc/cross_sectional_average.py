from __future__ import annotations

from pathlib import Path

import cantera as ct
import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import root

from inlet_moc.triangulated_solution import TriangulatedSolution


AVERAGE_COLUMNS = ("x", "rho_st", "u_st", "p_st", "a_st", "T_st", "mach")


def flux_avg(
    y,
    rho,
    u,
    p,
    T,
    mech,
    Y: np.ndarray | None = None,
    verbose: bool = False,
):
    gas = ct.Solution(mech)
    nsp = gas.n_species

    y = np.asarray(y)
    rho = np.asarray(rho)
    u = np.asarray(u)
    p = np.asarray(p)
    T = np.asarray(T)

    N = len(y)
    if N < 3:
        raise ValueError("At least three sample points are required.")

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

    species = np.zeros((nsp,))
    for isp in range(nsp):
        species[isp] = trapezoid(rho * Y_eval[:, isp] * u, x=y) / A

    Yst = species / mass

    def residual(unknowns):
        Tst, ust, pst = unknowns
        gas.TPY = Tst, pst, Yst

        rhost = gas.density_mass
        hst = gas.enthalpy_mass

        return np.array(
            [
                mass - rhost * ust,
                momentum - (rhost * ust * ust + pst),
                energy - rhost * ust * (hst + 0.5 * ust**2),
            ]
        )

    x0 = [np.mean(T), np.mean(u), np.mean(p)]
    sol = root(residual, x0, options={"maxfev": 5000})

    Tst, ust, pst = sol.x

    gas.TPY = Tst, pst, Yst
    rhost = gas.density_mass
    ast = np.sqrt(gas.cp / gas.cv * pst / rhost)
    return rhost, ust, pst, ast, Tst


def streamthrust_average(
    x: np.ndarray,
    y_1: np.ndarray,
    y_2: np.ndarray,
    soln_tri: TriangulatedSolution,
    mech: str = "air.yaml",
    composition: str | None = None,
) -> np.ndarray:
    gas = ct.Solution(mech)
    if composition is not None:
        gas.X = composition
    Y = gas.Y.copy()
    R_mix = ct.gas_constant / gas.mean_molecular_weight

    x = np.asarray(x)
    y_1 = np.asarray(y_1)
    y_2 = np.asarray(y_2)
    if x.shape != y_1.shape or x.shape != y_2.shape:
        raise ValueError("x, y_1, and y_2 must have matching shapes.")

    avg = np.full((x.size, len(AVERAGE_COLUMNS)), np.nan)
    avg[:, 0] = x

    for idx, x_q in enumerate(x):
        _, y_seg, prim_seg, _ = soln_tri.get_primitives(x_q, y_1[idx], y_2[idx])
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

        rho = prim[:, 0]
        u = prim[:, 1]
        p = prim[:, 3]
        T = p / (rho * R_mix)

        rho_st, u_st, p_st, a_st, T_st = flux_avg(
            y=y,
            rho=rho,
            u=u,
            p=p,
            T=T,
            mech=mech,
            Y=Y,
        )
        avg[idx, 1:6] = rho_st, u_st, p_st, a_st, T_st
        if a_st > 0.0:
            avg[idx, 6] = u_st / a_st

    return avg


def compute_stream_thrust_average(
    soln,
    nx: int | None = None,
    x: np.ndarray | None = None,
    *,
    mech: str = "air.yaml",
    composition: str | None = None,
    bounds: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    soln_tri = soln.collect_integration_points()
    if bounds is None:
        from inlet_moc.get_streamtube_bounds import get_streamtube_bounds

        if x is None:
            nx_val = soln.nx if nx is None else nx
            x = np.linspace(soln.x_start, soln.x_final(), nx_val)
        bounds = get_streamtube_bounds(soln, soln_tri, x_q=x)

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
        soln_tri=soln_tri,
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
        fmt="%.4f",
    )
    return output
