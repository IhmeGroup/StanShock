from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from inlet_moc.triangulated_solution import TriangulatedSolution


def mass_flux_x(y: np.ndarray, primitives: np.ndarray) -> float:
    if y.shape[0] == 0:
        return 0.0

    rho_u = primitives[:, :, 0] * primitives[:, :, 1]
    dy = np.abs(y[:, 1] - y[:, 0])
    return np.sum(0.5 * (rho_u[:, 0] + rho_u[:, 1]) * dy)


def _flux_residual(
    y_missing: float,
    soln_tri: TriangulatedSolution,
    x: float,
    y_known: float,
    ref_mass_flux: float,
) -> float:
    _, y, primitives, _ = soln_tri.get_primitives(x, y_known, y_missing)
    return mass_flux_x(y, primitives) - ref_mass_flux


def _solve_boundary_height(
    *,
    soln_tri: TriangulatedSolution,
    x: float,
    y_known: float,
    y_guess: float,
    ref_mass_flux: float,
    y_cap: float,
    tol: float,
) -> float:
    direction = np.sign(y_cap - y_known)

    eps = max(tol, 1.0e-10)
    y_near = y_known + direction * eps
    if direction > 0.0:
        y_far = np.clip(max(y_guess, y_near + eps), y_near + eps, y_cap)
    else:
        y_far = np.clip(min(y_guess, y_near - eps), y_cap, y_near - eps)

    f_near = _flux_residual(y_near, soln_tri, x, y_known, ref_mass_flux)
    if f_near >= 0.0:
        return y_near

    f_far = _flux_residual(y_far, soln_tri, x, y_known, ref_mass_flux)
    expansion = 1.35
    while f_far < 0.0 and abs(y_far - y_cap) > eps:
        span = min(abs(y_cap - y_known), expansion * abs(y_far - y_known))
        y_far = y_known + direction * span
        f_far = _flux_residual(y_far, soln_tri, x, y_known, ref_mass_flux)
        expansion *= 1.15

    y_a, y_b = sorted((y_near, y_far))
    return brentq(
        _flux_residual,
        y_a,
        y_b,
        args=(
            soln_tri,
            x,
            y_known,
            ref_mass_flux,
        ),
        xtol=max(tol, 1.0e-10),
        rtol=1.0e-10,
        maxiter=100,
    )


def get_streamtube_bounds(
    soln,
    soln_tri: TriangulatedSolution | None = None,
    *,
    x_q: np.ndarray | None = None,
    tol: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tol_val = soln.tol if tol is None else tol
    soln_tri = soln.collect_integration_points() if soln_tri is None else soln_tri
    cent = soln.inlet.centerbody
    cowl = soln.inlet.cowl

    x_ref = max(cent.x_min, cowl.x_min)
    x_lo = min(cent.x_min, cowl.x_min)
    x_hi = min(cent.x_max, cowl.x_max, soln.x_final())

    if x_q is None:
        x_q = np.linspace(x_lo, x_hi, soln.nx)
    x_q = np.asarray(x_q)
    x_q = np.unique(x_q[np.isfinite(x_q)])
    x_q = x_q[(x_q >= x_lo - tol_val) & (x_q <= x_hi + tol_val)]
    if x_q.size == 0:
        raise RuntimeError

    y_ref_cent = cent.get_y(x_ref)
    y_ref_cowl = cowl.get_y(x_ref)
    if y_ref_cent is None or y_ref_cowl is None:
        raise RuntimeError

    x_ref, y_ref, prim_ref, _ = soln_tri.get_primitives(x_ref, y_ref_cent, y_ref_cowl)
    mass_flux_x_ref = mass_flux_x(y_ref, prim_ref)
    if mass_flux_x_ref <= 0.0:
        raise RuntimeError

    x_external = x_q[x_q < x_ref - tol_val]
    x_internal = np.unique(np.concatenate(([x_ref], x_q[x_q >= x_ref - tol_val])))
    x_internal = x_internal[x_internal <= x_hi + tol_val]

    y_int_cent = [cent.get_y(x) for x in x_internal]
    y_int_cowl = [cowl.get_y(x) for x in x_internal]
    if any(y is None for y in y_int_cent) or any(y is None for y in y_int_cowl):
        raise RuntimeError
    y_int_cent = np.asarray(y_int_cent)
    y_int_cowl = np.asarray(y_int_cowl)

    missing_cowl = cowl.x_min > cent.x_min
    if missing_cowl:
        known_wall = cent
        y_ref_missing = y_ref_cowl
        y_cap = soln_tri.y_max_global
    else:
        known_wall = cowl
        y_ref_missing = y_ref_cent
        y_cap = soln_tri.y_min_global

    if not np.isfinite(y_cap):
        raise RuntimeError

    x_ext_desc = x_external[::-1]
    y_ext_known_desc = []
    y_ext_missing_desc = []
    y_guess = y_ref_missing
    for x in x_ext_desc:
        y_known = known_wall.get_y(x)
        if y_known is None:
            raise RuntimeError

        y_missing = _solve_boundary_height(
            soln_tri=soln_tri,
            x=x,
            y_known=y_known,
            y_guess=y_guess,
            ref_mass_flux=mass_flux_x_ref,
            y_cap=y_cap,
            tol=tol_val,
        )
        y_ext_known_desc.append(y_known)
        y_ext_missing_desc.append(y_missing)
        y_guess = y_missing

    x_ext = x_ext_desc[::-1]
    y_ext_known = np.asarray(y_ext_known_desc[::-1])
    y_ext_missing = np.asarray(y_ext_missing_desc[::-1])

    x_final = np.concatenate((x_ext, x_internal))
    if missing_cowl:
        y_cent = np.concatenate((y_ext_known, y_int_cent))
        y_cowl = np.concatenate((y_ext_missing, y_int_cowl))
    else:
        y_cent = np.concatenate((y_ext_missing, y_int_cent))
        y_cowl = np.concatenate((y_ext_known, y_int_cowl))

    return x_final, y_cent, y_cowl
