from __future__ import annotations

import time
from pathlib import Path
from typing import TypedDict

import cantera as ct
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve

from stanshock.components.shocktube import ShockTube
from stanshock.numerics.boundary_conditions import BCInput
from stanshock.numerics.face_extrapolation import (
    FaceExtrapolator,
    FifthOrderWeno,
    FirstOrder,
)
from stanshock.numerics.inviscid_flux import (
    RiemannSolver,
    hllc_flux_vectorized,
    lax_friedrichs_flux,
)
from stanshock.physics.fluid_base import FluidState
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.initialize import InitializeRiemannProblem
from stanshock.system.backend import Array
from stanshock.system.geometry import Geometry


class ShockTubeOptions(TypedDict, total=False):
    use_double_flux: bool
    flux_function: RiemannSolver
    inviscid_face_extrapolator: type[FaceExtrapolator]


cases: dict[str, ShockTubeOptions] = {
    "WENO5, HLLC": {
        "use_double_flux": False,
        "flux_function": hllc_flux_vectorized,
        "inviscid_face_extrapolator": FifthOrderWeno,
    },
    "WENO5, HLLC, +DF": {
        "use_double_flux": True,
        "flux_function": hllc_flux_vectorized,
        "inviscid_face_extrapolator": FifthOrderWeno,
    },
    "O1, HLLC": {
        "use_double_flux": False,
        "flux_function": hllc_flux_vectorized,
        "inviscid_face_extrapolator": FirstOrder,
    },
    "O1, LF": {
        "use_double_flux": False,
        "flux_function": lax_friedrichs_flux,
        "inviscid_face_extrapolator": FirstOrder,
    },
}

sod_case: dict[str, float] = {
    "t_final": 2.0,
    "L": 10.0,
    "rhoL": 1.0,
    "uL": 0.0,
    "pL": 1.0,
    "rhoR": 0.125,
    "uR": 0.0,
    "pR": 0.1,
}


def analytical_sod_solution(
    t: float,
    x: Array,
    gamma: float = 1.4,
) -> tuple[Array, Array, Array]:
    # Initial location of discontinuity
    x0 = 0.0

    # Initial properties in regions 1 and 4
    rho4 = sod_case["rhoL"]
    p4 = sod_case["pL"]
    u4 = sod_case["uL"]

    rho1 = sod_case["rhoR"]
    p1 = sod_case["pR"]
    u1 = sod_case["uR"]

    # Speeds of sound in regions 1 and 4
    c4 = np.sqrt(gamma * p4 / rho4)
    c1 = np.sqrt(gamma * p1 / rho1)

    def resid(y: Array) -> Array:
        # Nonlinear equation to get y = p2/p1
        C1 = np.sqrt((gamma + 1.0) / (2.0 * gamma) * (y - 1.0) + 1)
        C2 = (gamma - 1.0) / (2.0 * c4) * (u4 - u1 - c1 / gamma * (y - 1.0) / C1)
        exponent: float = -2.0 * gamma / (gamma - 1)
        return np.array(y * (1.0 + C2) ** exponent - p4 / p1)

    y0 = 0.5 * p4 / p1  # initial guess
    Y = float(fsolve(resid, y0)[0])

    # Region 2:
    p2 = Y * p1
    u2 = u1 + c1 / gamma * (p2 / p1 - 1) / np.sqrt(
        (gamma + 1) / (2 * gamma) * (p2 / p1 - 1) + 1
    )
    num = (gamma + 1) / (gamma - 1) + p2 / p1
    den = 1 + (gamma + 1) / (gamma - 1) * (p2 / p1)
    c2 = c1 * np.sqrt(p2 / p1 * num / den)
    # Shock speed
    V = u1 + c1 * np.sqrt((gamma + 1) / (2 * gamma) * (p2 / p1 - 1) + 1)
    rho2 = gamma * p2 / c2**2

    # Region 3:
    p3 = p2
    u3 = u2
    c3 = (gamma - 1) / 2 * (u4 - u3 + 2 / (gamma - 1) * c4)
    rho3 = gamma * p3 / c3**2

    # Expansion fan
    xe1 = (u4 - c4) * t + x0  # "start" of expansion fan
    xe2 = t * ((gamma + 1) / 2 * u3 - (gamma - 1) / 2 * u4 - c4) + x0  # end

    # Location of shock
    xs = V * t + x0
    # Location of contact
    xc = u2 * t + x0

    # Initialize with properties left of expansion fan (region 4)
    u = np.full_like(x, u4)
    p = np.full_like(x, p4)
    rho = np.full_like(x, rho4)

    # Expansion fan
    idx = np.where(np.logical_and(x > xe1, x <= xe2))[0]
    u[idx] = 2 / (gamma + 1) * ((x[idx] - x0) / t + (gamma - 1) / 2 * u4 + c4)
    c = u[idx] - (x[idx] - x0) / t
    p[idx] = p4 * (c / c4) ** (2 * gamma / (gamma - 1))
    rho[idx] = gamma * p[idx] / c**2

    # Between expansion fan and and contact discontinuity (region 3)
    idx = np.where(np.logical_and(x > xe2, x <= xc))[0]
    u[idx] = u3
    p[idx] = p3
    rho[idx] = rho3

    # Between the contact discontinuity and the shock (region 2)
    idx = np.where(np.logical_and(x > xc, x <= xs))[0]
    u[idx] = u2
    p[idx] = p2
    rho[idx] = rho2

    # Right of the shock (region 1)
    idx = np.where(x > xs)[0]
    u[idx] = u1
    p[idx] = p1
    rho[idx] = rho1

    return p, rho, u


# Get mechanism path relative to this script
mech_file = Path(__file__).resolve().parent / "../data/mechanisms/Nitrogen.yaml"

# Set up the Cantera Solutions for the left and right states
gas = ct.Solution(mech_file)
Tref = 600.0  # K
Pref = 101325.0  # Pa
gas.TP = Tref, Pref
g = gas.cp / gas.cv
Wm = gas.mean_molecular_weight
R = ct.gas_constant / Wm
rhoref = Pref / (R * Tref)

# Dimensionalize the case parameters
Lref = np.sqrt(g / 1.4 * Pref / rhoref)
gas_left = ct.Solution(mech_file)
gas_left.DP = sod_case["rhoL"] * rhoref, sod_case["pL"] * Pref
left_state = (gas_left, sod_case["uL"])

gas_right = ct.Solution(mech_file)
gas_right.DP = sod_case["rhoR"] * rhoref, sod_case["pR"] * Pref
right_state = (gas_right, sod_case["uR"])

# Set up geometry
L = sod_case["L"]
n_cells = 200  # mesh resolution
xf = np.linspace(-0.5 * L * Lref, 0.5 * L * Lref, n_cells + 1)
geometry = Geometry(xf, area=1.0)

# Get analytical solution on fine mesh
t_final = sod_case["t_final"]
x_analytical = np.linspace(-0.5 * L, L, 2001)
p_analytical, rho_analytical, u_analytical = analytical_sod_solution(
    t_final, x_analytical, g
)

# Set up solver parameters
boundary_conditions: BCInput = {"left": "reflecting", "right": "reflecting"}
physics = ThermoTable(gas_left)
initialization = InitializeRiemannProblem(
    geometry, physics, left_state, right_state, 0.0
)

final_states: dict[str, tuple[Array, FluidState]] = {}
for case_name, case_options in cases.items():
    print(f"Solving Sod shock tube problem {case_name}")
    case = ShockTube(
        geometry=geometry,
        physics=physics,
        initialization=initialization,
        boundary_conditions=boundary_conditions,
        cfl=0.9,
        output_every=100,
        **case_options,
    )

    # Solve
    t0 = time.perf_counter()
    case.advance_simulation(t_final)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)

    # Store the state
    idx = geometry.idx_cells
    x = geometry.xc[idx] / Lref
    state = case.state[idx]
    final_states[case_name] = (x, state)

# Plot the results
for ylabel in [r"$\rho$", "p", "u"]:
    ystring = ylabel.replace("\\", "").replace("$", "")
    y = np.zeros_like(x_analytical)

    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    ax.tick_params(axis="both", which="major", labelsize="18")

    # Plot analytical solution
    if ystring == "rho":
        y = rho_analytical
    elif ystring == "p":
        y = p_analytical
    elif ystring == "u":
        y = u_analytical
    ax.plot(x_analytical, y, "k", ls="--", lw=2, label="Analytical")

    # Plot simulation results
    for case_name, (x, state) in final_states.items():
        if ystring == "rho":
            assert state.density is not None
            y = state.density / rhoref
        elif ystring == "p":
            assert state.pressure is not None
            y = state.pressure / Pref
        elif ystring == "u":
            assert state.velocity is not None
            y = state.velocity / Lref

        ax.plot(x, y, lw=2, label=case_name)

    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.legend(loc="best", fontsize=18)

    fig.tight_layout()
    fig.savefig(f"sod_{ystring}.png")
    plt.close()
