from __future__ import annotations

import time
from pathlib import Path

import cantera as ct
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from stanshock.components.shocktube import ShockTube
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.probe import Probe
from stanshock.utils.csv_loader import get_pressure_data


def main(
    data_filename: str = "data/validation/case2.csv",
    mech_filename: str = "data/mechanisms/Nitrogen.yaml",
    plot_results: bool = True,
    show_results: bool = False,
    results_location: str | None = ".",
) -> dict[str, np.ndarray]:
    # =============================================================================
    # provided conditions for Case 2
    Ms = 2.518914
    T1 = 291.75
    p1 = 2026.499994
    p2 = 14730.642333
    tFinal = 60e-3

    # plotting parameters
    plot_results = plot_results or show_results
    fontsize = 12

    # provided geometry
    DDriven = 4.5 * 0.0254
    DDriver = 7.0 * 0.0254
    LDriver = 142.0 * 0.0254
    LDriven = 9.73

    # Set up gasses and determine the initial pressures
    u1 = 0.0
    u4 = 0.0  # initially 0 velocity
    gas1 = ct.Solution(mech_filename)
    gas4 = ct.Solution(mech_filename)
    T4 = T1  # assumed
    gas1.TP = T1, p1
    gas4.TP = T4, p1  # use p1 as a place holder
    g1 = gas1.cp / gas1.cv
    g4 = gas4.cp / gas4.cv
    a4oa1 = np.sqrt(
        g4 / g1 * T4 / T1 * gas1.mean_molecular_weight / gas4.mean_molecular_weight
    )
    p4 = p2 * (1.0 - (g4 - 1.0) / (g1 + 1.0) / a4oa1 * (Ms - 1.0 / Ms)) ** (
        -2.0 * g4 / (g4 - 1.0)
    )  # from handbook of shock waves
    p4 *= DDriven / DDriver  # just made this up
    p4 *= 1.05
    gas4.TP = T4, p4

    # set up geometry
    nX = 1000  # mesh resolution
    xLower = -LDriver
    xUpper = LDriven
    xShock = 0.0
    x = np.linspace(xLower, xUpper, nX)
    DeltaD = DDriven - DDriver
    DeltaX = (
        (xUpper - xLower) / float(nX) * 10
    )  # diffuse area change for numerical stability

    # DeltaX = 0.75 #from Eduardo's case
    def D(x):
        diameter = DDriven + (DeltaD / DeltaX) * (x - xShock)
        diameter[x < (xShock - DeltaX)] = DDriver
        diameter[x > xShock] = DDriven
        return diameter

    def dD_dx(x):
        dDiameterdx = np.ones(len(x)) * (DeltaD / DeltaX)
        dDiameterdx[x < (xShock - DeltaX)] = 0.0
        dDiameterdx[x > xShock] = 0.0
        return dDiameterdx

    def A(x):
        return np.pi / 4.0 * D(x) ** 2.0

    def dA_dx(x):
        return np.pi / 2.0 * D(x) * dD_dx(x)

    def dlnA_dx(x, t):
        return dA_dx(x) / A(x)

    # set up solver parameters
    print("Solving with boundary layer terms")
    boundary_conditions = ["reflecting", "reflecting"]
    state1 = (gas1, u1)
    state4 = (gas4, u4)
    physics_model = ThermoTable(gas1)

    ssbl = ShockTube(
        n=nX,
        x=x,
        physics=physics_model,
        initialization=("riemann", state4, state1, xShock),
        boundary_conditions=boundary_conditions,
        cfl=0.9,
        output_every=100,
        include_boundary_layer=True,
        wall_temperature=T1,  # assume wall temperature is in thermal eq. with gas
        d_outer=D,
        dlnA_dx=dlnA_dx,
    )
    ssbl.probes.append(Probe(ssbl, max(ssbl.x)))  # end wall probe

    # Solve
    t0 = time.perf_counter()
    ssbl.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)

    # without  boundary layer model
    print("Solving without boundary layer model")
    boundary_conditions = ["reflecting", "reflecting"]
    gas1.TP = T1, p1
    gas4.TP = T4, p4
    ssnbl = ShockTube(
        n=nX,
        x=x,
        physics=physics_model,
        initialization=("riemann", state4, state1, xShock),
        boundary_conditions=boundary_conditions,
        cfl=0.9,
        output_every=100,
        include_boundary_layer=False,
        d_outer=D,
        dlnA_dx=dlnA_dx,
    )
    ssnbl.probes.append(Probe(ssnbl, max(ssnbl.x)))  # end wall probe

    # Solve
    t0 = time.perf_counter()
    ssnbl.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)

    if plot_results:
        # import shock tube data
        tExp, pExp = get_pressure_data(data_filename)
        timeDifference = (
            12.211 - 8.10
        ) / 1000.0  # difference between the test data and simulation times
        tExp += timeDifference

        # make plots of probe and XT diagrams
        plt.close("all")
        mpl.rcParams["font.size"] = fontsize
        plt.rc("text", usetex=True)
        plt.figure(figsize=(4, 4))
        plt.plot(
            np.array(ssnbl.probes[0].t) * 1000.0,
            np.array(ssnbl.probes[0].p) / 1.0e5,
            "k",
            label=r"$\mathrm{Without\ BL\ Model}$",
            linewidth=2.0,
        )
        plt.plot(
            np.array(ssbl.probes[0].t) * 1000.0,
            np.array(ssbl.probes[0].p) / 1.0e5,
            "r",
            label=r"$\mathrm{With\ BL\ Model}$",
            linewidth=2.0,
        )
        plt.plot(tExp * 1000.0, pExp / 1.0e5, label=r"$\mathrm{Experiment}$", alpha=0.7)
        plt.axis([0, 60, -0.25, 2.75])
        plt.xlabel(r"$t\ [\mathrm{ms}]$")
        plt.ylabel(r"$p\ [\mathrm{bar}]$")
        plt.legend(loc="lower right")
        plt.tight_layout()
        if show_results:
            plt.show()

    results = {
        "pressure_with_boundary_layer": ssbl.probes[0].p,
        "pressure_without_boundary_layer": ssnbl.probes[0].p,
        "time_with_boundary_layer": ssbl.probes[0].t,
        "time_without_boundary_layer": ssnbl.probes[0].t,
    }

    if results_location is not None:
        np.savez(Path(results_location) / "case2.npz", **results)
        plt.savefig(Path(results_location) / "case2.png")

    return results


if __name__ == "__main__":
    main()
