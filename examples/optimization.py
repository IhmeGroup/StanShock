from __future__ import annotations

import time
from pathlib import Path

import cantera as ct
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import newton

from stanshock.components.shocktube import ShockTube
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.initialize import (
    smoothing_function,
    smoothing_function_gradient,
)
from stanshock.processing.plot import XTDiagram
from stanshock.processing.probe import Probe
from stanshock.system.backend import Array


def main(
    mech_filename: str = "data/mechanisms/HeliumArgon.yaml",
    plot_results: bool = True,
    show_results: bool = False,
    results_location: str | None = ".",
) -> None:
    # parameters
    fontsize = 12
    tFinal = 7.5e-3
    p5, p1 = 18 * ct.one_atm, 0.48e5
    T5 = 1698.0
    g4 = g1 = 5.0 / 3.0  # monatomic gas in driver and driven sections
    W4, W1 = 4.002602, 39.948  # Helium and argon
    MachReduction = 0.985  # account for shock wave attenuation
    nXCoarse, nXFine = 200, 1000  # mesh resolution
    LDriver, LDriven = 3.0, 5.0
    DDriver, DDriven = 7.5e-2, 5.0e-2

    plot_results = plot_results or show_results
    if plot_results:
        plt.close("all")
        mpl.rcParams["font.size"] = fontsize
        plt.rc("text", usetex=True)

    # set up geometry
    xLower = -LDriver
    xUpper = LDriven
    xShock = 0.0
    Delta = 10 * (xUpper - xLower) / float(nXFine)
    x = np.linspace(xLower, xUpper, nXCoarse)

    def d_inner(time: float, x: Array) -> Array:
        return np.zeros_like(x)

    def dd_inner_dx(time: float, x: Array) -> Array:
        return np.zeros_like(x)

    def d_outer(time: float, x: Array) -> Array:
        return smoothing_function(x, xShock, Delta, DDriver, DDriven)

    def dd_outer_dx(time: float, x: Array) -> Array:
        return smoothing_function_gradient(x, xShock, Delta, DDriver, DDriven)

    def A(time: float, x: Array) -> Array:
        return np.pi / 4.0 * (d_outer(time, x) ** 2.0 - d_inner(time, x) ** 2.0)

    def dA_dx(time: float, x: Array) -> Array:
        return (
            0.5
            * np.pi
            * (
                d_outer(time, x) * dd_outer_dx(time, x)
                - d_inner(time, x) * dd_inner_dx(time, x)
            )
        )

    def dlnA_dx(time: float, x: Array) -> Array:
        return dA_dx(time, x) / A(time, x)

    # compute the gas dynamics
    def res(Ms1):
        return p5 / p1 - ((2.0 * g1 * Ms1**2.0 - (g1 - 1.0)) / (g1 + 1.0)) * (
            (-2.0 * (g1 - 1.0) + Ms1**2.0 * (3.0 * g1 - 1.0))
            / (2.0 + Ms1**2.0 * (g1 - 1.0))
        )

    Ms1 = newton(res, 2.0)
    Ms1 *= MachReduction
    T5oT1 = (
        (2.0 * (g1 - 1.0) * Ms1**2.0 + 3.0 - g1)
        * ((3.0 * g1 - 1.0) * Ms1**2.0 - 2.0 * (g1 - 1.0))
        / ((g1 + 1.0) ** 2.0 * Ms1**2.0)
    )
    T1 = T5 / T5oT1
    a1oa4 = np.sqrt(W4 / W1)
    p4op1 = (1.0 + 2.0 * g1 / (g1 + 1.0) * (Ms1**2.0 - 1.0)) * (
        1.0 - (g4 - 1.0) / (g4 + 1.0) * a1oa4 * (Ms1 - 1.0 / Ms1)
    ) ** (-2.0 * g4 / (g4 - 1.0))
    p4 = p1 * p4op1

    # set up the gasses
    u1 = 0.0
    u4 = 0.0  # initially 0 velocity
    gas1 = ct.Solution(mech_filename)
    gas4 = ct.Solution(mech_filename)
    T4 = T1  # assumed
    gas1.TPX = T1, p1, "AR:1"
    gas4.TPX = T4, p4, "HE:1"

    # set up solver parameters
    boundary_conditions = ["reflecting", "reflecting"]
    state1 = (gas1, u1)
    state4 = (gas4, u4)
    physics_model = ThermoTable(gas1)

    ss = ShockTube(
        x=x,
        physics=physics_model,
        initialization=("riemann", state4, state1, xShock),
        boundary_conditions=boundary_conditions,
        cfl=0.9,
        output_every=100,
        include_boundary_layer=True,
        wall_temperature=T1,  # assume wall temperature is in thermal eq. with gas
        d_outer=d_outer,
        dlnA_dx=dlnA_dx,
    )
    ss.state.gamma = ss.physics.get_gamma(ss.state)

    # Solve
    t0 = time.perf_counter()
    tTest = 2e-3
    tradeoffParam = 1.0
    eps = 0.01**2.0 + tradeoffParam * 0.01**2.0
    ss.optimize_driver_insert(
        tFinal, p5=p5, tTest=tTest, tradeoffParam=tradeoffParam, eps=eps
    )
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)

    # recalculate at higher resolution with the insert
    x = np.linspace(xLower, xUpper, nXFine)
    gas1.TPX = T1, p1, "AR:1"
    gas4.TPX = T4, p4, "HE:1"
    ss = ShockTube(
        x=x,
        physics=physics_model,
        initialization=("riemann", state4, state1, xShock),
        boundary_conditions=boundary_conditions,
        cfl=0.9,
        output_every=100,
        include_boundary_layer=True,
        wall_temperature=T1,  # assume wall temperature is in thermal eq. with gas
        d_outer=d_outer,
        d_inner=ss.geometry.d_inner,
        dlnA_dx=ss.geometry.dlnA_dx,
    )

    if plot_results:
        diagram_settings = [
            ("pressure", [0.5, 25]),
            ("temperature", [200.0, 1800.0]),
        ]
        ss.xt_diagrams += [
            XTDiagram(ss, variable=variable, limits=limits)
            for variable, limits in diagram_settings
        ]
    ss.probes.append(Probe(ss, max(ss.geometry.x)))  # end wall probe
    t0 = time.perf_counter()
    ss.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)
    pInsert = np.array(ss.probes[0].p)
    tInsert = np.array(ss.probes[0].t)

    for diagram in ss.xt_diagrams:
        diagram.plot()

    xInsert = ss.geometry.x
    d_outer_insert = ss.geometry.d_outer(0.0, ss.geometry.x)
    d_inner_insert = ss.geometry.d_inner(0.0, ss.geometry.x)

    # recalculate at higher resolution without the insert
    gas1.TPX = T1, p1, "AR:1"
    gas4.TPX = T4, p4, "HE:1"
    ss = ShockTube(
        x=x,
        physics=physics_model,
        initialization=("riemann", state4, state1, xShock),
        boundary_conditions=boundary_conditions,
        cfl=0.9,
        output_every=100,
        include_boundary_layer=True,
        wall_temperature=T1,  # assume wall temperature is in thermal eq. with gas
        d_outer=d_outer,
        dlnA_dx=dlnA_dx,
    )
    if plot_results:
        ss.xt_diagrams += [
            XTDiagram(ss, variable=variable, limits=limits)
            for variable, limits in diagram_settings
        ]
    ss.probes.append(Probe(ss, max(ss.geometry.x)))  # end wall probe
    t0 = time.perf_counter()
    ss.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)
    pNoInsert = np.array(ss.probes[0].p)
    tNoInsert = np.array(ss.probes[0].t)
    # plot
    if plot_results:
        for diagram in ss.xt_diagrams:
            diagram.plot()

        plt.figure()
        plt.plot(tNoInsert / 1e-3, pNoInsert / 1e5, "k", label=r"$\mathrm{No\ Insert}$")
        plt.plot(
            tInsert / 1e-3, pInsert / 1e5, "r", label=r"$\mathrm{Optimized\ Insert}$"
        )
        plt.xlabel(r"$t\ [\mathrm{ms}]$")
        plt.ylabel(r"$p\ [\mathrm{bar}]$")
        plt.legend(loc="best")
        plt.tight_layout()

        plt.figure()
        plt.plot(xInsert, d_outer_insert, "k", label=r"$D_\mathrm{o}$")
        plt.plot(xInsert, d_inner_insert, "r", label=r"$D_\mathrm{i}$")
        plt.xlabel(r"$x\ [\mathrm{m}]$")
        plt.ylabel(r"$D\ [\mathrm{m}]$")
        plt.legend(loc="best")
        plt.tight_layout()
    if show_results:
        plt.show()

    results = {
        "pressure_with_insert": pInsert,
        "pressure_without_insert": pNoInsert,
        "insert_diameter": d_inner_insert,
        "shock_tube_diameter": d_outer_insert,
        "position": xInsert,
        "time_with_insert": tInsert,
        "time_without_insert": tNoInsert,
    }
    if results_location is not None:
        np.savez(Path(results_location) / "optimization.npz", **results)
        plt.savefig(Path(results_location) / "optimization.png")

    return results


if __name__ == "__main__":
    main()
