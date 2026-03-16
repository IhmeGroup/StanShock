from __future__ import annotations

import time
from pathlib import Path

import cantera as ct
import numpy as np
from matplotlib import pyplot as plt

from stanshock.components.shocktube import ShockTube
from stanshock.models.wall_models import (
    CompressibleHeatFlux,
    CompressibleInertSkinFriction,
)
from stanshock.numerics.boundary_conditions import BCInput
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.initialize import (
    InitializeRiemannProblem,
    smoothing_function,
    smoothing_function_gradient,
)
from stanshock.processing.probe import Probe
from stanshock.system.backend import Array
from stanshock.system.geometry import initialize_geometry
from stanshock.utils.csv_loader import get_pressure_data

data_dir = Path(__file__).resolve().parent / "../../data"


def main(
    data_filename: Path | str = data_dir / "validation/case3.csv",
    mech_filename: Path | str = data_dir / "mechanisms/Nitrogen.yaml",
    plot_results: bool = True,
    show_results: bool = False,
    results_location: str | None = ".",
) -> dict[str, np.ndarray]:
    # =============================================================================
    # provided conditions for case 3
    Ms = 2.409616
    T1 = 292.25
    p1 = 1999.83552
    p2 = 13267.880629
    tFinal = 60e-3

    plot_results = plot_results or show_results

    # provided geometry
    DDriven = 4.5 * 0.0254
    # DDriver = 4.5 * 0.0254
    LDriver = 142.0 * 0.0254
    LDriven = 9.73
    d_outerInsertBack = 3.375 * 0.0254
    d_outerInsertFront = 1.25 * 0.0254
    LOuterInsert = 102.0 * 0.0254
    d_innerInsert = 0.625 * 0.0254
    LInnerInsert = 117.0 * 0.0254

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
    p4 *= 1.04
    gas4.TP = T4, p4

    # set up geometry
    nX = 1000  # mesh resolution
    xLower = -LDriver
    xUpper = LDriven
    xShock = 0.0
    xf = np.linspace(xLower, xUpper, nX + 1)
    # DeltaD = DDriven - DDriver
    # dd_outerInsertdx = (d_outerInsertFront - d_outerInsertBack) / LOuterInsert
    DeltaSmoothingFunction = (xUpper - xLower) / float(nX) * 10.0

    def d_outer(time: float, x: Array) -> Array:
        return DDriven * np.ones(nX)

    def d_inner(time: float, x: Array) -> Array:
        diameter = np.zeros(nX)
        diameter += smoothing_function(
            x, xLower + LInnerInsert, DeltaSmoothingFunction, d_innerInsert, 0.0
        )
        diameter += smoothing_function(
            x,
            xLower + LOuterInsert,
            DeltaSmoothingFunction,
            d_outerInsertFront - d_innerInsert,
            0.0,
        )
        diameter += smoothing_function(
            x,
            xLower + LOuterInsert / 2.0,
            LOuterInsert,
            d_outerInsertBack - d_outerInsertFront,
            0.0,
        )
        return diameter

    def dd_outerdx(time: float, x: Array) -> Array:
        return np.zeros(nX)

    def dd_innerdx(time: float, x: Array) -> Array:
        dDiameterdx = np.zeros(nX)
        dDiameterdx += smoothing_function_gradient(
            x, xLower + LInnerInsert, DeltaSmoothingFunction, d_innerInsert, 0.0
        )
        dDiameterdx += smoothing_function_gradient(
            x,
            xLower + LOuterInsert,
            DeltaSmoothingFunction,
            d_outerInsertFront - d_innerInsert,
            0.0,
        )
        dDiameterdx += smoothing_function_gradient(
            x,
            xLower + LOuterInsert / 2.0,
            LOuterInsert,
            d_outerInsertBack - d_outerInsertFront,
            0.0,
        )
        return dDiameterdx

    def A(time: float, x: Array) -> Array:
        return np.pi / 4.0 * (d_outer(time, x) ** 2.0 - d_inner(time, x) ** 2.0)

    def dA_dx(time: float, x: Array) -> Array:
        return (
            np.pi
            / 2.0
            * (
                d_outer(time, x) * dd_outerdx(time, x)
                - d_inner(time, x) * dd_innerdx(time, x)
            )
        )

    def dlnA_dx(time: float, x: Array) -> Array:
        return dA_dx(time, x) / A(time, x)

    geometry = initialize_geometry(
        xf, d_inner=d_inner, d_outer=d_outer, dlnA_dx=dlnA_dx
    )

    # set up solver parameters
    print("Solving with boundary layer terms")
    boundary_conditions: BCInput = {"left": "reflecting", "right": "reflecting"}
    state1 = (gas1, u1)
    state4 = (gas4, u4)
    physics_model = ThermoTable(gas1)
    initialization = InitializeRiemannProblem(
        geometry, physics_model, state4, state1, xShock
    )

    ssbl = ShockTube(
        geometry=geometry,
        physics=physics_model,
        initialization=initialization,
        boundary_conditions=boundary_conditions,
        cfl=0.9,
        output_every=100,
        wall_models=(CompressibleInertSkinFriction(), CompressibleHeatFlux()),
        wall_temperature=T1,  # assume wall temperature is in thermal eq. with gas
    )
    ssbl.probes.append(
        Probe(geometry, physics_model, max(geometry.xf))
    )  # end wall probe

    # Solve
    t0 = time.perf_counter()
    ssbl.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)

    # without  boundary layer model
    print("Solving without boundary layer model")
    boundary_conditions = {"left": "reflecting", "right": "reflecting"}
    gas1.TP = T1, p1
    gas4.TP = T4, p4
    ssnbl = ShockTube(
        geometry=geometry,
        physics=physics_model,
        initialization=initialization,
        boundary_conditions=boundary_conditions,
        cfl=0.9,
        output_every=100,
    )
    ssnbl.probes.append(
        Probe(geometry, physics_model, max(geometry.xf))
    )  # end wall probe

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
        plt.axis([0, 60, -0.5, 2])
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
        np.savez(Path(results_location) / "case3.npz", **results)
        plt.savefig(Path(results_location) / "case3.png")

    return results


if __name__ == "__main__":
    main()
