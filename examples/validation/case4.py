from __future__ import annotations

import time
from pathlib import Path

import cantera as ct
import imageio
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from stanshock.components.shocktube import ShockTube
from stanshock.numerics.boundary_conditions import BCInput
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.initialize import InitializeRiemannProblem
from stanshock.processing.plot import XTDiagram
from stanshock.processing.probe import Probe
from stanshock.system.backend import Array
from stanshock.system.geometry import initialize_geometry


# =============================================================================
def get_pressure_data_from_image(fileName):
    """
    function getPressureData
    ==========================================================================
    This function returns the formatted pressure vs time data
        Inputs:
            fileName = name of csv data
        Outputs:
             t = time [s]
             p = pressure [Pa]
    """
    # parameters
    tLower, tUpper = 0.0, 0.05  # in [s]
    pLower, pUpper = 0.0, 1.032e5 * 5.0  # in [Pa]

    # read image file
    imageData = imageio.imread(fileName)
    imageData = imageData[-1::-1, :]
    imageData[imageData < 128] = 1.0
    imageData[imageData >= 128] = 0.0
    imageData = imageData[:, np.sum(imageData, axis=0) != 0]
    nP, nT = imageData.shape

    # extract pressure and time data
    p = np.linspace(pLower, pUpper, nP).reshape((nP, 1))
    t = np.linspace(tLower, tUpper, nT).reshape((nT, 1))
    p = imageData * p
    p = np.sum(p, axis=0) / np.sum(imageData, axis=0)
    return (t, p)


def main(
    data_filename: str = "data/validation/case4.png",
    mech_filename: str = "data/mechanisms/N2O2HeAr.yaml",
    plot_results: bool = True,
    show_results: bool = False,
    results_location: str | None = ".",
) -> dict[str, np.ndarray]:
    # =============================================================================
    # provided conditions for Case4
    T1 = T4 = 292.05
    p1 = 390.0 * 133.322
    p4 = 82.0 * 6894.76 * 0.9
    tFinal = 60e-3

    # plotting parameters
    plot_results = plot_results or show_results
    fontsize = 12

    # provided geometry
    DDriven = 4.5 * 0.0254
    # DDriver = DDriven
    LDriver = 142.0 * 0.0254
    LDriven = 9.73

    # Set up gasses and determine the initial pressures
    u1 = 0.0
    u4 = 0.0  # initially 0 velocity
    gas1 = ct.Solution(mech_filename)
    gas4 = ct.Solution(mech_filename)
    T4 = T1  # assumed
    gas1.TPX = T1, p1, "O2:0.21,AR:0.79"
    gas4.TPX = T4, p4, "HE:0.25,N2:0.75"

    # set up geometry
    nX = 1000  # mesh resolution
    xLower = -LDriver
    xUpper = LDriven
    xShock = 0.0
    xf = np.linspace(xLower, xUpper, nX + 1)
    # arrays from HTGL
    xInterp = -0.0254 * np.array(
        [142, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40, 36, 37, 30, 20, 10, 0]
    )
    dInterp = 0.0254 * np.array(
        [
            3.25,
            3.21,
            3.01,
            2.81,
            2.61,
            2.41,
            2.21,
            2.01,
            1.81,
            1.61,
            1.41,
            1.21,
            1.13,
            0.00,
            0.00,
            0.00,
            0.00,
            0.00,
        ]
    )
    dDInterpdxInterp = (dInterp[1:] - dInterp[:-1]) / (xInterp[1:] - xInterp[:-1])

    def d_outer(time: float, x: Array) -> Array:
        nX = x.shape[0]
        return DDriven * np.ones(nX)

    def d_inner(time: float, x: Array) -> Array:
        return np.interp(x, xInterp, dInterp)

    def dd_outerdx(time: float, x: Array) -> Array:
        return np.zeros(nX)

    def dd_innerdx(time: float, x: Array) -> Array:
        return np.interp(x, xInterp[:-1], dDInterpdxInterp)

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

    # solve with boundary layer model
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
        include_boundary_layer=True,
        wall_temperature=T1,  # assume wall temperature is in thermal eq. with gas
    )
    ssbl.state.gamma = ssbl.physics.get_gamma(ssbl.state)
    ssbl.probes.append(Probe(ssbl, max(ssbl.geometry.xf)))  # end wall probe
    diagram_settings = [
        ("pressure", (p1 / 101325, p4 / 101325)),
        ("temperature", (T1, 800.0)),
    ]
    ssbl.xt_diagrams += [
        XTDiagram(ssbl, variable=variable, limits=limits)
        for variable, limits in diagram_settings
    ]

    # adjust for partial filling strategy
    XN2Lower = 0.80  # assume smearing during fill
    XN2Upper = 1.5 - XN2Lower
    idx = ssbl.geometry.idx_cells
    xc = ssbl.geometry.xc[idx]
    dV = ssbl.geometry.volume(0.0, ssbl.geometry.xf)
    VDriver = np.sum(dV[xc < xShock])
    V = np.cumsum(dV)
    V -= V[0] / 2.0  # center
    VNorms = V / VDriver
    # get gas properties
    iHE, iN2 = gas4.species_index("HE"), gas4.species_index("N2")
    mt = ssbl.geometry.n_ghost_layers
    for iX, VNorm in enumerate(VNorms):
        if VNorm <= 1.0:
            # nitrogen and helium
            X = np.zeros(gas4.n_species)
            XN2 = XN2Lower + (XN2Upper - XN2Lower) * VNorm
            XHE = 1.0 - XN2
            X[[iHE, iN2]] = XHE, XN2
            gas4.TPX = T4, p4, X
            ssbl.state.density[iX + mt] = gas4.density
            ssbl.state.composition[iX + mt, :] = gas4.Y
            ssbl.state.gamma[iX + mt] = gas4.cp / gas4.cv

    # Solve
    t0 = time.perf_counter()
    ssbl.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)

    if plot_results:
        for diagram in ssbl.xt_diagrams:
            diagram.plot()

    # Solve without boundary layer model
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
        include_boundary_layer=False,
        wall_temperature=T1,  # assume wall temperature is in thermal eq. with gas
    )
    ssnbl.state.gamma = ssnbl.physics.get_gamma(ssnbl.state)
    ssnbl.probes.append(Probe(ssnbl, max(ssnbl.geometry.xf)))  # end wall probe
    ssnbl.xt_diagrams += [
        XTDiagram(ssnbl, variable=variable, limits=limits)
        for variable, limits in diagram_settings
    ]

    # adjust for partial filling strategy
    XN2Lower = 0.80  # assume smearing during fill
    XN2Upper = 1.5 - XN2Lower
    idx = ssnbl.geometry.idx_cells
    xc = ssnbl.geometry.xc[idx]
    dV = ssnbl.geometry.volume(0.0, ssnbl.geometry.xf)
    VDriver = np.sum(dV[xc < xShock])
    V = np.cumsum(dV)
    V -= V[0] / 2.0  # center
    VNorms = V / VDriver
    # get gas properties
    iHE, iN2 = gas4.species_index("HE"), gas4.species_index("N2")
    mt = ssnbl.geometry.n_ghost_layers
    for iX, VNorm in enumerate(VNorms):
        if VNorm <= 1.0:
            # nitrogen and helium
            X = np.zeros(gas4.n_species)
            XN2 = XN2Lower + (XN2Upper - XN2Lower) * VNorm
            XHE = 1.0 - XN2
            X[[iHE, iN2]] = XHE, XN2
            gas4.TPX = T4, p4, X
            ssnbl.state.density[iX + mt] = gas4.density
            ssnbl.state.composition[iX + mt, :] = gas4.Y
            ssnbl.state.gamma[iX + mt] = gas4.cp / gas4.cv

    # Solve
    t0 = time.perf_counter()
    ssnbl.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)

    if plot_results:
        for diagram in ssnbl.xt_diagrams:
            diagram.plot()

        # import shock tube data
        tExp, pExp = get_pressure_data_from_image(data_filename)
        timeDifference = (
            18.6 - 6.40
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
        plt.axis([0, 60, 0, 5])
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
        np.savez(Path(results_location, "case4.npz"), **results)
        plt.savefig(Path(results_location, "case4.png"))

    return results


if __name__ == "__main__":
    main()
