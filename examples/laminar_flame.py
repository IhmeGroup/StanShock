from __future__ import annotations

import time
from pathlib import Path

import cantera as ct
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from stanshock.components.combustor import Combustor
from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.physics.fluid_base import FluidState
from stanshock.physics.thermotable import ThermoTable


def main(
    sim_time: float | None = None,
    mech_filename: str = "ohn.yaml",
    plot_results: bool = True,
    show_results: bool = False,
    results_location: str | None = ".",
    physics_model: str = "ThermoTable",
) -> dict[str, np.ndarray]:
    # user parameters
    TU = 300.0
    p = 1e5
    estFlameThickness = 1e-2
    ntFlowThrough = 0.1
    fontsize = 12
    f = 0.1  # factor to reduce Cantera domain

    # find the initial state of the fluids
    gas = ct.Solution(mech_filename)
    unburnedState = TU, p, "H2:2,O2:1,N2:3.76"
    gas.TPX = unburnedState

    # get the flame thickness
    _, flame = flameSpeed(gas, estFlameThickness, returnFlame=True)
    TU, TB = flame.T[0], flame.T[-1]
    flameThickness = (TB - TU) / max(np.gradient(flame.T, flame.grid))

    # get flame parameters
    gasUnburned = ct.Solution(mech_filename)
    gasUnburned.TPY = flame.T[0], flame.P, flame.Y[:, 0]
    uUnburned = flame.velocity[0]
    unburnedState = gasUnburned, uUnburned
    gasBurned = ct.Solution(mech_filename)
    gasBurned.TPY = flame.T[-1], flame.P, flame.Y[:, -1]
    uBurned = flame.velocity[-1]
    burnedState = gasBurned, uBurned

    # set up grid
    nX = flame.grid.shape[0]
    flame_center = flame.grid[np.argmax(np.gradient(flame.T, flame.grid))]
    L = flame.grid[-1] - flame.grid[0]
    xUpper, xLower = flame_center + L * f, flame_center - L * f
    boundaryConditions = (
        (gasUnburned.density, uUnburned, None, gasUnburned.Y),
        (None, None, gasBurned.P, None),
    )
    if physics_model == "ThermoTable":
        physics = ThermoTable(gas)
    else:
        physics = CanteraInterface(gas)

    ss = Combustor(
        n=nX,
        x=np.linspace(xLower, xUpper, nX),
        dx=(xUpper-xLower)/(nX-1),
        initialization=("Riemann", unburnedState, burnedState, flame_center),
        physics=physics,
        boundaryConditions=boundaryConditions,
        cfl=0.9,
        reacting=True,
        includeDiffusion=True,
        outputEvery=10,
    )

    # interpolate flame solution
    Y = ss.state.composition
    for iSp in range(gas.n_species):
        Y[:, iSp] = np.interp(ss.x, flame.grid, flame.Y[iSp, :])

    ss.state = FluidState(
        shape=nX,
        density=np.interp(ss.x, flame.grid, flame.density),
        velocity=np.interp(ss.x, flame.grid, flame.velocity),
        pressure=flame.P * np.ones(ss.n),
        composition=Y,
    )
    T = ss.state.temperature = ss.physics.get_temperature(ss.state)
    ss.state.gamma = ss.physics.get_gamma(ss.state)

    # calculate the final time if not specified
    if sim_time is None:
        sim_time = ntFlowThrough * (xUpper - xLower) / (uUnburned + uBurned) * 2.0

    # Solve
    t0 = time.perf_counter()
    ss.advance_simulation(sim_time)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)

    # plot setup
    if plot_results:
        plt.close("all")
        font = {"family": "serif", "serif": ["computer modern roman"]}
        plt.rc("font", **font)
        mpl.rcParams["font.size"] = fontsize
        plt.rc("text", usetex=True)
        # plot
        plt.plot(
            (flame.grid - flame_center) / flameThickness,
            flame.T / flame.T[-1],
            "r",
            label=r"$T/T_\mathrm{F}$",
        )
        T = ss.physics.get_temperature(ss.state)
        plt.plot((ss.x - flame_center) / flameThickness, T / flame.T[-1], "r--s")
        iOH = gas.species_index("OH")
        plt.plot(
            (flame.grid - flame_center) / flameThickness,
            flame.Y[iOH, :] * 10,
            "k",
            label=r"$Y_\mathrm{OH}\times 10$",
        )
        plt.plot((ss.x - flame_center) / flameThickness, ss.state.mass_fractions[:, iOH] * 10, "k--s")
        iO2 = gas.species_index("O2")
        plt.plot(
            (flame.grid - flame_center) / flameThickness,
            flame.Y[iO2, :],
            "g",
            label=r"$Y_\mathrm{O_2}$",
        )
        plt.plot((ss.x - flame_center) / flameThickness, ss.state.mass_fractions[:, iO2], "g--s")
        iH2 = gas.species_index("H2")
        plt.plot(
            (flame.grid - flame_center) / flameThickness,
            flame.Y[iH2, :],
            "b",
            label=r"$Y_\mathrm{H_2}$",
        )
        plt.plot((ss.x - flame_center) / flameThickness, ss.state.mass_fractions[:, iH2], "b--s")
        plt.xlabel(r"$x/\delta_\mathrm{F}$")
        plt.legend(loc="best")
        if show_results:
            plt.show()

        if results_location is not None:
            results_location = Path(results_location)
            plt.savefig(results_location / "laminarFlame.pdf")

    results = {
        "position": ss.x,
        "temperature": ss.physics.get_temperature(ss.state),
    }
    if results_location is not None:
        results_location = Path(results_location)
        results_location.mkdir(parents=True, exist_ok=True)
        np.savez(results_location / "laminarFlame.npz", **results)

    return results


def flameSpeed(gas, flameThickness, returnFlame=False):
    """
    Function flameSpeed
    ======================================================================
    This function returns the flame speed for a gas. The cantera implementation
    is quite unstable. Therefore, this function is not very useful
        gas: cantera phase object at the desired state
        flameThickness: a guess on the flame thickness
        return: Sl
    """
    # solution parameters
    width = 5.0 * flameThickness  # m
    loglevel = 0  # amount of diagnostic output (0 to 8)
    # Flame object
    try:
        f = ct.FreeFlame(gas, width=width)
    except Exception:
        f = ct.FreeFlame(gas)
    f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
    # f.show()
    f.transport_model = "mixture-averaged"
    try:
        f.solve(loglevel=loglevel, auto=True)
    except Exception:
        f.solve(loglevel=loglevel)
    # f.show()
    print(f"mixture-averaged flamespeed = {f.velocity[0]:7f} m/s")
    if returnFlame:
        return f.velocity[0], f
    return f.velocity[0]


if __name__ == "__main__":
    main()
