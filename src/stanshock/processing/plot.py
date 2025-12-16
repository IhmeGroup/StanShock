from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from stanshock.system.geometry import AsymmetricBox, Box, Cylinder

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from stanshock.components.combustor import Combustor
    from stanshock.system.backend import Array


class XTDiagram:
    """
    This class is used to store the relevant data for the XT diagram
        inputs:
            domain=component to be plotted
            variable=string of the variable
            skipSteps=number of iterations between updates
            x=mesh for plotting
            limits = tuple of maximum and minimum for the pcolor (vMin,vMax)
    """

    def __init__(
        self,
        domain: Combustor,
        variable: str,
        skipSteps: int = 0,
        x: Array | None = None,
        limits: tuple[float, float] | None = None,
    ) -> None:
        self.skipSteps = 0
        self.limits = limits

        self.name = variable.lower()
        self.skipSteps = skipSteps  # number of timesteps to skip
        # check interpolation grid
        geometry = domain.geometry
        if x is None:
            self.x = geometry.xf
        elif (x[-1] > geometry.xc[-1]) or (x[0] < geometry.xc[0]):
            msg = "Invalid Interpolation Grid"
            raise Exception(msg)
        else:
            self.x = x

        self.variable: list[Array] = []  # list of numpy arrays of the variable w.r.t x
        self.t: list[float] = []  # list of times
        self.mdot: list[float] = []  # list of mass flow rates

    def update(self, domain: Combustor) -> None:
        """
        This method updates the XT diagram.
            inputs:
                XTDiagram: the XTDiagram object
        """
        variable = self.name
        state = domain.state
        geometry = domain.geometry

        if variable in ["density", "r", "rho"]:
            assert state.density is not None
            self.variable.append(np.interp(self.x, geometry.xc, state.density))
        elif variable in ["velocity", "u"]:
            assert state.velocity is not None
            self.variable.append(np.interp(self.x, geometry.xc, state.velocity))
        elif variable in ["pressure", "p"]:
            assert state.pressure is not None
            self.variable.append(np.interp(self.x, geometry.xc, state.pressure))
        elif variable in ["temperature", "t"]:
            T = domain.physics.get_temperature(state)
            self.variable.append(np.interp(self.x, geometry.xc, T))
        elif variable in ["gamma", "g", "specific heat ratio", "heat capacity ratio"]:
            assert state.gamma is not None
            self.variable.append(np.interp(self.x, geometry.xc, state.gamma))
        elif variable in domain.physics.scalar_names:
            assert state.composition is not None
            scalarIndex = domain.physics.scalar_names.index(variable)
            self.variable.append(
                np.interp(self.x, geometry.xc, state.composition[:, scalarIndex])
            )
        elif variable in ["mach", "m"]:
            assert state.velocity is not None
            M = np.abs(state.velocity) / domain.physics.get_sound_speed(state)
            self.variable.append(np.interp(self.x, geometry.xc, M))
        else:
            msg = f"Invalid Variable Name: {variable}"
            raise Exception(msg)
        self.t.append(domain.t)
        if domain.injector is not None:
            self.mdot.append(float(domain.injector.mdot_f_interp(domain.t)))

    def plot(self, figdir: Path | str = ".") -> None:
        """
        This method creates a contour plot of the XTDiagram data
            inputs:
                figdir = directory in which to save the plot
        """
        t = [t * 1000.0 for t in self.t]
        mdot = [mdot * 1.0e3 for mdot in self.mdot]
        X, T = np.meshgrid(self.x, t)
        variableMatrix = np.zeros(X.shape)
        for k, variablek in enumerate(self.variable):
            variableMatrix[k, :] = variablek
        variable = self.name
        if variable in ["density", "r", "rho"]:
            title = r"$\rho~[\mathrm{kg/m^3}]$"
        elif variable in ["velocity", "u"]:
            title = r"$u~[\mathrm{m/s}]$"
        elif variable in ["pressure", "p"]:
            variableMatrix /= 1.0e5  # convert to bar
            title = r"$p~[\mathrm{bar}]$"
        elif variable in ["temperature", "t"]:
            title = r"$T~[\mathrm{K}]$"
        elif variable in ["gamma", "g", "specific heat ratio", "heat capacity ratio"]:
            title = r"$\gamma~[\mathrm{-}]$"
        elif variable in ["mixture fraction"]:
            title = r"$Z~[\mathrm{-}]$"
        elif variable in ["progress variable"]:
            title = r"$C~[\mathrm{-}]$"
        elif variable in ["mach", "m"]:
            title = r"$M~[\mathrm{-}]$"
        else:
            title = r"$\mathrm{" + variable + "}$"

        has_mdot = mdot and any(mdot)
        figsize = (6, 4) if has_mdot else (6, 3)

        fig: Figure = plt.figure(figsize=figsize)
        if has_mdot:
            gs = fig.add_gridspec(1, 3, width_ratios=[6, 1, 0.5], wspace=0.125)
            main_ax = fig.add_subplot(gs[0, 0])
            mdot_ax = fig.add_subplot(gs[0, 1], sharey=main_ax)
            cbar_ax = fig.add_subplot(gs[0, 2])
        else:
            main_ax = plt.gca()

        fig.suptitle(title, fontsize=14)

        if self.limits is None:
            pcm = main_ax.pcolormesh(X, T, variableMatrix, cmap="jet")
        else:
            pcm = main_ax.pcolormesh(
                X,
                T,
                variableMatrix,
                cmap="jet",
                vmin=self.limits[0],
                vmax=self.limits[1],
            )

        main_ax.set_xlabel(r"$x~[\mathrm{m}]$")
        main_ax.set_ylabel(r"$t~[\mathrm{ms}]$")
        main_ax.set_xlim(min(self.x), max(self.x))
        main_ax.set_ylim(min(t), max(t))

        if has_mdot:
            fig.colorbar(pcm, cax=cbar_ax)
        else:
            fig.colorbar(pcm, ax=main_ax)

        if has_mdot:
            mdot_ax.plot(mdot, t, "r-", linewidth=2)
            mdot_ax.set_xlabel(r"$\dot{m}~[\mathrm{g/s}]$")
            mdot_ax.set_yticklabels([])
            mdot_ax.grid(True, axis="x", linestyle="--", alpha=0.7)

        fig.tight_layout()
        fig.savefig(Path(figdir) / f"{variable}.png", bbox_inches="tight", dpi=300)


def add_h_plot(domain: Combustor, ax: Axes, scale: float = 1.0e3) -> Axes:
    ax1 = ax.twinx()
    ax1.set_zorder(-np.inf)
    ax.patch.set_visible(False)

    geometry = domain.geometry
    t = domain.t
    x = geometry.xf

    yname = "h"
    if isinstance(geometry, Cylinder):
        d = geometry.d_outer(t, x)
        ax1.plot(x * scale, d * scale, color="0.8", linestyle="--")
        yname = r"$d_{outer}$"
    elif isinstance(geometry, Box):
        ax1.plot(x * scale, geometry.h(t, x) * scale, color="0.8", linestyle="--")
        ax1.axhline(0, color="0.8", linestyle="--")
    elif isinstance(geometry, AsymmetricBox):
        ax1.plot(
            x * scale, geometry.upper_wall(t, x) * scale, color="0.8", linestyle="--"
        )
        ax1.plot(
            x * scale, geometry.lower_wall(t, x) * scale, color="0.8", linestyle="--"
        )

    ax1.set_aspect("equal")
    ax1.set_ylabel(f"{yname} [mm]")
    return ax1


def plot_state(
    domain: Combustor, filename: Path | str, plot_geometry: bool = True
) -> None:
    xscale = 1.0e3
    physics = domain.physics
    state = domain.state
    assert state.density is not None
    assert state.velocity is not None
    assert state.pressure is not None
    geometry = domain.geometry
    idx_cells = geometry.idx_cells
    x = xscale * geometry.xc[idx_cells]
    T = physics.get_temperature(state)
    nrows = 7 if domain.injector is not None else 6

    fig: Figure
    axs: list[Axes]
    fig, axs = plt.subplots(nrows, 1, sharex=True, figsize=(6, 9))
    axs[0].plot(x, state.density[idx_cells])
    axs[0].set_ymargin(0.1)
    axs[0].set_ylabel(r"$\rho$ [kg/m$^3$]")

    axs[1].plot(x, state.velocity[idx_cells])
    axs[1].set_ymargin(0.1)
    axs[1].set_ylabel(r"$u$ [m/s]")

    axs[2].plot(x, state.pressure[idx_cells])
    axs[2].set_ymargin(0.1)
    axs[2].set_ylabel(r"$p$ [Pa]")

    axs[3].plot(x, T[idx_cells])
    axs[3].set_ymargin(0.1)
    axs[3].set_ylabel(r"$T$ [K]")

    M = np.abs(state.velocity) / physics.get_sound_speed(state)
    axs[4].plot(x, M[idx_cells])
    axs[4].axhline(1.0, color="r", linestyle="--")
    axs[4].set_ymargin(0.1)
    axs[4].set_ylabel(r"$M$ [-]")

    if physics.is_flamelet:
        state = physics.set_state(state)
        Y_H2 = physics.lookup("H2", state)[idx_cells]
        Y_OH = physics.lookup("OH", state)[idx_cells]
        Y_H2O = physics.lookup("H2O", state)[idx_cells]
    else:
        assert state.mass_fractions is not None
        Y = state.mass_fractions[idx_cells]
        Y_H2 = Y[:, physics.gas.species_index("H2")]
        Y_OH = Y[:, physics.gas.species_index("OH")]
        Y_H2O = Y[:, physics.gas.species_index("H2O")]
    axs[5].plot(x, Y_H2, label=r"$\mathrm{H}_2$")
    axs[5].plot(x, Y_OH, label=r"$\mathrm{OH}$")
    axs[5].plot(x, Y_H2O, label=r"$\mathrm{H}_2\mathrm{O}$")
    if Y_H2.max() < 1e-6:
        axs[5].set_ylim(-1e-3, 1e-3)
    else:
        axs[5].set_ymargin(0.1)
    axs[5].set_ylabel(r"$Y_k$ [-]")
    axs[5].legend(loc="upper right")

    if domain.injector is not None:
        axs[6].scatter(
            domain.injector.fluid_tips[:, 0] * xscale,
            domain.injector.fluid_tips[:, 1] * 1e3 * domain.injector.n_inj,
            s=1,
        )
        axs[6].set_ymargin(0.1)
        axs[6].set_ylabel(r"$\dot{m}_f$ [g/s]")

        axs[6].set_xlabel("x [mm]")

    if plot_geometry:
        for ax in axs:
            add_h_plot(domain, ax, scale=xscale)

    fig.suptitle(rf"$t = {domain.t * 1.0e3:.4f}$ ms")

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
