from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

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
            self.variable.append(np.interp(self.x, geometry.xc, state.density))
        elif variable in ["velocity", "u"]:
            self.variable.append(np.interp(self.x, geometry.xc, state.velocity))
        elif variable in ["pressure", "p"]:
            self.variable.append(np.interp(self.x, geometry.xc, state.pressure))
        elif variable in ["temperature", "t"]:
            T = domain.physics.get_temperature(state)
            self.variable.append(np.interp(self.x, geometry.xc, T))
        elif variable in ["gamma", "g", "specific heat ratio", "heat capacity ratio"]:
            self.variable.append(np.interp(self.x, geometry.xc, state.gamma))
        elif variable in domain.physics.scalar_names:
            scalarIndex = domain.physics.scalar_names.index(variable)
            self.variable.append(
                np.interp(self.x, geometry.xc, state.composition[:, scalarIndex])
            )
        elif variable in ["mach", "m"]:
            M = np.abs(state.velocity) / domain.physics.get_sound_speed(state)
            self.variable.append(np.interp(self.x, geometry.xc, M))
        else:
            msg = f"Invalid Variable Name: {variable}"
            raise Exception(msg)
        self.t.append(domain.t)
        if domain.injectors is not None:

            self.mdot.append(domain.injectors.mdot_f_interp(domain.t))

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
    h = geometry.h(t, x) if geometry.h is not None else geometry.d_outer(t, x)
    ax1.plot(x * scale, h * scale, color="0.5", linestyle="--",linewidth=1.0)
    ax1.axhline(0, color="0.5", linestyle="--",linewidth=1.0)
    ax1.set_aspect("equal")
    ax1.set_ylabel("h [mm]")
    if domain.injectors is not None:
        for inj in domain.injectors:
            h_inj = 0.0
            ax1.annotate("", xy=(inj.x_inj * scale, h_inj * scale), xytext=(inj.x_inj * scale, h_inj * scale - 0.02),
        arrowprops=dict(arrowstyle="-|>", color="0.5",lw=1))
            # ax1.scatter(inj.x_inj * scale, h_inj * scale,
            #             s=10,
            #             c='k',
            #             marker="^")

    return ax1


def build_plot_registry():
    return {
        "rho": {
            "label": r"$\rho$ [kg/m$^3$]",
            "getter": lambda domain: domain.state.density,
        },
        "u": {
            "label": r"$u$ [m/s]",
            "getter": lambda domain: domain.state.velocity,
        },
        "p": {
            "label": r"$p$ [Pa]",
            "getter": lambda domain: domain.state.pressure,
        },
        "T": {
            "label": r"$T$ [K]",
            "getter": lambda domain: domain.physics.get_temperature(domain.state),
        },
        "M": {
            "label": r"$M$ [-]",
            "getter": lambda domain: (
                np.abs(domain.state.velocity)
                / domain.physics.get_sound_speed(domain.state)
            ),
        },
        "Y": {
            "label": r"$Y_k$ [-]",
            "getter": lambda domain: domain.state.mass_fractions,
        },
        "mdot": {
            "label": r"$\dot{m}_f$ [g/s]",
            "getter": lambda domain: domain.injector.fluid_tips,
        },
    }


def plot_state(domain: Combustor, filename: Path | str) -> None:
    xscale = 1.0e3
    physics = domain.physics
    state = domain.state
    geometry = domain.geometry
    idx_cells = geometry.idx_cells
    x = xscale * geometry.xc[idx_cells]
    T = physics.get_temperature(state)
    plot_vars = ["rho", "u", "p", "T", "M"]
    if domain.reacting:
        plot_vars.extend(["Y","mdot"])
    fig, ax = plt.subplots(len(plot_vars), 1, sharex=True, figsize=(6, 9))

    fig: Figure
    ax: list[Axes]
    
    plot_col = 'b'
    
    ax[0].plot(x, state.density[idx_cells],c='b')
    ax[0].set_ylabel(r"$\rho$ [kg/m$^3$]")
    ax[0].set_ylim([0, 0.80])

    if geometry.h is not None:
        add_h_plot(domain, ax[0], scale=xscale)

    ax[1].plot(x, state.velocity[idx_cells],c='b')
    ax[1].set_ylabel(r"$u$ [m/s]")
    if geometry.h is not None:
        add_h_plot(domain, ax[1], scale=xscale)

    ax[2].plot(x, state.pressure[idx_cells]/1e3,c='b')
    ax[2].set_ylabel(r"$p$ [kPa]")
    ax[2].set_ylim([0, 500])
    if geometry.h is not None:
        add_h_plot(domain, ax[2], scale=xscale)

    ax[3].plot(x, T[idx_cells],c='b')
    ax[3].set_ylabel(r"$T$ [K]")
    ax[3].set_ylim([300, 3000])
    if geometry.h is not None:
        add_h_plot(domain, ax[3], scale=xscale)

    M = np.abs(state.velocity) / physics.get_sound_speed(state)
    ax[4].plot(x, M[idx_cells],c=plot_col)
    ax[4].axhline(1.0, color="r", linestyle="--")
    ax[4].set_ylabel(r"$M$ [-]")
    ax[4].set_ylim([0.0, 3.5])
    if geometry.h is not None:
        add_h_plot(domain, ax[4], scale=xscale)
    fuel_vars = ["C2H4", "CH4"]
    fuel_name = "JP-7"
    prog_vars = ["H2O","CO2","CO"]
    prog_labels= [r"$\mathrm{H}_2 \mathrm{O}$",
             r"$\mathrm{CO}_2$",
             r"$\mathrm{CO}$",
    ]
    if domain.reacting:
        if physics.is_flamelet:
            state = physics.set_state(state)
            Y_fuel = np.zeros_like(x)
            for i in fuel_vars:
                Y_fuel+=physics.lookup(i,state)[idx_cells]
            Y_prog = np.zeros((len(prog_vars), len(x)))
            for j, var in enumerate(prog_vars):
                Y_prog[j,:] = physics.lookup(var,state)[idx_cells]
            # Y_H2O = physics.lookup("H2O", state)[idx_cells]
            
        else:
            Y = state.mass_fractions[idx_cells]
            fuel_inds = [physics.gas_species_index(i) for i in fuel_vars]
            Y_fuel = np.sum(Y[fuel_inds])
            Y_H2 = Y[:, physics.gas.species_index("H2")]
            Y_OH = Y[:, physics.gas.species_index("OH")]
            Y_H2O = Y[:, physics.gas.species_index("H2O")]

        ax[5].plot(x,Y_fuel,label=fuel_name)
        for k in range(len(prog_vars)):
            ax[5].plot(x, Y_prog[k,:], label=prog_labels[k])

        if Y_fuel.max() < 1e-6:
            ax[5].set_ylim(-1e-3, 1e-3)
        else:
            ax[5].set_ymargin(0.01)
        ax[5].set_ylabel(r"$Y_k$ [-]")
        ax[5].legend(loc="center left")
        if geometry.h is not None:
            add_h_plot(domain, ax[5], scale=xscale)
        phi_tot = 0
        for inj in domain.injectors:
            ax[6].scatter(inj.fluid_tips[:,0] * xscale,
                          inj.fluid_tips[:,1] * 1e3 * inj.n_inj,
                          s=1)
            phi_tot+=inj.phi_f_interp(domain.t)
        ax[6].set_title(rf"$\phi={phi_tot:.2f}$")
        ax[6].set_ymargin(0.1)
        ax[6].set_ylabel(r"$\dot{m}_f$ [g/s]")
        if geometry.h is not None:
            add_h_plot(domain, ax[6], scale=xscale)
        ax[6].set_xlabel("x [mm]")
    else:
        ax[4].set_xlabel("x [mm]")
    

    fig.suptitle(rf"$t = {domain.t * 1.0e3:.4f}$ ms")

    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
