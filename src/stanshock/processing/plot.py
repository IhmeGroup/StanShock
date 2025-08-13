from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

    def __init__(self, domain, variable, skipSteps=0, x=None, limits=None):
        self.name = None
        self.skipSteps = 0
        self.limits = limits

        self.name = variable.lower()
        self.skipSteps = skipSteps  # number of timesteps to skip
        # check interpolation grid
        geometry = domain.geometry
        if x is None:
            self.x = geometry.x
        elif (x[-1] > geometry.x[-1]) or (x[0] < geometry.x[0]):
            msg = "Invalid Interpolation Grid"
            raise Exception(msg)
        else:
            self.x = x

        self.variable = []  # list of numpy arrays of the variable w.r.t x
        self.t = []  # list of times
        self.mdot = []  # list of mass flow rates

        self.update(domain)

    def update(self, domain):
        """
        This method updates the XT diagram.
            inputs:
                XTDiagram: the XTDiagram object
        """
        variable = self.name
        state = domain.state
        geometry = domain.geometry
        idx_cells = domain.idx_cells

        if variable in ["density", "r", "rho"]:
            self.variable.append(
                np.interp(self.x, geometry.x, state.density[idx_cells])
            )
        elif variable in ["velocity", "u"]:
            self.variable.append(
                np.interp(self.x, geometry.x, state.velocity[idx_cells])
            )
        elif variable in ["pressure", "p"]:
            self.variable.append(
                np.interp(self.x, geometry.x, state.pressure[idx_cells])
            )
        elif variable in ["temperature", "t"]:
            T = domain.physics.get_temperature(state)
            self.variable.append(np.interp(self.x, geometry.x, T[idx_cells]))
        elif variable in ["gamma", "g", "specific heat ratio", "heat capacity ratio"]:
            self.variable.append(np.interp(self.x, geometry.x, state.gamma[idx_cells]))
        elif variable in domain.physics.scalar_names:
            scalarIndex = domain.physics.scalar_names.index(variable)
            self.variable.append(
                np.interp(self.x, geometry.x, state.composition[idx_cells, scalarIndex])
            )
        elif variable in ["mach", "m"]:
            M = np.abs(state.velocity) / domain.physics.get_sound_speed(state)
            self.variable.append(np.interp(self.x, self.x, M[idx_cells]))
        else:
            msg = f"Invalid Variable Name: {variable}"
            raise Exception(msg)
        self.t.append(domain.t)
        if domain.injector is not None:
            self.mdot.append(domain.injector.mdot_f_interp(domain.t))

    def plot(self, figdir="."):
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

        fig = plt.figure(figsize=figsize)
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

        plt.tight_layout()
        plt.savefig(Path(figdir) / f"{variable}.png", bbox_inches="tight", dpi=300)


def add_h_plot(domain, ax, scale=1.0):
    ax1 = ax.twinx()
    ax1.set_zorder(-np.inf)
    ax.patch.set_visible(False)

    geometry = domain.geometry
    t = domain.t
    x = geometry.x
    h = geometry.h(t, x) if geometry.h is not None else geometry.d_outer(t, x)
    ax1.plot(x * scale, h * scale, color="0.8", linestyle="--")
    ax1.axhline(0, color="0.8", linestyle="--")
    ax1.set_aspect("equal")
    ax1.set_ylabel("h [mm]")
    return ax1


def plot_state(domain, filename):
    xscale = 1.0e3
    physics = domain.physics
    state = domain.state
    geometry = domain.geometry
    idx_cells = domain.idx_cells
    T = physics.get_temperature(state)

    fig, ax = plt.subplots(7, 1, sharex=True, figsize=(6, 9))
    ax[0].plot(geometry.x * xscale, state.density[idx_cells])
    ax[0].set_ymargin(0.1)
    ax[0].set_ylabel(r"$\rho$ [kg/m$^3$]")
    if geometry.h is not None:
        add_h_plot(domain, ax[0], scale=xscale)

    ax[1].plot(geometry.x * xscale, state.velocity[idx_cells])
    ax[1].set_ymargin(0.1)
    ax[1].set_ylabel(r"$u$ [m/s]")
    if geometry.h is not None:
        add_h_plot(domain, ax[1], scale=xscale)

    ax[2].plot(geometry.x * xscale, state.pressure[idx_cells])
    ax[2].set_ymargin(0.1)
    ax[2].set_ylabel(r"$p$ [Pa]")
    if geometry.h is not None:
        add_h_plot(domain, ax[2], scale=xscale)

    ax[3].plot(geometry.x * xscale, T[idx_cells])
    ax[3].set_ymargin(0.1)
    ax[3].set_ylabel(r"$T$ [K]")
    if geometry.h is not None:
        add_h_plot(domain, ax[3], scale=xscale)

    M = np.abs(state.velocity) / physics.get_sound_speed(state)
    ax[4].plot(geometry.x * xscale, M[idx_cells])
    ax[4].axhline(1.0, color="r", linestyle="--")
    ax[4].set_ymargin(0.1)
    ax[4].set_ylabel(r"$M$ [-]")
    if geometry.h is not None:
        add_h_plot(domain, ax[4], scale=xscale)

    if physics.is_flamelet:
        state = physics.set_state(state)
        Y_H2 = physics.lookup("H2", state)[idx_cells]
        Y_OH = physics.lookup("OH", state)[idx_cells]
        Y_H2O = physics.lookup("H2O", state)[idx_cells]
    else:
        Y = state.mass_fractions[idx_cells]
        Y_H2 = Y[:, physics.gas.species_index("H2")]
        Y_OH = Y[:, physics.gas.species_index("OH")]
        Y_H2O = Y[:, physics.gas.species_index("H2O")]
    ax[5].plot(geometry.x * xscale, Y_H2, label=r"$\mathrm{H}_2$")
    ax[5].plot(geometry.x * xscale, Y_OH, label=r"$\mathrm{OH}$")
    ax[5].plot(geometry.x * xscale, Y_H2O, label=r"$\mathrm{H}_2\mathrm{O}$")
    if Y_H2.max() < 1e-6:
        ax[5].set_ylim(-1e-3, 1e-3)
    else:
        ax[5].set_ymargin(0.1)
    ax[5].set_ylabel(r"$Y_k$ [-]")
    ax[5].legend(loc="upper right")
    if geometry.h is not None:
        add_h_plot(domain, ax[5], scale=xscale)

    ax[6].scatter(
        domain.injector.fluid_tips[:, 0] * xscale,
        domain.injector.fluid_tips[:, 1] * 1e3 * domain.injector.n_inj,
        s=1,
    )
    ax[6].set_ymargin(0.1)
    ax[6].set_ylabel(r"$\dot{m}_f$ [g/s]")
    if geometry.h is not None:
        add_h_plot(domain, ax[6], scale=xscale)

    ax[6].set_xlabel("x [mm]")

    fig.suptitle(rf"$t = {domain.t * 1.0e3:.4f}$ ms")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
