"""
Copyright 2017 Kevin Grogan
Copyright 2024 Matthew Bonanni

This file is part of StanScram.

StanScram is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License.

StanScram is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with StanScram.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

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
    """

    def __init__(self, domain, variable, skipSteps=0, x=None):
        self.name = None
        self.skipSteps = 0

        self.name = variable.lower()
        self.skipSteps = skipSteps  # number of timesteps to skip
        # check interpolation grid
        if x is None:
            self.x = domain.x
        elif (x[-1] > domain.x[-1]) or (x[0] < domain.x[0]):
            msg = "Invalid Interpolation Grid"
            raise Exception(msg)
        else:
            self.x = x

        self.variable = []  # list of numpy arrays of the variable w.r.t x
        self.t = []  # list of times

        self.update(domain)

    def update(self, domain):
        """
        This method updates the XT diagram.
            inputs:
                XTDiagram: the XTDiagram object
        """
        variable = self.name
        if domain.physics == "FPV":
            scalarNames = ["mixture fraction", "progress variable"]
        elif domain.physics == "FRC":
            scalarNames = [species.lower() for species in domain.gas.species_names]
        if variable in ["density", "r", "rho"]:
            self.variable.append(np.interp(self.x, domain.x, domain.r))
        elif variable in ["velocity", "u"]:
            self.variable.append(np.interp(self.x, domain.x, domain.u))
        elif variable in ["pressure", "p"]:
            self.variable.append(np.interp(self.x, domain.x, domain.p))
        elif variable in ["temperature", "t"]:
            T = domain.getTemperature(domain.r, domain.p, domain.Y)
            self.variable.append(np.interp(self.x, domain.x, T))
        elif variable in ["gamma", "g", "specific heat ratio", "heat capacity ratio"]:
            self.variable.append(np.interp(self.x, domain.x, domain.gamma))
        elif variable in scalarNames:
            scalarIndex = scalarNames.index(variable)
            self.variable.append(np.interp(self.x, domain.x, domain.Y[:, scalarIndex]))
        else:
            msg = f"Invalid Variable Name: {variable}"
            raise Exception(msg)
        self.t.append(domain.t)

    def plot(self, limits=None):
        """
        This method creates a contour plot of the XTDiagram data
            inputs:
                XTDiagram=XTDiagram object; obtained from the XTDiagrams dictionary
                limits = tuple of maximum and minimum for the pcolor (vMin,vMax)

        """
        plt.figure()
        t = [t * 1000.0 for t in self.t]
        X, T = np.meshgrid(self.x, t)
        variableMatrix = np.zeros(X.shape)
        for k, variablek in enumerate(self.variable):
            variableMatrix[k, :] = variablek
        variable = self.name
        if variable in ["density", "r", "rho"]:
            plt.title(r"$\rho [\mathrm{kg/m^3}]$")
        elif variable in ["velocity", "u"]:
            plt.title(r"$u [\mathrm{m/s}]$")
        elif variable in ["pressure", "p"]:
            variableMatrix /= 1.0e5  # convert to bar
            plt.title(r"$p [\mathrm{bar}]$")
        elif variable in ["temperature", "t"]:
            plt.title(r"$T [\mathrm{K}]$")
        elif variable in ["gamma", "g", "specific heat ratio", "heat capacity ratio"]:
            plt.title(r"$\gamma$")
        else:
            plt.title(r"$\mathrm{" + variable + "}$")
        if limits is None:
            plt.pcolormesh(X, T, variableMatrix, cmap="jet")
        else:
            plt.pcolormesh(
                X, T, variableMatrix, cmap="jet", vmin=limits[0], vmax=limits[1]
            )
        plt.xlabel(r"$x [\mathrm{m}]$")
        plt.ylabel(r"$t [\mathrm{ms}]$")
        plt.axis([min(self.x), max(self.x), min(t), max(t)])
        plt.colorbar()


def add_h_plot(domain, ax, scale=1.0):
    ax1 = ax.twinx()
    ax1.set_zorder(-np.inf)
    ax.patch.set_visible(False)
    ax1.plot(domain.x * scale, domain.h * scale, color="0.8", linestyle="--")
    ax1.axhline(0, color="0.8", linestyle="--")
    ax1.set_aspect("equal")
    ax1.set_ylabel("h [mm]")
    return ax1


def plot_state(domain, filename):
    xscale = 1.0e3
    T = domain.getTemperature(domain.r, domain.p, domain.Y)

    fig, ax = plt.subplots(7, 1, sharex=True, figsize=(6, 9))
    ax[0].plot(domain.x * xscale, domain.r)
    ax[0].set_ymargin(0.1)
    ax[0].set_ylabel(r"$\rho$ [kg/m$^3$]")
    if domain.h is not None:
        add_h_plot(domain, ax[0], scale=xscale)

    ax[1].plot(domain.x * xscale, domain.u)
    ax[1].set_ymargin(0.1)
    ax[1].set_ylabel(r"$u$ [m/s]")
    if domain.h is not None:
        add_h_plot(domain, ax[1], scale=xscale)

    ax[2].plot(domain.x * xscale, domain.p)
    ax[2].set_ymargin(0.1)
    ax[2].set_ylabel(r"$p$ [Pa]")
    if domain.h is not None:
        add_h_plot(domain, ax[2], scale=xscale)

    ax[3].plot(domain.x * xscale, T)
    ax[3].set_ymargin(0.1)
    ax[3].set_ylabel(r"$T$ [K]")
    if domain.h is not None:
        add_h_plot(domain, ax[3], scale=xscale)

    M = np.abs(domain.u) / domain.soundSpeed(domain.r, domain.p, domain.gamma)
    ax[4].plot(domain.x * xscale, M)
    ax[4].axhline(1.0, color="r", linestyle="--")
    ax[4].set_ymargin(0.1)
    ax[4].set_ylabel(r"$M$ [-]")
    if domain.h is not None:
        add_h_plot(domain, ax[4], scale=xscale)

    if domain.physics == "FPV":
        Z = domain.Y[:, 0]
        C = domain.Y[:, 1]
        Q = np.zeros_like(domain.x)
        L = domain.fpv_table.L_from_C(Z, C)
        Y_H2 = domain.fpv_table.lookup("H2", Z, Q, L)
        Y_OH = domain.fpv_table.lookup("OH", Z, Q, L)
        Y_H2O = domain.fpv_table.lookup("H2O", Z, Q, L)
    elif domain.physics == "FRC":
        Y_H2 = domain.Y[:, domain.gas.species_index("H2")]
        Y_OH = domain.Y[:, domain.gas.species_index("OH")]
        Y_H2O = domain.Y[:, domain.gas.species_index("H2O")]
    ax[5].plot(domain.x * xscale, Y_H2, label=r"$\mathrm{H}_2$")
    ax[5].plot(domain.x * xscale, Y_OH, label=r"$\mathrm{OH}$")
    ax[5].plot(domain.x * xscale, Y_H2O, label=r"$\mathrm{H}_2\mathrm{O}$")
    if Y_H2.max() < 1e-6:
        ax[5].set_ylim(-1e-3, 1e-3)
    else:
        ax[5].set_ymargin(0.1)
    ax[5].set_ylabel(r"$Y_k$ [-]")
    ax[5].legend(loc="upper left")
    if domain.h is not None:
        add_h_plot(domain, ax[5], scale=xscale)

    ax[6].scatter(
        domain.injector.fluid_tips[:, 0] * xscale, domain.injector.fluid_tips[:, 1], s=1
    )
    ax[6].set_ymargin(0.1)
    ax[6].set_ylabel(r"$\dot{m}$ [kg/s]")
    if domain.h is not None:
        add_h_plot(domain, ax[6], scale=xscale)

    ax[6].set_xlabel("x [mm]")

    fig.suptitle(rf"$t = {domain.t * 1.0e3:.4f}$ ms")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
