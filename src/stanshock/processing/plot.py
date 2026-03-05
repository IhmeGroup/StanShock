from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import matplotlib.pyplot as plt
import numpy as np

from stanshock.physics.flamelet import FPVTable
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.geometry import AsymmetricBox, Box, Cylinder

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from stanshock.components.combustor import Combustor
    from stanshock.system.backend import Array


@dataclass
class VariableInfo:
    short_name: str
    plot_label: str
    fun: Callable[[FluidState], Array]
    scale: float = 1.0
    fmt: str = "%.4e"


VariableInfoMap: TypeAlias = dict[str, VariableInfo]


def get_variable_info_map(physics: FluidPhysics) -> VariableInfoMap:
    """Create mapping between strings and variable information."""
    # Common variables available to all FluidPhysics:
    plot_variables: VariableInfoMap = {
        "density": VariableInfo(
            short_name="r",
            plot_label=r"$\rho~[\mathrm{kg/m^3}]$",
            fun=physics.get_density,
        ),
        "velocity": VariableInfo(
            short_name="u", plot_label=r"$u~[\mathrm{m/s}]$", fun=physics.get_velocity
        ),
        "pressure": VariableInfo(
            short_name="p",
            plot_label=r"$p~[\mathrm{bar}]$",
            fun=physics.get_pressure,
            scale=1e-5,
        ),
        "temperature": VariableInfo(
            short_name="T", plot_label=r"$T~[\mathrm{K}]$", fun=physics.get_temperature
        ),
        "gamma": VariableInfo(
            short_name="g", plot_label=r"$\gamma~[\mathrm{-}]$", fun=physics.get_gamma
        ),
        "sound speed": VariableInfo(
            short_name="a",
            plot_label=r"$a~[\mathrm{m/s}]$",
            fun=physics.get_sound_speed,
        ),
        "mach": VariableInfo(
            short_name="m",
            plot_label=r"$Ma~[\mathrm{-}]$",
            fun=lambda x: np.abs(physics.get_velocity(x)) / physics.get_sound_speed(x),
        ),
    }

    # All transported scalars
    short_name_map = {"mixture fraction": "Z", "progress variable": "C"}
    for iscalar, scalar in enumerate(physics.scalar_names):
        if scalar == "density":
            continue

        def get_scalar(state: FluidState, idx: int = iscalar) -> Array:
            assert state.composition is not None
            return state.composition[:, idx]

        # Check for species mass fractions
        name = short_name_map.get(scalar, scalar)
        fancy_name = name if scalar in short_name_map else rf"\mathrm{{{name}}}"
        if scalar in physics.gas.species_names:
            name = f"Y_{scalar}"

            # Fancy formatting for species names
            groups = re.split(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=\D)", scalar)
            fancy_name = "".join(
                [f"_{{{group}}}" if group.isnumeric() else group for group in groups]
            )

        plot_variables[scalar] = VariableInfo(
            short_name=name,
            plot_label=rf"${fancy_name}~[\mathrm{{-}}]$",
            fun=get_scalar,
        )

    # Physics-specific variables
    if isinstance(physics, FPVTable):
        for var in physics.variables:

            def get_lookup(state: FluidState, var: str = var) -> Array:
                return physics.lookup(var, state)

            # Check for species mass fractions
            name = var
            fancy_name = rf"$\mathrm{{{var}}}$"
            if var in physics.gas.species_names:
                name = f"Y_{var}"

                # Fancy formatting for species names
                groups = re.split(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=\D)", var)
                fancy_name = "".join(
                    [
                        f"_{{{group}}}" if group.isnumeric() else group
                        for group in groups
                    ]
                )
                fancy_name = rf"${{{fancy_name}}}~[\mathrm{{-}}]$"

            plot_variables[var] = VariableInfo(
                short_name=name,
                plot_label=fancy_name,
                fun=get_lookup,
            )

            def get_L(state: FluidState) -> Array:
                if state.normalized_progress_variable is None:
                    state = physics.set_state(state)
                assert state.normalized_progress_variable is not None
                return state.normalized_progress_variable

            plot_variables["normalized progress variable"] = VariableInfo(
                short_name="L", plot_label=r"$L~[\mathrm{-}]$", fun=get_L
            )

    else:
        if physics.ox_def is not None and physics.fuel_def is not None:

            def get_Z(state: FluidState) -> Array:
                Y = physics.get_mass_fractions(state)
                return physics.get_bilger_mixture_fraction(Y)

            plot_variables["mixture fraction"] = VariableInfo(
                short_name="Z", plot_label=r"$Z~[\mathrm{-}]$", fun=get_Z
            )

        if physics.prog_def is not None:

            def get_C(state: FluidState) -> Array:
                Y = physics.get_mass_fractions(state)
                return physics.get_progress_variable(Y)

            plot_variables["progress variable"] = VariableInfo(
                short_name="C", plot_label=r"$C~[\mathrm{-}]$", fun=get_C
            )

    # Common aliases:
    plot_variables["rho"] = plot_variables["density"]
    for vname in ["specific heat ratio", "heat capacity ratio"]:
        plot_variables[vname] = plot_variables["gamma"]

    keys = list(plot_variables.keys())
    for key in keys:
        var_info = plot_variables[key]
        plot_variables[var_info.short_name] = var_info

    return plot_variables


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

    interpolator: Callable[[Array], Array] | None

    def __init__(
        self,
        domain: Combustor,
        variable: str,
        variable_info_map: VariableInfoMap | None = None,
        skip_steps: int = 0,  # number of timesteps to skip
        x: Array | None = None,  # mesh to interpolate solution onto
        limits: tuple[float, float] | None = None,  # colormap range
    ) -> None:
        self.name = variable.lower()
        if variable_info_map is None:
            variable_info_map = get_variable_info_map(domain.physics)
        self.variable_info = variable_info_map[self.name]
        self.skip_steps = skip_steps
        self.limits = limits

        # check interpolation grid
        geometry = domain.geometry
        if x is None:
            self.x = geometry.xf
        elif (x[-1] > geometry.xc[-1]) or (x[0] < geometry.xc[0]):
            msg = "Invalid Interpolation Grid"
            raise Exception(msg)
        else:
            self.x = x

        self.interpolator = lambda x: np.interp(self.x, geometry.xc, x)

        self.variable: list[Array] = []  # list of numpy arrays of the variable w.r.t x
        self.t: list[float] = []  # list of times
        self.mdot: list[float] = []  # list of mass flow rates

    def update(self, domain: Combustor) -> None:
        """
        This method updates the XT diagram.
            inputs:
                XTDiagram: the XTDiagram object
        """
        state = domain.state

        value = self.variable_info.fun(state) * self.variable_info.scale
        if self.interpolator is not None:
            value = self.interpolator(value)

        self.variable.append(value)
        self.t.append(domain.t)
        if domain.injectors is not None:
            mdot_f = 0.0
            for inj in domain.injectors:
                mdot_f += float(inj.mdot_f_interp(domain.t))
            self.mdot.append(mdot_f)

    def plot(self, figdir: Path | str = ".") -> None:
        """
        This method creates a contour plot of the XTDiagram data
            inputs:
                figdir = directory in which to save the plot
        """
        t = 1e3 * np.array(self.t)
        X, T = np.meshgrid(self.x, t)
        variableMatrix = np.zeros(X.shape)
        for k, variablek in enumerate(self.variable):
            variableMatrix[k, :] = variablek

        title = self.variable_info.plot_label
        short_name = self.variable_info.short_name

        has_mdot = False
        if self.mdot:
            mdot = 1e3 * np.array(self.mdot)
            has_mdot = np.any(mdot)

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
        fig.savefig(Path(figdir) / f"{short_name}.png", bbox_inches="tight", dpi=300)


def add_h_plot(domain: Combustor, ax: Axes, scale: float = 1.0e3) -> Axes:
    ax1 = ax.twinx()
    ax1.set_zorder(-np.inf)
    ax.patch.set_visible(False)

    geometry = domain.geometry
    t = domain.t
    x = geometry.xf

    yname = "h"
    if isinstance(geometry, Cylinder):
        d = np.broadcast_to(geometry.d_outer(t, x), x.shape)
        ax1.plot(x * scale, d * scale, color="0.8", linestyle="--")
        yname = r"$d_{outer}$"
    elif isinstance(geometry, Box):
        h = np.broadcast_to(geometry.h(t, x), x.shape)
        ax1.plot(x * scale, h * scale, color="0.8", linestyle="--")
        ax1.axhline(0, color="0.8", linestyle="--")
    elif isinstance(geometry, AsymmetricBox):
        upper = np.broadcast_to(geometry.upper_wall(t, x), x.shape)
        lower = np.broadcast_to(geometry.lower_wall(t, x), x.shape)
        ax1.plot(x * scale, upper * scale, color="0.8", linestyle="--")
        ax1.plot(x * scale, lower * scale, color="0.8", linestyle="--")

    if domain.injectors is not None:
        for inj in domain.injectors:
            h_inj = 0.0
            ax1.annotate(
                "",
                xy=(inj.x_inj * scale, h_inj * scale),
                xytext=(inj.x_inj * scale, h_inj * scale - 0.02),
                arrowprops={"arrowstyle": "-|>", "color": "0.5", "lw": 1},
            )
            # ax1.scatter(inj.x_inj * scale, h_inj * scale,
            #             s=10,
            #             c='k',
            #             marker="^")

    ax1.set_xlim(x.min() * scale, x.max() * scale)
    ax1.set_aspect("equal")
    ax1.set_ylabel(f"{yname} [mm]")
    return ax1


def plot_state(
    domain: Combustor,
    filename: Path | str,
    variable_info_map: dict[str, VariableInfo] | None = None,
    plot_geometry: bool = True,
    plot_variables: list[str | list[str]] | None = None,
) -> None:
    xscale = 1.0e3
    geometry = domain.geometry
    idx_cells = geometry.idx_cells
    x = xscale * geometry.xc[idx_cells]
    state = domain.state[idx_cells]

    if variable_info_map is None:
        variable_info_map = get_variable_info_map(domain.physics)

    # Set default variables to plot
    if plot_variables is None:
        plot_variables = ["r", "u", "p", "T", "m"]

        sp_plot = ["Y_H2", "Y_OH", "Y_H2O"]
        if all(sp in variable_info_map for sp in sp_plot):
            plot_variables += [sp_plot]

    nrows: int = len(plot_variables)
    if domain.injectors is not None:
        nrows += 1

    fig: Figure
    axs: list[Axes]
    fig, axs = plt.subplots(nrows, 1, sharex=True, figsize=(6, 9))

    for iax, vnames in enumerate(plot_variables):
        ax = axs[iax]

        if isinstance(vnames, list):
            vmax = 0.0
            Y_labels = True
            plot_labels = []
            for vname in vnames:
                variable_info = variable_info_map[vname]
                value = variable_info.fun(state) * variable_info.scale
                vmax = max(value.max(), vmax)

                legend_label = variable_info.plot_label.split("~")[0] + "$"
                ax.plot(x, value, label=legend_label)
                plot_labels += [variable_info.plot_label]
                if variable_info.short_name[0] != "Y":
                    Y_labels = False

            ax.legend(loc="upper right")
            if Y_labels:
                plot_label = r"$Y_k~[\mathrm{-}]$"
                if vmax < 1e-6:
                    ax.set_ylim(-1e-3, 1e-3)
            else:
                plot_label = ", ".join(plot_labels)
        else:
            variable_info = variable_info_map[vnames]
            ax.plot(x, variable_info.fun(state) * variable_info.scale)
            plot_label = variable_info.plot_label

            if variable_info.short_name == "m":
                ax.axhline(1.0, color="r", linestyle="--")

        ax.set_ymargin(0.1)
        ax.set_ylabel(plot_label)

    ax = axs[-1]
    if domain.injectors is not None:
        phi_tot = 0.0
        for inj in domain.injectors:
            ax.scatter(
                inj.fluid_tips[:, 0] * xscale,
                inj.fluid_tips[:, 1] * 1e3 * inj.n_inj,
                s=1,
            )
            phi_tot += inj.phi_f_interp(domain.t)
        ax.set_title(rf"$\phi={phi_tot:.2f}$")
        ax.set_ymargin(0.1)
        ax.set_ylabel(r"$\dot{m}_f$ [g/s]")
    ax.set_xlabel("x [mm]")

    if plot_geometry:
        for ax in axs:
            add_h_plot(domain, ax, scale=xscale)

    fig.suptitle(rf"$t = {domain.t * 1.0e3:.4f}$ ms")

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
