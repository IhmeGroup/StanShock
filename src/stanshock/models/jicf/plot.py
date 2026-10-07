from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from stanshock.models.jicf.generate import JICModel
    from stanshock.system.backend import Array


@dataclass
class PlotOptions:
    title: str
    xlabel: str
    ylabel: str
    filename: Path
    figsize: tuple[float, float]
    label: str = ""
    xscale: Literal["linear", "log"] = "linear"
    yscale: Literal["linear", "log"] = "linear"


def _line_plot(
    x: Array,
    y: Array,
    opts: PlotOptions,
    extra: dict[str, tuple[Array, Array]] | None = None,
) -> None:
    """Plot variable over a range of secondary values."""
    nJ = y.shape[1]

    fig, ax = plt.subplots(figsize=opts.figsize)
    ax.tick_params(axis="both", which="major")
    ax.set_xscale(opts.xscale)
    ax.set_yscale(opts.yscale)

    labels: list[str] = []
    lines: list[plt.Line2D] = []
    if extra is not None:
        for label, (xl, yl) in extra.items():
            nyl = yl.shape[1]
            for i in range(nyl):
                alpha = 0.8 * (i / (nyl - 1) + 0.25) if nyl > 1 else 1.0
                xli = xl[:, i] if xl.ndim == 2 else xl
                line = ax.plot(xli, yl[:, i], "r--", alpha=alpha, label=label)[0]

            if label:
                labels += [label]
                lines += [line]

    for i in range(nJ):
        alpha = 0.8 * (i / (nJ - 1) + 0.25) if nJ > 1 else 1.0
        ax.set_prop_cycle(None)
        xi = x[:, i] if x.ndim == 2 else x
        line = ax.plot(xi, y[:, i], alpha=alpha)[0]

    if opts.label:
        labels = [opts.label, *labels]
        lines = [line, *lines]

    if labels:
        ax.legend(lines, labels, loc="best")

    ax.set_xlabel(opts.xlabel)
    ax.set_ylabel(opts.ylabel)

    if opts.yscale == "log":
        ymax = np.max(y)
        ylim = ax.get_ylim()
        ymin = max(1e-3 * ymax, ylim[0], 1e-8)
        ax.set_ylim((ymin, ylim[1]))

    if opts.title:
        ax.set_title(opts.title)

    fig.tight_layout()
    fig.savefig(opts.filename)
    plt.close()


def _contour_plot(
    x: Array,
    y: Array,
    z: Array,
    opts: PlotOptions,
    extra: dict[str, tuple[Array, Array]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=opts.figsize)

    vmin = z.min()
    vmax = z.max()
    cs = ax.contourf(x, y, z, 256, cmap="inferno", vmin=vmin, vmax=vmax)

    norm = Normalize(vmin=vmin, vmax=vmax)
    axcb = fig.colorbar(cs, fraction=0.046, pad=0.01).ax
    _cb = ColorbarBase(axcb, cmap=plt.get_cmap("inferno"), norm=norm)

    if extra is not None:
        # Overlay lines on the contour plot, adding a legend if there are more than one
        multi = len(extra) > 1

        for label, (xl, yl) in extra.items():
            ax.plot(xl, yl, "-" if multi else "r--", label=label)

        if multi:
            ax.legend(loc="best")

    ax.set_aspect("equal", "box")
    ax.set_xlabel(opts.xlabel)
    ax.set_ylabel(opts.ylabel)
    ax.set_title(opts.title)

    fig.tight_layout()
    fig.savefig(opts.filename)
    plt.close(fig)


def plot_jicf_flowfield(
    jicf: JICModel,
    rho_ratio: float = 1.0,
    plot_dir: Path = Path("figures/jicf"),
    plot_all_inj: bool = False,
    plot_all_J: bool = False,
    plot_centerlines: bool = True,
    truncate_plot: bool = True,
) -> None:
    """Generate contour plots of the JICF flow field."""
    print("Generating contour plots of the JICF flow field.")
    plot_dir.mkdir(parents=True, exist_ok=True)
    x = jicf.x_3D_data
    y = jicf.y_3D_data
    z = jicf.z_3D_data
    Z = jicf.Z_3D_data * np.sqrt(rho_ratio)

    # Convert local mass ratio to actual mixture fraction
    # phi = Z * jicf.physics.stoich_mass_ratio
    # Z_st = jicf.physics.Z_stoich
    # Z = phi * Z_st / (1.0 - Z_st + phi * Z_st)

    # Get the momentum flux ratios to plot over
    nJ = jicf.nJ
    J = jicf.analytic.J
    nstep = 1
    if not plot_all_J:
        nstep = max(len(J) // 5, 1)
        J = J[::nstep]
        nJ = len(J)
        Z = Z[..., ::nstep]

    # Generate plots in transverse slices
    opts = PlotOptions(
        "Centerline Mixture Fraction",
        "z [m]",
        "y [m]",
        plot_dir / "Z_centerline.png",
        (38.4, 6.4),
    )

    # Get transverse slice locations
    n_plot = 5
    nx = len(x)
    ix_inj = np.argmin(np.abs(x - jicf.x_inj))
    di = (nx - ix_inj) // (n_plot - 1)
    if n_plot * di + ix_inj > nx - 1:
        di -= 1

    for i in range(n_plot):
        ix = ix_inj + i * di
        for iJ in range(nJ):
            opts.title = f"Mixture Fraction at $x/d_{{inj}}$={x[ix] / jicf.d_inj:.3f}, J={J[iJ]:.2f}"
            opts.filename = plot_dir / f"Z_x{i:02d}_J{iJ:02d}.png"
            _contour_plot(z, y, Z[ix, :, :, iJ], opts)

    # Get injector location info for constraining the plots
    x_min = 0.9 * jicf.x_inj + 0.1 * x[0]
    ix_min = np.argmin(np.abs(x - x_min))
    r = 0.9 if truncate_plot else 0.6
    x_max = r * jicf.x_inj + (1.0 - r) * x[-1]
    ix_max = np.argmin(np.abs(x - x_max))
    x = x[ix_min:ix_max]
    Z = Z[ix_min:ix_max]

    z_inj = jicf.analytic.z_inj
    if not plot_all_inj:
        # Keep center injector
        i_mid = len(z_inj) // 2
        z_inj = z_inj[[i_mid]]

    # Compute jet centerline profiles
    jicf.analytic.i_m = slice(None)
    x_cl = np.linspace(0.0, x[-1] - jicf.x_inj, 1001)
    y_cl = jicf.analytic.y_cl(x_cl[:, None])
    x_cl += jicf.x_inj
    y_cl[y_cl > y[-1]] = np.nan
    if not plot_all_J:
        y_cl = y_cl[:, ::nstep]

    # Generate plot along injector centerlines
    opts.xlabel = "x [m]"
    opts.figsize = (19.2, 6.4) if truncate_plot else (38.4, 4.0)

    for i in range(len(z_inj)):
        iz_inj = np.argmin(np.abs(z - z_inj[i]))
        for iJ in range(nJ):
            opts.title = (
                f"Jet Centerline Mixture Fraction at z={z_inj[i]:.3f}, J={J[iJ]:.2f}"
            )
            if plot_centerlines:
                opts.filename = plot_dir / f"Z_z{i:02d}_J{iJ:02d}_cl.png"
                _contour_plot(
                    x, y, Z[:, :, iz_inj, iJ].T, opts, extra={"": (x_cl, y_cl[:, iJ])}
                )
            else:
                opts.filename = plot_dir / f"Z_z{i:02d}_J{iJ:02d}.png"
                _contour_plot(x, y, Z[:, :, iz_inj, iJ].T, opts)


def plot_jicf_mean_variance(
    jicf: JICModel, rho_ratio: float = 1.0, plot_dir: Path = Path("figures/jicf")
) -> None:
    """Plot the mixture fraction mean and variance profiles."""
    print("Plotting mixture fraction mean and variance profiles.")
    jicf.analytic.i_m = slice(None)
    Zmean = jicf.Z_avg_profile * np.sqrt(rho_ratio)
    Zvar = jicf.Z_var_profile * rho_ratio
    x = jicf.x_profile
    x_d = x / jicf.d_inj
    xr_d = (x - jicf.x_inj) / jicf.d_inj

    # Get the theoretical values
    mdot_ratio = jicf.analytic.r_u * np.sqrt(rho_ratio) * jicf.Ae / jicf.A
    phi = mdot_ratio * jicf.physics.stoich_mass_ratio
    Z_st = jicf.physics.Z_stoich
    Zmean_target = phi * Z_st / (1.0 - Z_st + phi * Z_st)
    extra = {"Target": (x_d[[0, -1]], np.tile(Zmean_target[None, :], (2, 1)))}

    # Plot mean mixture fraction vs. J
    opts = PlotOptions(
        "",
        "$x/d_{inj}$",
        "Mean Mixture Fraction",
        plot_dir / "Zmean.png",
        (12.8, 6.4),
    )
    _line_plot(x_d, Zmean, opts, extra)

    # Plot mixture fraction variance vs. J
    opts.ylabel = "Mixture Fraction Variance"
    opts.xscale = "log"
    opts.yscale = "log"
    opts.filename = plot_dir / "Zvar.png"

    Zvar_0 = 1.05 * np.max(Zvar)  # Truncate slope line at this value
    idx = np.logical_and(Zvar_0 * xr_d**2 > 1.0, xr_d > 0.0)
    Zvar_target = xr_d[idx] ** -2
    extra = {"$(x/d)^{-2}$ slope": (x_d[idx], Zvar_target[:, None])}

    idx = xr_d > -1.0
    _line_plot(x_d[idx], Zvar[idx], opts, extra)

    # Debug plots
    _ = jicf.analytic.adjustment_factors
    data = jicf.analytic._aux_adjust_data
    if data is not None:
        x_d = data[0] / jicf.d_inj
        y_d = data[1] / jicf.d_inj

        opts.title = ""
        opts.ylabel = r"Centerline-Normal Planar Area [$m^2$]"
        opts.filename = plot_dir / "debug_area.png"
        _line_plot(x_d, data[2], opts)

        opts.xscale = "linear"
        opts.yscale = "linear"

        opts.title = "Centerline Trajectory"
        opts.ylabel = r"$y/d_{inj}$ [m]"
        opts.filename = plot_dir / "debug_centerline.png"
        h_d = jicf.h / jicf.d_inj
        extra = {"h": (x_d[(0, -1), -1], np.array([[h_d], [h_d]]))}
        _line_plot(x_d, y_d, opts, extra)

        opts.title = ""
        opts.ylabel = r"Volume Under Distribution [-]"
        opts.filename = plot_dir / "debug_curve_volume.png"
        _line_plot(x_d, data[3], opts)
    sys.exit()
