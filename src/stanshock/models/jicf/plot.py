from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
    plot_dir: Path = Path("figures/jicf"),
    plot_all_inj: bool = False,
    plot_all_J: bool = True,
    plot_centerlines: bool = True,
    truncate_plot: bool = True,
) -> None:
    """Generate contour plots of the JICF flow field"""
    plot_dir.mkdir(parents=True, exist_ok=True)
    x = jicf.x_3D_data
    y = jicf.y_3D_data
    z = jicf.z_3D_data
    Z = jicf.Z_3D_data

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
    nx = len(x)
    ix_inj = np.argmin(np.abs(x - jicf.x_inj))

    # Compute jet centerline profiles
    jicf.analytic.i_m = slice(None)
    x_cl = np.linspace(0.0, x[-1] - jicf.x_inj, 1001)
    y_cl = jicf.analytic.y_cl(x_cl[:, None])
    x_cl += jicf.x_inj
    y_cl[y_cl > y[-1]] = np.nan

    # Get the momentum flux ratios to plot over
    nJ = jicf.nJ
    J = jicf.analytic.J
    if not plot_all_J:
        nstep = max(len(J) // 5, 1)
        J = J[::nstep]
        nJ = len(J)
        x_cl = x_cl[:, ::nstep]
        y_cl = y_cl[:, ::nstep]

    # Generate plot along injector centerlines
    opts = PlotOptions(
        "Centerline Mixture Fraction",
        "x [m]",
        "y [m]",
        plot_dir / "Z_centerline.png",
        (19.2, 6.4) if truncate_plot else (38.4, 4.0),
    )

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

    # Generate plots in transverse slices
    n_plot = 5
    di = (nx - ix_inj) // (n_plot - 1)
    if n_plot * di + ix_inj > nx - 1:
        di -= 1

    opts.xlabel = "z [m]"
    opts.figsize = (38.4, 6.4)
    for i in range(n_plot):
        ix = ix_inj + i * di
        for iJ in range(jicf.nJ):
            opts.title = (
                f"Mixture Fraction at x={x[ix]:.3f}, J={jicf.analytic.J[iJ]:.2f}"
            )
            opts.filename = plot_dir / f"Z_x{i:02d}_J{iJ:02d}.png"
            _contour_plot(z, y, Z[ix, :, :, iJ], opts)
