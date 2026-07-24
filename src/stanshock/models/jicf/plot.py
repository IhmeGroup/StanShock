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


def _contour_plot(x: Array, y: Array, z: Array, opts: PlotOptions) -> None:
    fig, ax = plt.subplots(figsize=opts.figsize)

    vmin = z.min()
    vmax = z.max()
    cs = ax.contourf(x, y, z, 256, cmap="inferno", vmin=vmin, vmax=vmax)

    norm = Normalize(vmin=vmin, vmax=vmax)
    axcb = fig.colorbar(cs, fraction=0.046, pad=0.01).ax
    _cb = ColorbarBase(axcb, cmap=plt.get_cmap("inferno"), norm=norm)

    ax.set_aspect("equal", "box")
    ax.set_xlabel(opts.xlabel)
    ax.set_ylabel(opts.ylabel)
    ax.set_title(opts.title)

    fig.tight_layout()
    fig.savefig(opts.filename)
    plt.close(fig)


def plot_jicf_flowfield(jicf: JICModel, plot_dir: Path = Path("plots/jicf")) -> None:
    """Generate contour plots of the JICF flow field"""
    plot_dir.mkdir(parents=True, exist_ok=True)
    x = jicf.x_3D_data
    y = jicf.y_3D_data
    z = jicf.z_3D_data
    Z = jicf.Z_3D_data

    # Generate plot along injector centerlines
    z_inj = jicf.analytic.z_inj

    opts = PlotOptions(
        "Centerline Mixture Fraction",
        "x [m]",
        "y [m]",
        plot_dir / "Z_centerline.png",
        (38.4, 3.2),
    )
    for i in range(len(z_inj)):
        iz_inj = np.argmin(np.abs(z - z_inj[i]))
        for iJ in range(jicf.nJ):
            opts.title = f"Jet Centerline Mixture Fraction at z={z_inj[i]:.3f}, J={jicf.analytic.J[iJ]:.2f}"
            opts.filename = plot_dir / f"Z_cl_z{i:02d}_J{iJ:02d}.png"
            _contour_plot(x, y, Z[:, :, iz_inj, iJ].T, opts)

    # Generate plots in transverse slices
    nx = len(x)
    ix_inj = np.argmin(np.abs(x - jicf.x_inj))
    n_plot = 5
    di = (nx - ix_inj) // (n_plot - 1)
    if n_plot * di + ix_inj > nx - 1:
        di -= 1

    opts.xlabel = "z [m]"
    opts.figsize = (38.4, 6.4)
    for i in range(n_plot):
        ix = ix_inj + i * di
        for iJ in range(jicf.nJ):
            opts.title = f"Mixture Fraction at x={x[ix]:.3f}, J={jicf.analytic.J[iJ]:.2f}"
            opts.filename = plot_dir / f"Z_x{i:02d}_J{iJ:02d}.png"
            _contour_plot(z, y, Z[ix, :, :, iJ], opts)
