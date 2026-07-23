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
    # ax.tick_params(axis="both", which="major", labelsize="18")

    vmin = z.min()
    vmax = z.max()
    cs = ax.contourf(x, y, z, 256, cmap="inferno", vmin=vmin, vmax=vmax)

    norm = Normalize(vmin=vmin, vmax=vmax)
    axcb = fig.colorbar(cs, fraction=0.046, pad=0.01).ax
    _cb = ColorbarBase(axcb, cmap=plt.get_cmap("inferno"), norm=norm)
    # _cb.ax.tick_params(labelsize=18)

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

    # Generate plot along injector centerline
    z_inj = jicf.analytic.z_inj
    k_center = np.argmin(np.abs(z - z_inj[0]))

    opts = PlotOptions(
        "Centerline Mixture Fraction",
        "x [m]",
        "y [m]",
        plot_dir / "Z_centerline.png",
        (19.2, 6.4),
    )
    for iJ in range(jicf.nJ):
        opts.title = f"Centerline Mixture Fraction, J={jicf.analytic.J[iJ]}"
        _contour_plot(x, y, Z[:, :, k_center, iJ], opts)
