from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from inlet_moc.flow_physics import total_pressure_ratio
from inlet_moc.plot.helpers import PlotBounds
from inlet_moc.plot.solution import (
    plot_moc_soln,
    plot_stream_thrust_average,
)
from inlet_moc.processing.cross_sectional_average import (
    compute_stream_thrust_average,
    save_stream_thrust_average_csv,
)
from inlet_moc.processing.get_streamtube_bounds import get_streamtube_bounds
from inlet_moc.processing.triangulated_solution import TriangulatedSolution

if TYPE_CHECKING:
    from inlet_moc.moc_solution import MOCSolution


DIRECT_AVERAGE_KEYS = ("x", "rho_st", "u_st", "p_st", "a_st", "T_st", "mach")


@dataclass(frozen=True)
class SolutionProcessingResult:
    final_state: np.ndarray
    average_profile: np.ndarray | None
    eta_inlet: float
    p0_loss: float


def process_solution(
    soln: MOCSolution,
    mode: str = "direct",
    nx: int = 100,
    *,
    figdir: str | Path | None = None,
    write_csv: bool = True,
    plot_vars: tuple[str, ...] = ("rho", "p", "M", "T"),
) -> SolutionProcessingResult:
    if not getattr(soln, "nets", []):
        msg = "No solved nets are available for processing."
        raise RuntimeError(msg)

    soln.point_mesh = TriangulatedSolution(soln)
    point_mesh = soln.point_mesh

    output_dir = Path(figdir if figdir is not None else soln.figdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_start = soln.x_start
    x_final = soln.x_final()
    if x_start is None or x_final is None:
        msg = "x_start and x_final are required for processing."
        raise RuntimeError(msg)

    mode = str(mode).lower()
    if mode == "analytical":
        x_q = np.linspace(x_start, x_final, nx)
    elif mode == "direct":
        x_q = np.array([x_start, x_final])
    else:
        msg = "mode must be either 'analytical' or 'direct'."
        raise ValueError(msg)

    x_st, y_cent_st, y_cowl_st = get_streamtube_bounds(soln.inlet, point_mesh, x_q)
    soln._streamtube = (x_st, y_cowl_st, y_cent_st)
    bounds = (x_st, y_cent_st, y_cowl_st)
    avg = compute_stream_thrust_average(point_mesh=point_mesh, bounds=bounds)
    final_state = avg[-1]

    A_cap = abs(y_cent_st[0] - y_cowl_st[0])
    A_0 = abs(soln.inlet.cowl.y[0] - soln.inlet.centerbody.y[0])
    eta_inlet = A_cap / A_0

    p0_amb = soln.p_amb * total_pressure_ratio(soln.M_init, soln.gamma)
    p0_final = final_state[3] * total_pressure_ratio(
        final_state[2] / final_state[4],
        soln.gamma,
    )
    p0_loss = p0_final / p0_amb

    if mode == "analytical":
        x_plot_max = float(x_final) + 0.05
        plot_bounds = PlotBounds.from_solution_with_xmax(soln, x_plot_max)
        plot_label = "_".join(str(name) for name in plot_vars)

        fig_soln, _ = plot_moc_soln(
            soln,
            plot_var=plot_vars,
            show_nets=False,
            show_points=False,
            bounds=plot_bounds,
            point_mesh=point_mesh,
        )
        path_soln = output_dir / f"{plot_label}_{soln.N_idl}_IDL.png"
        fig_soln.savefig(path_soln, dpi=500, bbox_inches="tight")
        plt.close(fig_soln)

        fig_net, _ = plot_moc_soln(
            soln,
            plot_var=None,
            show_nets=True,
            show_points=False,
            bounds=plot_bounds,
        )

        path_net = output_dir / f"net_{soln.N_idl}_IDL.png"
        fig_net.savefig(path_net, dpi=500, bbox_inches="tight")
        plt.close(fig_net)

        fig_avg, _ = plot_stream_thrust_average(
            soln.inlet,
            avg,
            plot_vars=plot_vars,
            bounds=bounds,
            n_idl=soln.N_idl,
        )
        path_avg = output_dir / f"stream_thrust_average_{soln.N_idl}_IDL.png"
        fig_avg.savefig(path_avg, dpi=500, bbox_inches="tight")
        plt.close(fig_avg)

        if write_csv:
            path_csv = output_dir / f"stream_thrust_average_{soln.N_idl}_IDL.csv"
            save_stream_thrust_average_csv(path_csv, avg)

    return SolutionProcessingResult(
        final_state=final_state,
        average_profile=avg,
        eta_inlet=eta_inlet,
        p0_loss=p0_loss,
    )
