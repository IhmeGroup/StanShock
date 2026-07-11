from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from inlet_moc.cross_sectional_average import (
    compute_stream_thrust_average,
    save_stream_thrust_average_csv,
)
from inlet_moc.flow_physics import total_pressure_ratio
from inlet_moc.get_streamtube_bounds import get_streamtube_bounds
from inlet_moc.plot_helpers import PlotBounds
from inlet_moc.plot_soln import (
    plot_moc_soln,
    plot_stream_thrust_average,
)

if TYPE_CHECKING:
    from inlet_moc.moc_soln import MOCSolution


DIRECT_AVERAGE_KEYS = ("x", "rho_st", "u_st", "p_st", "a_st", "T_st", "mach")


@dataclass(frozen=True)
class SolutionProcessingResult:
    mode: str
    final_state: np.ndarray
    average_profile: np.ndarray | None
    eta_inlet: float
    p0_loss: float
    saved_paths: dict[str, Path] = field(default_factory=dict)


def _capture_metrics_from_bounds(
    soln: MOCSolution,
    *,
    bounds,
) -> tuple[float, float]:
    if soln.x_start is None:
        msg = "Solution x_start is not available."
        raise RuntimeError(msg)
    x_b, y_bottom_b, y_top_b = bounds
    y_bottom = np.interp(soln.x_start, x_b, y_bottom_b)
    y_top = np.interp(soln.x_start, x_b, y_top_b)
    return soln.x_start, abs(y_top - y_bottom)


def _build_direct_mode_bounds(
    soln: MOCSolution,
    *,
    x_capture: float,
    y_bottom_capture: float,
    y_top_capture: float,
):
    x_final = soln.x_final()
    if x_final is None:
        msg = "x_final is not available for direct-mode processing."
        raise RuntimeError(msg)

    y_bottom_final = soln.inlet.centerbody.get_y(x_final)
    y_top_final = soln.inlet.cowl.get_y(x_final)

    x = np.array([x_capture, x_final])
    y_bottom = np.array([y_bottom_capture, y_bottom_final])
    y_top = np.array([y_top_capture, y_top_final])
    return x, y_bottom, y_top


def process_solution(
    soln: MOCSolution,
    *,
    figdir: str | Path | None = None,
    plot_vars: tuple[str, ...] = ("rho", "p", "M", "T"),
    gamma_total_pressure: float = 1.4,
) -> SolutionProcessingResult:
    if not getattr(soln, "nets", []):
        msg = "No solved nets are available for processing."
        raise RuntimeError(msg)

    x_final = soln.x_final()
    if x_final is None:
        msg = "x_final is not available for processing."
        raise RuntimeError(msg)

    mode = str(getattr(soln, "solution_mode", "analytical")).lower()

    saved_paths: dict[str, Path] = {}
    output_dir = Path(figdir if figdir is not None else soln.figdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "analytical":
        bounds = get_streamtube_bounds(soln)
        avg = compute_stream_thrust_average(soln, nx=soln.nx, bounds=bounds)
        final_state = avg[-1]
        _, A_cap = _capture_metrics_from_bounds(soln, bounds=bounds)

        x_plot_max = float(x_final) + 0.05
        plot_bounds = PlotBounds.from_solution_with_xmax(soln, x_plot_max)
        plot_label = "_".join(str(name) for name in plot_vars)

        fig_soln, _ = plot_moc_soln(
            soln,
            plot_var=plot_vars,
            show_nets=False,
            show_points=False,
            bounds=plot_bounds,
        )
        path_soln = output_dir / f"{plot_label}_{soln.N_idl}_IDL.png"
        fig_soln.savefig(path_soln, dpi=500, bbox_inches="tight")
        plt.close(fig_soln)
        saved_paths["solution_plot"] = path_soln

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
        saved_paths["net_plot"] = path_net

        fig_avg, _ = plot_stream_thrust_average(
            soln,
            avg,
            plot_vars=plot_vars,
            bounds=bounds,
        )
        path_avg = output_dir / f"stream_thrust_average_{soln.N_idl}_IDL.png"
        fig_avg.savefig(path_avg, dpi=500, bbox_inches="tight")
        plt.close(fig_avg)
        saved_paths["stream_thrust_plot"] = path_avg

        path_csv = output_dir / f"stream_thrust_average_{soln.N_idl}_IDL.csv"
        save_stream_thrust_average_csv(path_csv, avg)
        saved_paths["stream_thrust_csv"] = path_csv

        average_profile = avg
    else:
        bounds = get_streamtube_bounds(soln, x_q=np.array([soln.x_start, x_final]))
        avg = compute_stream_thrust_average(
            soln,
            x=np.array([x_final]),
            bounds=bounds,
        )
        final_state = avg[0]
        _, A_cap = _capture_metrics_from_bounds(soln, bounds=bounds)
        average_profile = avg

    A_0 = abs(soln.inlet.cowl.y[0] - soln.inlet.centerbody.y[0])
    eta_inlet = A_cap / A_0
    p0_amb = soln.p_amb * total_pressure_ratio(soln.M_init, gamma_total_pressure)
    p0_final = final_state[3] * total_pressure_ratio(
        final_state[2] / final_state[4],
        gamma_total_pressure,
    )
    p0_loss = p0_final / p0_amb

    result = SolutionProcessingResult(
        mode=mode,
        final_state=final_state,
        average_profile=average_profile,
        eta_inlet=eta_inlet,
        p0_loss=p0_loss,
        saved_paths=saved_paths,
    )
    soln.processing_result = result
    return result
