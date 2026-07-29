from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from matplotlib.tri import Triangulation

from inlet_moc.plot.helpers import (
    PlotBounds,
    PlotSettings,
    _format_solution_axis,
    nice_ylims_ticks,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from inlet_moc.char_net import CharNet
    from inlet_moc.moc_solution import MOCSolution

datadir = Path(__file__).parent / "../../../data"
plt.style.use(datadir / "stylelib/publication.mplstyle")


def _primitive_values(
    soln: MOCSolution, primitives: np.ndarray, plot_key: str
) -> np.ndarray:
    key = str(plot_key).lower()
    # primitives are [rho, u, v, p, a], derived from rotational states.
    rho = primitives[:, 0]
    u = primitives[:, 1]
    v = primitives[:, 2]
    p = primitives[:, 3]
    a = np.sqrt(soln.gamma * p / rho)
    match key:
        case "rho":
            return rho
        case "u":
            return u
        case "v":
            return v
        case "p":
            return p / 1000.0
        case "a" | "c":
            return a
        case "m" | "mach":
            return np.hypot(u, v) / a
        case "t" | "temperature":
            return p / (rho * soln.R)
        case _:
            msg = f"Unsupported plot variable: {plot_key}"
            raise KeyError(msg)


def _stream_thrust_x(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.dtype.names is not None:
        return np.asarray(data["x"], dtype=float)
    return np.asarray(data[:, 0], dtype=float)


def _stream_thrust_profile(data: np.ndarray, var: str) -> np.ndarray:
    data = np.asarray(data)
    var = str(var).lower()
    if data.dtype.names is not None:
        if var == "rho":
            return np.asarray(data["rho_st"], dtype=float)
        if var == "u":
            return np.asarray(data["u_st"], dtype=float)
        if var == "p":
            return np.asarray(data["p_st"], dtype=float) / 1000.0
        if var in {"a", "c"}:
            return np.asarray(data["a_st"], dtype=float)
        if var in {"t", "temperature"}:
            return np.asarray(data["T_st"], dtype=float)
        if var in {"m", "mach"}:
            if "mach" in data.dtype.names:
                return np.asarray(data["mach"], dtype=float)
            return np.asarray(data["u_st"], dtype=float) / np.asarray(
                data["a_st"], dtype=float
            )

    columns = {
        "rho": 1,
        "u": 2,
        "p": 3,
        "a": 4,
        "c": 4,
        "t": 5,
        "temperature": 5,
        "m": 6,
        "mach": 6,
    }
    profile = np.asarray(data[:, columns[var]], dtype=float)
    if var == "p":
        return profile / 1000.0
    return profile


def _apply_nice_yaxis(ax, y_values: np.ndarray | None = None) -> None:
    if y_values is None:
        profiles = [np.asarray(line.get_ydata(), dtype=float) for line in ax.lines]
        if not profiles:
            return
        y_values = np.concatenate(profiles)
    else:
        y_values = np.asarray(y_values, dtype=float)
    finite = y_values[np.isfinite(y_values)]
    if finite.size == 0:
        return
    y_min, y_max, y_ticks, y_fmt = nice_ylims_ticks(
        float(np.min(finite)),
        float(np.max(finite)),
    )
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_fmt))


def _refresh_stream_thrust_legends(axes: np.ndarray) -> None:
    for ax in np.asarray(axes, dtype=object).ravel():
        handles, labels = ax.get_legend_handles_labels()
        keep = [
            idx for idx, label in enumerate(labels) if not str(label).startswith("_")
        ]
        legend = ax.get_legend()
        if len(keep) <= 1:
            if legend is not None:
                legend.remove()
            continue

        ax.legend(
            [handles[idx] for idx in keep],
            [labels[idx] for idx in keep],
            loc="best",
            frameon=False,
            fontsize=8,
            labelspacing=0.2,
            borderpad=0.2,
            handletextpad=0.4,
        )


def plot_net_lines(ax, net: CharNet) -> None:
    xy_mask = net.xy_mask()
    for i in range(net.N):
        cols = np.flatnonzero(xy_mask[i, :])
        if cols.size >= 2:
            ax.plot(net.x[i, cols], net.y[i, cols], color="0.55", lw=0.1, zorder=6)
    for j in range(net.N):
        rows = np.flatnonzero(xy_mask[:, j])
        if rows.size >= 2:
            ax.plot(net.x[rows, j], net.y[rows, j], color="0.55", lw=0.1, zorder=6)


def plot_net_points(
    ax,
    net: CharNet,
    show_ids: bool = False,
    color_by_point_type: bool = False,
) -> None:
    xy_mask = net.xy_mask()
    pt_size = 0.1
    if color_by_point_type:  # REMOVE THIS ARGUMENT!
        point_types = np.asarray(
            getattr(net, "point_type", np.full(net.x.shape, np.nan))
        )
        type_specs = (
            (0, "k", "Internal"),
            (1, "b", "Wall"),
            (2, "r", "Fluid Boundary"),
            (3, "g", "Corner"),
        )
        for point_type, color, label in type_specs:
            type_mask = xy_mask & np.isclose(point_types, point_type, equal_nan=False)
            if not np.any(type_mask):
                continue
            ax.scatter(
                net.x[type_mask],
                net.y[type_mask],
                s=pt_size,
                marker=".",
                c=color,
                linewidths=0.0,
                label=label,
                zorder=7,
            )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles, strict=False))
            ax.legend(
                unique.values(),
                unique.keys(),
                loc="upper right",
                fontsize=8,
                frameon=True,
                handletextpad=0.4,
                borderpad=0.3,
            )
    else:
        ax.scatter(
            net.x[xy_mask],
            net.y[xy_mask],
            s=pt_size,
            marker=".",
            c="k",
            linewidths=0.0,
            zorder=7,
        )

    if show_ids:
        for i, j in np.argwhere(xy_mask):
            ax.annotate(
                f"({int(i)}, {int(j)})",
                (net.x[i, j], net.y[i, j]),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=6,
                color="0.25",
                ha="left",
                va="bottom",
                zorder=8,
            )


def debug_plot(
    net: CharNet,
    fig: Figure | None = None,
    ax: Axes | None = None,
    show_point_types: bool = False,
    show_point_ids: bool = True,
) -> tuple[Figure, Axes]:
    if fig is None or ax is None:
        xy_mask = net.xy_mask()
        if not np.any(xy_mask):
            xy_mask = np.isfinite(net.x) & np.isfinite(net.y)
        dx = 1.0e-2
        x_valid = net.x[xy_mask]
        y_valid = net.y[xy_mask]
        bounds = PlotBounds(
            x_min=float(np.nanmin(x_valid) - dx),
            x_max=float(np.nanmax(x_valid) + dx),
            y_min=float(np.nanmin(y_valid) - dx),
            y_max=float(np.nanmax(y_valid) + dx),
        )
        fig, ax = net.inlet.plot_inlet()
        _format_solution_axis(ax, bounds)

    plot_net_lines(ax, net)
    plot_net_points(
        ax,
        net,
        show_ids=show_point_ids,
        color_by_point_type=show_point_types,
    )
    fig.tight_layout()
    return fig, ax


def _plot_tri_fills(
    ax: Axes,
    plot_key: str,
    points: np.ndarray,
    triangles: np.ndarray,
    primitives: np.ndarray,
    soln: MOCSolution,
    cmap: str,
    norm: Normalize,
    *,
    plot_tri_edges: bool = False,
) -> bool:
    if points.shape[0] == 0 or triangles.shape[0] == 0:
        return False
    tri_obj = Triangulation(points[:, 0], points[:, 1], triangles)
    ax.tripcolor(
        tri_obj,
        _primitive_values(soln, primitives, plot_key),
        shading="gouraud",
        cmap=cmap,
        norm=norm,
        zorder=3,
    )
    if plot_tri_edges:
        ax.triplot(
            points[:, 0],
            points[:, 1],
            triangles,
            color="0.25",
            lw=0.5,
            zorder=8,
        )
    return True


def plot_moc_soln(
    soln: MOCSolution,
    plot_var: str | Sequence[str] | None = None,
    show_points: bool = False,
    show_nets: bool = False,
    show_point_ids: bool = False,
    bounds: PlotBounds | None = None,
    point_mesh=None,
    plot_tri_edges: bool = False,
) -> tuple[Figure, Axes]:
    if plot_var is None:
        plot_vars: list[str] = []
    elif isinstance(plot_var, str):
        plot_vars = [plot_var] if plot_var.strip() else []
    else:
        plot_vars = [name for name in plot_var if str(name).strip()]

    bounds = PlotBounds.from_solution(soln) if bounds is None else bounds
    fig_width = 11.0
    row_height = float(np.clip(fig_width * (bounds.y_span / bounds.x_span), 2.4, 5.0))

    if not plot_vars:
        fig = plt.figure(figsize=(fig_width, row_height))
        ax = fig.add_subplot(111)
        soln.inlet.plot_inlet(fig=fig, ax=ax)
        if show_nets:
            for net in soln.nets:
                if net is not None:
                    plot_net_lines(ax, net)
        if show_points or show_point_ids:
            for net in soln.nets:
                if net is not None:
                    plot_net_points(ax, net, show_ids=show_point_ids)
        _format_solution_axis(ax, bounds)
        _draw_completed_x_marker(soln, ax)
        fig.tight_layout()
        return fig, ax

    plot_settings = [PlotSettings.get(name) for name in plot_vars]
    if point_mesh is None:
        point_mesh = getattr(soln, "point_mesh", None)
    if point_mesh is None:
        msg = "point_mesh is required to plot solution fields."
        raise RuntimeError(msg)
    fig = plt.figure(figsize=(fig_width, row_height * len(plot_settings)))
    gs = fig.add_gridspec(
        nrows=len(plot_settings),
        ncols=2,
        width_ratios=[30, 1.6],
        hspace=0.34,
        wspace=0.08,
    )

    axes = [fig.add_subplot(gs[idx, 0]) for idx in range(len(plot_settings))]
    cbar_axes = [fig.add_subplot(gs[idx, 1]) for idx in range(len(plot_settings))]

    for ax, cax, plot_cfg in zip(
        axes,
        cbar_axes,
        plot_settings,
        strict=False,
    ):
        soln.inlet.plot_inlet(fig=fig, ax=ax)
        norm = Normalize(vmin=plot_cfg.ylims[0], vmax=plot_cfg.ylims[1])
        drew_data = False

        if _plot_tri_fills(
            ax,
            plot_cfg.key,
            point_mesh.points,
            point_mesh.triangles,
            point_mesh.primitives,
            soln,
            plot_cfg.cmap,
            norm,
            plot_tri_edges=plot_tri_edges,
        ):
            drew_data = True

        if drew_data:
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=plot_cfg.cmap), cax=cax)
            cbar.set_label(plot_cfg.cbar_label)
        else:
            cax.set_visible(False)

        if show_nets:
            for net in soln.nets:
                if net is not None:
                    plot_net_lines(ax, net)
        if show_points or show_point_ids:
            for net in soln.nets:
                if net is not None:
                    plot_net_points(ax, net, show_ids=show_point_ids)
        soln.inlet.plot_inlet(fig=fig, ax=ax)
        _format_solution_axis(ax, bounds)

    _draw_completed_x_marker(soln, axes)
    if len(axes) == 1:
        return fig, axes[0]
    return fig, np.array(axes, dtype=object)


def plot_stream_thrust_average(
    inlet,
    avg: np.ndarray | None = None,
    plot_vars: Sequence[str] = ("rho", "u", "p", "mach", "t"),
    bounds=None,
    global_legend: str | None = None,
    *,
    n_idl: int | None = None,
    fig=None,
    axes=None,
    dataset: np.ndarray | None = None,
    color: str = "b",
):
    plot_settings = tuple(PlotSettings.get_stream_thrust(var) for var in plot_vars)
    n_plots = len(plot_settings)
    plot_data = dataset if dataset is not None else avg

    if plot_data is None:
        msg = "avg or dataset is required for stream-thrust plotting."
        raise ValueError(msg)
    if bounds is None:
        msg = "bounds are required for stream-thrust plotting."
        raise ValueError(msg)

    x_bounds, y_cent_st, y_cowl_st = bounds

    y_all = np.concatenate((y_cent_st, y_cowl_st))
    x_all = x_bounds
    x_span = max(float(np.max(x_all) - np.min(x_all)), 1.0e-8)
    y_span = max(float(np.max(y_all) - np.min(y_all)), 1.0e-8)
    fig_width = 6.8
    row_height = float(np.clip(fig_width * (y_span / x_span), 1.7, 3.0))

    draw_streamtube = axes is None
    if axes is None:
        fig, axes = plt.subplots(
            nrows=n_plots,
            ncols=1,
            figsize=(fig_width, row_height * n_plots),
            squeeze=False,
        )
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.ndim == 0:
            axes = axes.reshape(1, 1)
        if fig is None:
            fig = axes.ravel()[0].figure

    axes_flat = axes.ravel()
    x_plot = _stream_thrust_x(plot_data)
    label = global_legend
    if label is None:
        if dataset is not None:
            label = "Dataset"
        elif n_idl is None:
            label = "MOC"
        else:
            label = f"MOC ({n_idl} IDL)"

    for idx, (ax, plot_cfg) in enumerate(zip(axes_flat, plot_settings, strict=False)):
        var = plot_cfg.key
        profile = _stream_thrust_profile(plot_data, var)
        ax.plot(x_plot, profile, c=color, lw=1.2, label=label)
        ax.set_ylabel(plot_cfg.cbar_label)
        ax.grid(alpha=0.25)
        ax.set_xlabel("x [m]")
        _apply_nice_yaxis(ax)
        if draw_streamtube:
            _add_h_plot(
                inlet,
                ax,
                x_plot,
                bounds=bounds,
                show_ylabel=(idx == 0),
            )

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    _refresh_stream_thrust_legends(axes_flat[:n_plots])
    fig.tight_layout()
    return fig, axes


def _add_h_plot(
    inlet,
    ax,
    x_plot: np.ndarray,
    *,
    bounds,
    show_ylabel: bool,
):
    ax1 = ax.twinx()
    ax1.set_zorder(-np.inf)
    ax.patch.set_visible(False)

    x_plot = np.asarray(x_plot, dtype=float)
    lower_wall = np.asarray(
        [inlet.centerbody.get_y(float(x_q)) for x_q in x_plot],
        dtype=float,
    )
    upper_wall = np.asarray(
        [inlet.cowl.get_y(float(x_q)) for x_q in x_plot],
        dtype=float,
    )
    x_bounds, y_lower, y_upper = bounds
    lower_cap = np.interp(x_plot, x_bounds, y_lower)
    upper_cap = np.interp(x_plot, x_bounds, y_upper)
    x_geom_last = float(min(inlet.centerbody.x_max, inlet.cowl.x_max))
    y_upper_last = float(inlet.cowl.get_y(x_geom_last))
    y_lower_last = float(inlet.centerbody.get_y(x_geom_last))
    y_marg = 0.5 * abs(y_upper_last - y_lower_last)

    y_geom_max = float(
        max(
            inlet.centerbody.y_max,
            inlet.cowl.y_max,
        )
    )
    y_geom_min = float(
        min(
            inlet.centerbody.y_min,
            inlet.cowl.y_min,
        )
    )
    y_anchor_top = y_geom_max + y_marg
    y_anchor_bot = y_geom_min - y_marg

    for y_vals, linestyle in (
        (upper_wall, "-"),
        (lower_wall, "-"),
        (upper_cap, "--"),
        (lower_cap, "--"),
    ):
        ax1.plot(
            x_plot,
            y_vals,
            color="0.55",
            linestyle=linestyle,
            lw=0.9,
        )

    ax1.scatter(
        [x_geom_last, x_geom_last],
        [y_anchor_bot, y_anchor_top],
        s=0.1,
        c="0.55",
        alpha=0.0,
        zorder=0,
    )

    y_min, y_max, y_ticks, y_fmt = nice_ylims_ticks(y_anchor_bot, y_anchor_top)
    ax1.set_xlim(float(np.min(x_plot)), float(np.max(x_plot)))
    ax1.set_ylim(y_min, y_max)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_yticks(y_ticks)
    ax1.yaxis.set_major_formatter(FormatStrFormatter(y_fmt))
    ax1.yaxis.set_ticks_position("right")
    ax1.yaxis.set_label_position("right")
    ax1.spines["right"].set_visible(True)
    ax1.tick_params(axis="y", right=True, labelright=True)
    ax1.set_ylabel("y [m]")
    return ax1


def _draw_completed_x_marker(soln: MOCSolution, axes: Axes | list[Axes]) -> None:
    x_final = soln.x_final()
    if x_final is None:
        return
    y_lower = soln.inlet.centerbody.get_y(float(x_final))
    y_upper = soln.inlet.cowl.get_y(float(x_final))
    if y_lower is None or y_upper is None:
        return

    for ax in np.atleast_1d(axes).ravel():
        ax.plot(
            [float(x_final), float(x_final)],
            [float(y_lower), float(y_upper)],
            color="k",
            linestyle="--",
            lw=1.0,
            zorder=60,
        )


def save_error_state_plot(
    soln: MOCSolution,
    *,
    subtitle: str | None = None,
    highlight_net: CharNet | None = None,
    highlight_ids: bool = False,
    output_path: str | Path | None = None,
) -> Path:
    plot_var = None if getattr(soln, "point_mesh", None) is None else "mach"
    plot_kwargs = {
        "plot_var": plot_var,
        "show_points": False,
        "show_nets": True,
        "show_point_ids": False,
    }

    appended_temp_net = False
    if highlight_net is not None and highlight_net not in soln.nets:
        soln.nets.append(highlight_net)
        appended_temp_net = True

    try:
        fig, axes = plot_moc_soln(soln, **plot_kwargs)
    finally:
        if appended_temp_net:
            soln.nets.pop()

    axes_arr = np.atleast_1d(axes).ravel()
    if highlight_net is not None:
        for ax in axes_arr:
            plot_net_points(ax, highlight_net, show_ids=highlight_ids)
    if subtitle:
        fig.suptitle(subtitle, fontsize=8)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        fig.tight_layout()

    if output_path is None:
        plot_dir = Path(soln.figdir).expanduser().resolve()
        plot_dir.mkdir(parents=True, exist_ok=True)
        filename = "last_progress_plot.png"
        if hasattr(soln, "case_filename"):
            filename = soln.case_filename(filename)
        output = plot_dir / filename
    else:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return output
