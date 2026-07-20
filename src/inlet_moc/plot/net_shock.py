from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from inlet_moc.plot.helpers import PlotBounds, _format_solution_axis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from inlet_moc.char_net import CharNet


@dataclass(frozen=True)
class ShockStepOverlay:
    shock_pt: np.ndarray | None = None
    shock_idx: tuple[int, int] | None = None
    mesh_pt: np.ndarray | None = None
    mesh_idx: tuple[int, int] | None = None
    left_interp: np.ndarray | None = None
    left_idx: tuple[int, int] | None = None
    right_interp: np.ndarray | None = None
    right_idx: tuple[int, int] | None = None


def _finite_net_xy(net: CharNet) -> tuple[np.ndarray, np.ndarray]:
    xy_mask = net.xy_mask()
    if np.any(xy_mask):
        return np.asarray(net.x[xy_mask], dtype=float), np.asarray(
            net.y[xy_mask], dtype=float
        )

    finite = np.isfinite(net.x) & np.isfinite(net.y)
    if np.any(finite):
        return np.asarray(net.x[finite], dtype=float), np.asarray(
            net.y[finite], dtype=float
        )

    return np.array([], dtype=float), np.array([], dtype=float)


def _plot_net_lines(
    ax,
    net: CharNet,
    *,
    color: str,
    lw: float,
    label: str,
) -> None:
    xy_mask = net.xy_mask()
    first = True
    for i in range(net.N):
        cols = np.flatnonzero(xy_mask[i, :])
        if cols.size >= 2:
            ax.plot(
                net.x[i, cols],
                net.y[i, cols],
                color=color,
                lw=lw,
                zorder=6,
                label=label if first else None,
            )
            first = False
    for j in range(net.N):
        rows = np.flatnonzero(xy_mask[:, j])
        if rows.size >= 2:
            ax.plot(
                net.x[rows, j],
                net.y[rows, j],
                color=color,
                lw=lw,
                zorder=6,
                label=label if first else None,
            )
            first = False


def _get_debug_axes(
    debug_plots: bool,
    debug_plotter: Callable[[], tuple[Figure, Axes]] | None,
) -> tuple[Figure, Axes] | tuple[None, None]:
    if not debug_plots or debug_plotter is None:
        return None, None
    return debug_plotter()


def _cleanup_char_index_arrays(
    net: CharNet,
    fixed_idx: int,
    free_inds: np.ndarray,
    *,
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    free_inds = np.asarray(free_inds, dtype=int).reshape(-1)
    if free_inds.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    free_inds = np.unique(free_inds)
    if family == "cminus":
        rows = free_inds
        cols = np.full_like(rows, int(fixed_idx))
    else:
        rows = np.full_like(free_inds, int(fixed_idx))
        cols = free_inds

    in_bounds = (rows >= 0) & (rows < net.N) & (cols >= 0) & (cols < net.N)
    rows = rows[in_bounds]
    cols = cols[in_bounds]
    if rows.size == 0:
        return rows, cols

    finite = net.xy_mask()[rows, cols]
    return rows[finite], cols[finite]


def _cleanup_active_free_indices(
    net: CharNet,
    fixed_idx: int,
    family: str,
) -> np.ndarray:
    mask = net.state_mask()
    if family == "cminus":
        return np.flatnonzero(mask[:, int(fixed_idx)])
    return np.flatnonzero(mask[int(fixed_idx), :])


def _plot_cleanup_candidates(
    net: CharNet,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    debug_plots: bool,
    debug_plotter: Callable[[], tuple[Figure, Axes]] | None,
    boundary_xy: np.ndarray | None = None,
) -> None:
    if rows.size == 0:
        return

    fig, ax = _get_debug_axes(debug_plots, debug_plotter)
    if fig is None or ax is None:
        return

    x_delete = net.x[rows, cols]
    y_delete = net.y[rows, cols]

    if boundary_xy is not None:
        boundary_xy = np.asarray(boundary_xy, dtype=float).reshape(-1, 2)
    if boundary_xy is not None and boundary_xy.shape[0] >= 2:
        ax.plot(
            boundary_xy[:, 0],
            boundary_xy[:, 1],
            color="tab:red",
            linestyle="--",
            lw=1.2,
            zorder=13,
            label="cleanup boundary",
        )

    ax.scatter(
        x_delete,
        y_delete,
        s=3,
        c="red",
        zorder=14,
        label="delete",
    )
    fig.show()
    plt.close(fig)


def _plot_characteristic(
    ax: Axes,
    net: CharNet,
    *,
    fixed_idx: int,
    family: str,
    color: str,
    label: str,
) -> None:
    free_inds = _cleanup_active_free_indices(net, int(fixed_idx), family)
    rows, cols = _cleanup_char_index_arrays(
        net,
        int(fixed_idx),
        free_inds,
        family=family,
    )
    if rows.size == 0:
        return

    ax.plot(
        net.x[rows, cols],
        net.y[rows, cols],
        color=color,
        lw=1.4,
        zorder=12,
        label=label,
    )
    ax.scatter(
        net.x[rows, cols],
        net.y[rows, cols],
        s=2,
        c=color,
        zorder=13,
    )


def _plot_coalescing_cleanup(
    net: CharNet,
    *,
    family: str,
    clip_idx: int,
    keep_idx: int,
    clip_start: int,
    intersection_xy: np.ndarray,
    debug_plots: bool,
    debug_plotter: Callable[[], tuple[Figure, Axes]] | None,
) -> None:
    free_inds_delete = np.arange(int(clip_start), net.N, dtype=int)
    rows_delete, cols_delete = _cleanup_char_index_arrays(
        net,
        int(clip_idx),
        free_inds_delete,
        family=family,
    )
    if rows_delete.size == 0:
        return

    fig, ax = _get_debug_axes(debug_plots, debug_plotter)
    if ax is None:
        return

    _plot_characteristic(
        ax,
        net,
        fixed_idx=int(clip_idx),
        family=family,
        color="tab:orange",
        label=f"clip {family}={clip_idx}",
    )
    _plot_characteristic(
        ax,
        net,
        fixed_idx=int(keep_idx),
        family=family,
        color="tab:blue",
        label=f"keep {family}={keep_idx}",
    )

    if intersection_xy.size:
        ax.scatter(
            intersection_xy[:, 0],
            intersection_xy[:, 1],
            s=6,
            c="k",
            marker="x",
            zorder=14,
            label="intersection",
        )

    ax.scatter(
        net.x[rows_delete, cols_delete],
        net.y[rows_delete, cols_delete],
        s=6,
        c="red",
        zorder=15,
        label="delete",
    )
    ax.set_title(f"Coalescing {family} cleanup")
    ax.legend()
    fig.show()
    plt.close(fig)


class ShockNetPlotter:
    def __init__(
        self,
        net_L: CharNet,
        net_R: CharNet,
        *,
        family: str,
        minimal: bool = False,
        output_path: str | Path | None = None,
    ) -> None:
        self.net_L = net_L
        self.net_R = net_R
        self.family = str(family)
        self.minimal = bool(minimal)
        self.output_path = None if output_path is None else Path(output_path)
        self._history: list[ShockStepOverlay] = []
        self._resampled_shock_state = np.empty((0, 6), dtype=float)

        plt.ion()
        self.fig, self.ax = self.net_L.inlet.plot_inlet()
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title(f"net_shock: {self.family}")

        self.redraw()

    def _x_limits_data(self) -> tuple[float, float]:
        x_L, y_L = _finite_net_xy(self.net_L)
        x_R, y_R = _finite_net_xy(self.net_R)

        x_parts = []
        y_parts = [
            np.asarray(self.net_L.inlet.centerbody.y, dtype=float),
            np.asarray(self.net_L.inlet.cowl.y, dtype=float),
        ]
        if x_L.size > 0:
            x_parts.append(x_L)
            y_parts.append(y_L)
        if x_R.size > 0:
            x_parts.append(x_R)
            y_parts.append(y_R)

        if x_parts:
            x_min_data = (
                float(np.nanmin(x_L)) if x_L.size > 0 else float(np.nanmin(x_parts[0]))
            )
            if x_R.size > 0:
                x_max_data = float(np.nanmax(x_R))
            elif x_L.size > 0:
                x_max_data = float(np.nanmax(x_L))
            else:
                x_max_data = x_min_data + 1.0e-6

        else:
            x_min_data = min(
                float(self.net_L.inlet.centerbody.x_min),
                float(self.net_L.inlet.cowl.x_min),
            )
            x_max_data = max(
                float(self.net_L.inlet.centerbody.x_max),
                float(self.net_L.inlet.cowl.x_max),
            )

        return float(x_min_data), float(x_max_data)

    def _fallback_y_limits(self) -> tuple[float, float]:
        _, y_L = _finite_net_xy(self.net_L)
        _, y_R = _finite_net_xy(self.net_R)
        y_parts = [
            np.asarray(self.net_L.inlet.centerbody.y, dtype=float),
            np.asarray(self.net_L.inlet.cowl.y, dtype=float),
        ]
        if y_L.size > 0:
            y_parts.append(y_L)
        if y_R.size > 0:
            y_parts.append(y_R)
        y_all = np.concatenate(y_parts)
        y_min = float(np.nanmin(y_all))
        y_max = float(np.nanmax(y_all))
        y_span = max(y_max - y_min, 1.0e-6)
        y_pad = max(0.08 * y_span, 2.0e-4)
        return y_min - y_pad, y_max + y_pad

    def _bounds(self) -> PlotBounds:
        x_min_data, x_max_data = self._x_limits_data()
        x_span = max(float(x_max_data - x_min_data), 1.0e-6)
        x_pad = 0.02 * x_span

        wall_y = []
        for wall in (
            self.net_L.inlet.centerbody,
            self.net_L.inlet.cowl,
        ):
            for x_val in (x_min_data, x_max_data):
                try:
                    y_val = wall.get_y(float(x_val))
                except Exception:
                    y_val = None
                if y_val is None:
                    continue
                if np.isfinite(y_val):
                    wall_y.append(float(y_val))

        if wall_y:
            y_min = float(np.min(wall_y))
            y_max = float(np.max(wall_y))
            y_span = max(y_max - y_min, 1.0e-6)
            y_pad = max(0.12 * y_span, 5.0e-4)
            y_min -= y_pad
            y_max += y_pad
        else:
            y_min, y_max = self._fallback_y_limits()

        return PlotBounds(
            x_min=float(x_min_data - x_pad),
            x_max=float(x_max_data + x_pad),
            y_min=float(y_min),
            y_max=float(y_max),
        )

    @staticmethod
    def _idx_label(idx: tuple[int, int] | None) -> str | None:
        if idx is None:
            return None
        i, j = (int(idx[0]), int(idx[1]))
        return rf"$\mathbf{{{i}}},{j}$"

    def add_step(
        self,
        *,
        shock_pt: np.ndarray | None = None,
        shock_idx: tuple[int, int] | None = None,
        mesh_pt: np.ndarray | None = None,
        mesh_idx: tuple[int, int] | None = None,
        left_interp: np.ndarray | None = None,
        left_idx: tuple[int, int] | None = None,
        right_interp: np.ndarray | None = None,
        right_idx: tuple[int, int] | None = None,
    ) -> None:
        self._history.append(
            ShockStepOverlay(
                shock_pt=shock_pt,
                shock_idx=shock_idx,
                mesh_pt=mesh_pt,
                mesh_idx=mesh_idx,
                left_interp=left_interp,
                left_idx=left_idx,
                right_interp=right_interp,
                right_idx=right_idx,
            )
        )
        self.redraw()

    def set_resampled_points(self, shock_state: np.ndarray) -> None:
        shock_state = np.asarray(shock_state, dtype=float)
        if shock_state.ndim == 1:
            shock_state = shock_state.reshape(1, -1)
        if shock_state.size == 0:
            self._resampled_shock_state = np.empty((0, 6), dtype=float)
        else:
            self._resampled_shock_state = shock_state[:, :6]
        self.redraw()

    def _annotate_point(
        self,
        pt: np.ndarray,
        idx: tuple[int, int] | None,
        *,
        color: str,
    ) -> None:
        label = self._idx_label(idx)
        if label is None:
            return
        self.ax.annotate(
            label,
            (float(pt[0]), float(pt[1])),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            color=color,
            ha="left",
            va="bottom",
            zorder=13,
        )

    def _scatter_role(
        self,
        pt: np.ndarray | None,
        idx: tuple[int, int] | None,
        *,
        marker: str,
        edgecolor: str,
        facecolor: str | None,
        label: str | None,
        seen_labels: set[str],
        linewidth: float = 1.0,
        size: float = 24.0,
    ) -> None:
        if pt is None or not np.all(np.isfinite(pt[:2])):
            return
        scatter_label = None
        if label is not None and label not in seen_labels:
            scatter_label = label

        scatter_kwargs = {
            "s": size,
            "marker": marker,
            "linewidths": linewidth,
            "zorder": 14,
            "label": scatter_label,
        }
        if marker == "x":
            scatter_kwargs["color"] = edgecolor
        else:
            scatter_kwargs["edgecolors"] = edgecolor
            scatter_kwargs["facecolors"] = facecolor

        self.ax.scatter(float(pt[0]), float(pt[1]), **scatter_kwargs)
        if label is not None:
            seen_labels.add(label)
        self._annotate_point(pt, idx, color=edgecolor)

    @staticmethod
    def _shock_style() -> dict[str, str | float | None]:
        return {
            "marker": "o",
            "edgecolor": "r",
            "facecolor": "r",
            "label": "shock",
            "linewidth": 1.0,
            "size": 3.0,
        }

    @staticmethod
    def _mesh_style() -> dict[str, str | float | None]:
        return {
            "marker": "o",
            "edgecolor": "b",
            "facecolor": "b",
            "label": "mesh",
            "linewidth": 1.0,
            "size": 6.0,
        }

    @staticmethod
    def _interp_style(side: str) -> dict[str, str | float | None]:
        color = "0.55" if side == "left" else "k"
        return {
            "marker": "o",
            "edgecolor": color,
            "facecolor": "none",
            "label": None,
            "linewidth": 1.0,
            "size": 22.0,
        }

    def _plot_resampled_points(self, seen_labels: set[str] | None = None) -> None:
        if self._resampled_shock_state.size == 0:
            return
        pts = self._resampled_shock_state
        finite = np.all(np.isfinite(pts[:, :2]), axis=1)
        if not np.any(finite):
            return

        label = None
        if seen_labels is not None and "resampled" not in seen_labels:
            label = "resampled"

        self.ax.scatter(
            pts[finite, 0],
            pts[finite, 1],
            s=3,
            marker="o",
            edgecolors="r",
            facecolors="none",
            linewidths=0.7,
            zorder=16,
            label=label,
        )
        if seen_labels is not None:
            seen_labels.add("resampled")

    def redraw(self) -> None:
        self.ax.clear()
        self.net_L.inlet.plot_inlet(fig=self.fig, ax=self.ax)
        _format_solution_axis(self.ax, self._bounds())
        self.ax.set_title(f"{self.family} shock solve")

        _plot_net_lines(
            self.ax,
            self.net_L,
            color="0.55",
            lw=0.30,
            label="_nolegend_",
        )
        _plot_net_lines(
            self.ax,
            self.net_R,
            color="k",
            lw=0.30,
            label="_nolegend_",
        )

        if self.minimal:
            for overlay in self._history:
                if overlay.shock_pt is not None and np.all(
                    np.isfinite(overlay.shock_pt[:2])
                ):
                    self.ax.scatter(
                        float(overlay.shock_pt[0]),
                        float(overlay.shock_pt[1]),
                        s=6,
                        c="r",
                        zorder=14,
                    )
                    self._annotate_point(overlay.shock_pt, overlay.shock_idx, color="r")
                if overlay.mesh_pt is not None and np.all(
                    np.isfinite(overlay.mesh_pt[:2])
                ):
                    self.ax.scatter(
                        float(overlay.mesh_pt[0]),
                        float(overlay.mesh_pt[1]),
                        s=6,
                        c="b",
                        zorder=14,
                    )
                    self._annotate_point(overlay.mesh_pt, overlay.mesh_idx, color="b")

            self._plot_resampled_points()
            self.fig.tight_layout()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            backend = str(plt.get_backend()).lower()
            if "agg" not in backend:
                plt.pause(0.001)
            return

        seen_labels = set()
        accepted_mesh_indices: set[tuple[int, int]] = set()
        for overlay in self._history:
            shock_style = self._shock_style()
            mesh_style = self._mesh_style()
            left_style = self._interp_style("left")
            right_style = self._interp_style("right")
            self._scatter_role(
                overlay.shock_pt,
                overlay.shock_idx,
                **shock_style,
                seen_labels=seen_labels,
            )
            self._scatter_role(
                overlay.mesh_pt,
                overlay.mesh_idx,
                **mesh_style,
                seen_labels=seen_labels,
            )
            if overlay.mesh_idx is not None:
                accepted_mesh_indices.add(tuple(map(int, overlay.mesh_idx)))

            left_is_reused_mesh = (
                overlay.left_idx is not None
                and tuple(map(int, overlay.left_idx)) in accepted_mesh_indices
            )
            right_is_reused_mesh = (
                overlay.right_idx is not None
                and tuple(map(int, overlay.right_idx)) in accepted_mesh_indices
            )

            if not left_is_reused_mesh:
                self._scatter_role(
                    overlay.left_interp,
                    overlay.left_idx,
                    **left_style,
                    seen_labels=seen_labels,
                )
            if not right_is_reused_mesh:
                self._scatter_role(
                    overlay.right_interp,
                    overlay.right_idx,
                    **right_style,
                    seen_labels=seen_labels,
                )

        self._plot_resampled_points(seen_labels)

        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles, strict=False))
            self.ax.legend(
                unique.values(),
                unique.keys(),
                loc="best",
                frameon=True,
                fontsize=5,
            )

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        backend = str(plt.get_backend()).lower()
        if "agg" not in backend:
            plt.pause(0.001)

    def finalize(self) -> None:
        self.redraw()
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(self.output_path, dpi=300, bbox_inches="tight")
        plt.close(self.fig)
