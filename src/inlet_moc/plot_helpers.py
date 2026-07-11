from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from matplotlib.ticker import FormatStrFormatter

if TYPE_CHECKING:
    from inlet_moc.moc_soln import MOCSolution

@dataclass(frozen=True)
class PlotVariable:
    latex_name: str
    cbar_label: str
    ylims: tuple[float, float]
    cmap: str
    key: str = ""


@dataclass(frozen=True)
class PlotBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def x_span(self) -> float:
        return max(self.x_max - self.x_min, 1e-12)

    @property
    def y_span(self) -> float:
        return max(self.y_max - self.y_min, 1e-12)

    @classmethod
    def from_solution(cls, soln: MOCSolution) -> PlotBounds:
        x_max = max(soln.inlet.centerbody.x_max, soln.inlet.cowl.x_max)
        x_stop = getattr(soln, "x_stop", None)
        if x_stop is not None:
            x_max = min(float(x_max), float(x_stop))
        return cls(
            x_min=min(soln.inlet.centerbody.x_min, soln.inlet.cowl.x_min),
            x_max=x_max,
            y_min=min(soln.inlet.centerbody.y_min, soln.inlet.cowl.y_min),
            y_max=max(soln.inlet.centerbody.y_max, soln.inlet.cowl.y_max),
        )

    @classmethod
    def from_solution_with_xmax(cls, soln: MOCSolution, x_max: float) -> PlotBounds:
        return cls(
            x_min=min(soln.inlet.centerbody.x_min, soln.inlet.cowl.x_min),
            x_max=float(x_max),
            y_min=min(soln.inlet.centerbody.y_min, soln.inlet.cowl.y_min),
            y_max=max(soln.inlet.centerbody.y_max, soln.inlet.cowl.y_max),
        )


def scatter_point(ax, pt, label=None, c="k"):
    if label is None:
        ax.scatter(pt[0], pt[1], s=2, c=c)
    else:
        ax.scatter(pt[0], pt[1], s=2, c=c, label=label)


class PlotSettings:
    cmap_mag = "Oranges"
    cmap_neg = "viridis"
    cmap_wave = "viridis"
    rho = PlotVariable(
        r"$\rho$",
        r"$\rho\ \left[\mathrm{kg}/\mathrm{m}^3\right]$",
        (0.4, 2.5),
        cmap_mag,
        key="rho",
    )
    mach = PlotVariable(r"$M$", r"$M$", (1.0, 5.0), cmap_mag, key="mach")
    p = PlotVariable(
        r"$p$",
        r"$p\ \left[\mathrm{kPa}\right]$",
        (0.0, 120.0),
        cmap_mag,
        key="p",
    )
    T = PlotVariable(
        r"$T$",
        r"$T\ \left[\mathrm{K}\right]$",
        (0.0, 200.0),
        cmap_mag,
        key="t",
    )
    a = PlotVariable(
        r"$a$",
        r"$a\ \left[\mathrm{m/s}\right]$",
        (0.0, 400.0),
        cmap_wave,
        key="a",
    )
    u = PlotVariable(
        r"$u$",
        r"$u\ \left[\mathrm{m/s} \right]$",
        (0.0, 800.0),
        cmap_neg,
        key="u",
    )
    v = PlotVariable(
        r"$v$",
        r"$v\ \left[\mathrm{m/s} \right]$",
        (-200.0, 200.0),
        cmap_neg,
        key="v",
    )
    rho_st = PlotVariable(
        r"$\rho_{st}$",
        r"$\rho_{st}\ [\mathrm{kg}/\mathrm{m}^3]$",
        rho.ylims,
        cmap_mag,
        key="rho",
    )
    u_st = PlotVariable(
        r"$u_{st}$",
        r"$u_{st}\ [\mathrm{m/s}]$",
        u.ylims,
        cmap_neg,
        key="u",
    )
    p_st = PlotVariable(
        r"$p_{st}$",
        r"$p_{st}\ [\mathrm{kPa}]$",
        p.ylims,
        cmap_mag,
        key="p",
    )
    a_st = PlotVariable(
        r"$a_{st}$",
        r"$a_{st}\ [\mathrm{m/s}]$",
        a.ylims,
        cmap_wave,
        key="a",
    )
    T_st = PlotVariable(
        r"$T_{st}$",
        r"$T_{st}\ [\mathrm{K}]$",
        T.ylims,
        cmap_mag,
        key="t",
    )
    mach_st = PlotVariable(
        r"$M_{st}$",
        r"$M_{st}\ [-]$",
        mach.ylims,
        cmap_mag,
        key="mach",
    )

    registry: ClassVar[dict[str, PlotVariable]] = {
        "m": mach,
        "rho": rho,
        "mach": mach,
        "p": p,
        "t": T,
        "temperature": T,
        "a": a,
        "c": a,
        "u": u,
        "v": v,
    }
    stream_thrust_registry: ClassVar[dict[str, PlotVariable]] = {
        "rho": rho_st,
        "u": u_st,
        "p": p_st,
        "a": a_st,
        "c": a_st,
        "t": T_st,
        "temperature": T_st,
        "m": mach_st,
        "mach": mach_st,
    }

    @classmethod
    def get(cls, key):
        return cls.registry[str(key).lower()]

    @classmethod
    def get_stream_thrust(cls, key):
        return cls.stream_thrust_registry[str(key).lower()]


def _nice_axis_step(span: float, target_ticks: int = 4) -> float:
    span = max(float(span), 1.0e-12)
    raw_step = span / max(int(target_ticks), 1)
    exponent = 10.0 ** np.floor(np.log10(raw_step))
    scaled = raw_step / exponent
    if scaled <= 1.0:
        nice = 1.0
    elif scaled <= 2.0:
        nice = 2.0
    elif scaled <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * exponent


def _nice_y_limits_and_ticks(
    bounds: PlotBounds,
) -> tuple[float, float, np.ndarray, str]:
    step = _nice_axis_step(bounds.y_span, target_ticks=3)
    y_min = float(np.floor(bounds.y_min / step) * step)
    y_max = float(np.ceil(bounds.y_max / step) * step)
    if (bounds.y_min >= 0.0) and (abs(y_min) <= 0.5 * step):
        y_min = 0.0
    ticks = np.arange(y_min, y_max + 0.5 * step, step, dtype=float)
    dec_place = max(0, int(np.ceil(-np.log10(step))) + 1) if step > 0 else 2
    fmt = f"%.{dec_place}f"
    return y_min, y_max, ticks, fmt


def nice_ylims_ticks(
    y_min: float,
    y_max: float,
) -> tuple[float, float, np.ndarray, str]:
    bounds = PlotBounds(x_min=0.0, x_max=1.0, y_min=float(y_min), y_max=float(y_max))
    return _nice_y_limits_and_ticks(bounds)


def _format_solution_axis(ax, bounds: PlotBounds) -> None:
    y_min, y_max, y_ticks, y_fmt = _nice_y_limits_and_ticks(bounds)
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_fmt))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
