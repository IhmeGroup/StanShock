"""Validation utilities for the HyShot II example.

Reproduces the wall-pressure and wall-heat-flux comparison figures from the
project's reference MATLAB script (``plotdd_newppt.m``), comparing the 1D
StanShock solution against 3D RANS (CTR) results and experimental measurements
for the HyShot II scramjet combustor.

The CTR reference CSVs (in ``reference_data/``) store *nondimensional* wall
quantities that must be scaled by the freestream reference dynamic pressure and
reference heat flux to recover physical units. Each file is laid out as::

    row 0:  group headers (Experiment (Center) / Upper / Lower / 1/8C No Walls / ...)
    row 1:  X,Y pair labels
    row 2+: data, NaN-padded because the series have unequal lengths

    col 0,1 -> experiment center   (x, y)
    col   3 -> experiment upper-bound y
    col   5 -> experiment lower-bound y
    col 6,7 -> CTR 3D RANS          (x, y)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Freestream reference scales used to redimensionalize the CTR/experiment data.
P_REF_PA = 17.7e6  # reference dynamic pressure [Pa]
Q_REF_W_M2 = 12.37e9  # reference heat flux [W/m^2]

REFERENCE_DIR = Path(__file__).resolve().parent / "reference_data"

# Consistent colors for the two combustor walls.
COLOR_BODY = "black"
COLOR_COWL = "tab:red"
COLOR_SIM = "tab:blue"


@dataclass
class WallReference:
    """Experimental + 3D RANS reference data for one wall/quantity."""

    exp_x: np.ndarray  # [mm]
    exp_y: np.ndarray  # physical units (kPa or W/m^2)
    exp_err_lower: np.ndarray  # asymmetric lower error bar magnitude
    exp_err_upper: np.ndarray  # asymmetric upper error bar magnitude
    rans_x: np.ndarray  # [mm]
    rans_y: np.ndarray  # physical units


def _drop_nan(*columns: np.ndarray) -> tuple[np.ndarray, ...]:
    """Drop rows where any of the given columns is NaN (the padding value)."""
    mask = np.ones(len(columns[0]), dtype=bool)
    for col in columns:
        mask &= ~np.isnan(col)
    return tuple(col[mask] for col in columns)


def load_ctr_reference(filename: str, scale: float) -> WallReference:
    """Parse a CTR reference CSV and scale it to physical units.

    Parameters
    ----------
    filename:
        Name of the CSV inside ``reference_data/``.
    scale:
        Multiplicative factor converting the nondimensional ``y`` values to the
        desired physical units (e.g. ``P_REF_PA / 1e3`` for kPa).
    """
    data = pd.read_csv(REFERENCE_DIR / filename, skiprows=2, header=None).to_numpy(
        dtype=float
    )

    exp_x, exp_center, exp_upper, exp_lower = _drop_nan(
        data[:, 0], data[:, 1], data[:, 3], data[:, 5]
    )
    rans_x, rans_y = _drop_nan(data[:, 6], data[:, 7])

    return WallReference(
        exp_x=exp_x,
        exp_y=exp_center * scale,
        exp_err_lower=np.abs(exp_center - exp_lower) * scale,
        exp_err_upper=np.abs(exp_upper - exp_center) * scale,
        rans_x=rans_x,
        rans_y=rans_y * scale,
    )


@dataclass
class StanShockResult:
    """1D StanShock wall profiles extracted from a CSVWriter output file."""

    x_mm: np.ndarray
    pressure_kpa: np.ndarray
    heat_flux_w_m2: np.ndarray


def load_stanshock_result(path: str | Path) -> StanShockResult:
    """Load a StanShock CSVWriter result file.

    Expects the columns emitted by ``CSVWriter``: ``x [m]``, ``pressure [Pa]``,
    ..., and ``wall_heat_flux [W/m^2]`` as the final column.
    """
    df = pd.read_csv(path)
    x_m = df.iloc[:, 0].to_numpy(dtype=float)
    pressure_pa = df["pressure [Pa]"].to_numpy(dtype=float)
    heat_flux = df.iloc[:, -1].to_numpy(dtype=float)
    return StanShockResult(
        x_mm=x_m * 1e3,
        pressure_kpa=pressure_pa / 1e3,
        heat_flux_w_m2=heat_flux,
    )


def compute_error_metrics(
    sim_x: np.ndarray,
    sim_y: np.ndarray,
    ref: WallReference,
) -> dict[str, float]:
    """L1 and L-infinity error of the StanShock profile vs. experiment.

    The StanShock solution is linearly interpolated onto the (scattered)
    experimental measurement locations before the norms are taken.
    """
    sim_at_exp = np.interp(ref.exp_x, sim_x, sim_y)
    error = np.abs(sim_at_exp - ref.exp_y)
    return {
        "l1": float(np.mean(error)),
        "linf": float(np.max(error)),
        "l1_normalized": float(np.mean(error) / np.mean(np.abs(ref.exp_y))),
    }


def _plot_wall_comparison(
    ax: plt.Axes,
    sim_x: np.ndarray,
    sim_y: np.ndarray,
    body: WallReference,
    cowl: WallReference,
    *,
    ylabel: str,
    title: str,
) -> None:
    """Draw one comparison panel: experiment (markers), RANS (dotted), sim (line)."""
    for ref, color, label in (
        (cowl, COLOR_COWL, "Cowl-Side"),
        (body, COLOR_BODY, "Body-Side"),
    ):
        ax.errorbar(
            ref.exp_x,
            ref.exp_y,
            yerr=[ref.exp_err_lower, ref.exp_err_upper],
            linestyle="none",
            marker="s",
            markersize=4,
            capsize=3,
            color=color,
            label=f"Experiment, {label}",
        )
        ax.plot(
            ref.rans_x,
            ref.rans_y,
            linestyle=":",
            color=color,
            label=f"CTR (3D RANS), {label}",
        )

    ax.plot(sim_x, sim_y, color=COLOR_SIM, linewidth=2, label="StanShock (1D)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 400)


def plot_pressure_and_heat_flux(
    result: StanShockResult,
    *,
    reacting: bool,
    output_path: Path,
) -> None:
    """Reproduce a fuel-on (reacting) or fuel-off (inert) two-panel figure."""
    if reacting:
        p_body = load_ctr_reference("CTR-React-Pressure-Body.csv", P_REF_PA / 1e3)
        p_cowl = load_ctr_reference("CTR-React-Pressure-Cowl.csv", P_REF_PA / 1e3)
        q_body = load_ctr_reference("CTR-React-HeatFlux-Body.csv", Q_REF_W_M2)
        q_cowl = load_ctr_reference("CTR-React-HeatFlux-Cowl.csv", Q_REF_W_M2)
        prefix = "Fuel-On"
    else:
        p_body = load_ctr_reference(
            "CTR-Inert-Pressure-Body-Menter.csv", P_REF_PA / 1e3
        )
        p_cowl = load_ctr_reference(
            "CTR-Inert-Pressure-Cowl-Menter.csv", P_REF_PA / 1e3
        )
        q_body = load_ctr_reference("CTR-Inert-HeatFlux-Body-Menter.csv", Q_REF_W_M2)
        q_cowl = load_ctr_reference("CTR-Inert-HeatFlux-Cowl-Menter.csv", Q_REF_W_M2)
        prefix = "Fuel-Off"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _plot_wall_comparison(
        axes[0],
        result.x_mm,
        result.pressure_kpa,
        p_body,
        p_cowl,
        ylabel="Pressure [kPa]",
        title=f"{prefix} Pressure Distribution",
    )
    _plot_wall_comparison(
        axes[1],
        result.x_mm,
        result.heat_flux_w_m2,
        q_body,
        q_cowl,
        ylabel=r"Heat Flux [W/m$^2$]",
        title=f"{prefix} Heat Flux",
    )
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_phi_sweep_reference() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load the CTR body-pressure equivalence-ratio sweep (phi = 0.3 and 0.5).

    The ``-EQs`` file stores the phi=0.3 curve in columns (0, 1) and the
    phi=0.5 curve in columns (2, 3), in nondimensional pressure.
    """
    data = pd.read_csv(
        REFERENCE_DIR / "CTR-React-Pressure-Body-EQs.csv", skiprows=2, header=None
    ).to_numpy(dtype=float)
    scale = P_REF_PA / 1e3
    x_lo, y_lo = _drop_nan(data[:, 0], data[:, 1])
    x_hi, y_hi = _drop_nan(data[:, 2], data[:, 3])
    return {"0.3": (x_lo, y_lo * scale), "0.5": (x_hi, y_hi * scale)}


def plot_phi_sweep(
    result_phi_low: StanShockResult,
    result_phi_high: StanShockResult,
    *,
    output_path: Path,
) -> None:
    """Reproduce the fuel-on body-pressure equivalence-ratio comparison figure."""
    ctr = load_phi_sweep_reference()

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(
        result_phi_low.x_mm,
        result_phi_low.pressure_kpa,
        color=COLOR_SIM,
        linewidth=2,
        label=r"StanShock (1D), $\phi=0.3$",
    )
    ax.plot(
        result_phi_high.x_mm,
        result_phi_high.pressure_kpa,
        color=COLOR_SIM,
        linewidth=2,
        linestyle="-.",
        label=r"StanShock (1D), $\phi=0.5$",
    )
    ax.plot(
        *ctr["0.3"], color=COLOR_BODY, linewidth=1, label=r"CTR (3D RANS), $\phi=0.3$"
    )
    ax.plot(
        *ctr["0.5"],
        color=COLOR_BODY,
        linewidth=1,
        linestyle="-.",
        label=r"CTR (3D RANS), $\phi=0.5$",
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Pressure [kPa]")
    ax.set_title(r"Fuel-On Body Pressure, Varying $\phi$")
    ax.set_xlim(0, 400)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def report_error_metrics(result: StanShockResult, *, reacting: bool) -> dict[str, dict]:
    """Compute and print pressure/heat-flux error metrics vs. experiment."""
    tag = "React" if reacting else "Inert"
    suffix = "" if reacting else "-Menter"
    refs = {
        "pressure_body": load_ctr_reference(
            f"CTR-{tag}-Pressure-Body{suffix}.csv", P_REF_PA / 1e3
        ),
        "pressure_cowl": load_ctr_reference(
            f"CTR-{tag}-Pressure-Cowl{suffix}.csv", P_REF_PA / 1e3
        ),
        "heatflux_body": load_ctr_reference(
            f"CTR-{tag}-HeatFlux-Body{suffix}.csv", Q_REF_W_M2
        ),
        "heatflux_cowl": load_ctr_reference(
            f"CTR-{tag}-HeatFlux-Cowl{suffix}.csv", Q_REF_W_M2
        ),
    }
    metrics = {}
    for key, ref in refs.items():
        sim_y = result.pressure_kpa if "pressure" in key else result.heat_flux_w_m2
        metrics[key] = compute_error_metrics(result.x_mm, sim_y, ref)

    print(f"\n{'Fuel-On' if reacting else 'Fuel-Off'} error metrics (vs experiment):")
    for key, m in metrics.items():
        print(
            f"  {key:16s}  L1={m['l1']:.3e}  Linf={m['linf']:.3e}  "
            f"L1_norm={m['l1_normalized']:.3f}"
        )
    return metrics


if __name__ == "__main__":
    # Phase-A self-test: reproduce the figures from a precomputed baseline
    # StanShock result rather than running the full simulation.
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reacting-csv", type=Path, required=True, help="StanShock reacting result CSV"
    )
    parser.add_argument(
        "--inert-csv", type=Path, default=None, help="StanShock inert result CSV"
    )
    parser.add_argument(
        "--phi-high-csv",
        type=Path,
        default=None,
        help="StanShock reacting result CSV at higher phi (for the phi sweep)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./output/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reacting = load_stanshock_result(args.reacting_csv)
    plot_pressure_and_heat_flux(
        reacting, reacting=True, output_path=args.output_dir / "hyshot_ii_fuel_on.png"
    )
    report_error_metrics(reacting, reacting=True)

    if args.inert_csv is not None:
        inert = load_stanshock_result(args.inert_csv)
        plot_pressure_and_heat_flux(
            inert,
            reacting=False,
            output_path=args.output_dir / "hyshot_ii_fuel_off.png",
        )
        report_error_metrics(inert, reacting=False)

    if args.phi_high_csv is not None:
        phi_high = load_stanshock_result(args.phi_high_csv)
        plot_phi_sweep(
            reacting,
            phi_high,
            output_path=args.output_dir / "hyshot_ii_phi_sweep.png",
        )

    print(f"\nFigures written to {args.output_dir}")
