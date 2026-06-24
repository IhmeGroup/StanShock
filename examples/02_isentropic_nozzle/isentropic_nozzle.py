from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

from stanshock.components.combustor import Combustor
from stanshock.numerics.boundary_conditions import BCInput, FreezeCells
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.csv_writer import CSVWriter
from stanshock.processing.initialize import InitializeIsentropic
from stanshock.processing.plot import VariableInfo, get_variable_info_map
from stanshock.system.geometry import Box
from stanshock.utils.isentropic import mach_from_area_ratio, property_ratios


def make_nozzle_area(
    xf: np.ndarray,
    area_ratio_max: float = 4.0,
    sigma_fraction: float = 0.18,
) -> np.ndarray:
    """Return a smooth converging-diverging nozzle area profile.

    The profile is normalized such that A/A* = 1 at the throat and
    A/A* = area_ratio_max at the inlet and outlet.
    """
    x_mid = 0.5 * (xf[0] + xf[-1])
    length = xf[-1] - xf[0]
    sigma = sigma_fraction * length

    profile = 1.0 - np.exp(-((xf - x_mid) ** 2) / (2.0 * sigma**2))
    profile /= profile.max()

    area_ratio = 1.0 + (area_ratio_max - 1.0) * profile
    throat_area = 1.0 / area_ratio_max
    return throat_area * area_ratio


def build_theory_profiles(
    geometry: Box,
    gamma: float,
    inflow_pressure: float,
    inflow_density: float,
    throat_area: float,
    subsonic_inflow: bool = True,
    subsonic_outflow: bool = False,
) -> dict[str, np.ndarray | float]:
    """Compute analytical quasi-1D isentropic nozzle profiles."""
    xc = geometry.xc[geometry.idx_cells]
    area = geometry.area(0.0, xc)
    if not isinstance(area, np.ndarray):
        area = np.full_like(xc, area)

    area_ratio = area / throat_area
    mach_sub = mach_from_area_ratio(area_ratio, gamma, subsonic=True)
    mach_sup = mach_from_area_ratio(area_ratio, gamma, subsonic=False)

    mach = mach_sub.copy()
    throat_idx = int(np.argmin(area_ratio))
    if subsonic_inflow != subsonic_outflow:
        if subsonic_inflow:
            mach[throat_idx + 1 :] = mach_sup[throat_idx + 1 :]
        else:
            mach[:throat_idx] = mach_sup[:throat_idx]
    elif not subsonic_inflow and not subsonic_outflow:
        mach = mach_sup

    inflow_area_ratio = float(geometry.area(0.0, geometry.xf[0]) / throat_area)
    inflow_mach = float(
        mach_from_area_ratio(
            np.array([inflow_area_ratio]), gamma, subsonic=subsonic_inflow
        )[0]
    )
    _, p_ratio_in, rho_ratio_in = property_ratios(inflow_mach, gamma)
    _, p_ratio_profile, rho_ratio_profile = property_ratios(mach, gamma)

    pressure = inflow_pressure / p_ratio_in * p_ratio_profile
    density = inflow_density / rho_ratio_in * rho_ratio_profile
    velocity = mach * np.sqrt(gamma * pressure / density)

    return {
        "x": xc,
        "area": area,
        "area_ratio": area_ratio,
        "mach": mach,
        "pressure": pressure,
        "density": density,
        "velocity": velocity,
        "throat_index": throat_idx,
        "inflow_mach": inflow_mach,
    }


def extract_numerical_profiles(
    ss: Combustor, idx: slice | np.ndarray
) -> dict[str, np.ndarray]:
    """Extract primitive profiles from the current combustor state."""
    state = ss.state[idx]
    sound_speed = ss.physics.get_sound_speed(state)
    velocity = ss.physics.get_velocity(state)
    return {
        "x": ss.geometry.xc[idx],
        "mach": np.abs(velocity) / sound_speed,
        "pressure": ss.physics.get_pressure(state),
        "density": ss.physics.get_density(state),
        "velocity": velocity,
        "temperature": ss.physics.get_temperature(state),
    }


def compute_error_metrics(
    x: np.ndarray,
    numerical: dict[str, np.ndarray],
    theory: dict[str, np.ndarray | float],
    prefix: str,
) -> dict[str, float]:
    """Compute L1 and max errors against analytical nozzle theory."""
    dx = np.gradient(x)
    metrics: dict[str, float] = {}
    for key in ["mach", "pressure", "density", "velocity"]:
        reference = np.asarray(theory[key])
        value = numerical[key]
        error = np.abs(value - reference)
        metrics[f"{prefix}_l1_{key}"] = float(np.sum(error * dx) / np.sum(dx))
        metrics[f"{prefix}_linf_{key}"] = float(np.max(error))
    return metrics


def compute_drift_metrics(
    x: np.ndarray,
    initial: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute short-time drift metrics between initial and final profiles."""
    dx = np.gradient(x)
    metrics: dict[str, float] = {}
    for key in ["mach", "pressure", "density", "velocity"]:
        drift = np.abs(final[key] - initial[key])
        metrics[f"drift_l1_{key}"] = float(np.sum(drift * dx) / np.sum(dx))
        metrics[f"drift_linf_{key}"] = float(np.max(drift))
    return metrics


def plot_results(
    output_dir: Path,
    theory: dict[str, np.ndarray | float],
    initial: dict[str, np.ndarray],
    final: dict[str, np.ndarray],
) -> None:
    """Save comparison figures for the nozzle case."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    x_mm = 1e3 * initial["x"]
    area_ratio = np.asarray(theory["area_ratio"])

    fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

    axes[0].plot(x_mm, initial["mach"], label="Initialized StanShock")
    axes[0].plot(x_mm, final["mach"], label="Advanced StanShock")
    axes[0].plot(x_mm, np.asarray(theory["mach"]), "--", label="Theory")
    axes[0].set_ylabel("Mach [-]")
    axes[0].legend()

    axes[1].plot(x_mm, 1.0e-5 * initial["pressure"], label="Initialized")
    axes[1].plot(x_mm, 1.0e-5 * final["pressure"], label="Advanced")
    axes[1].plot(x_mm, 1.0e-5 * np.asarray(theory["pressure"]), "--", label="Theory")
    axes[1].set_ylabel("Pressure [bar]")

    axes[2].plot(x_mm, initial["density"], label="Initialized")
    axes[2].plot(x_mm, final["density"], label="Advanced")
    axes[2].plot(x_mm, np.asarray(theory["density"]), "--", label="Theory")
    axes[2].set_ylabel(r"Density [kg/m$^3$]")

    axes[3].plot(x_mm, area_ratio, color="k")
    axes[3].set_ylabel(r"$A/A^*$ [-]")
    axes[3].set_xlabel("x [mm]")

    fig.tight_layout()
    fig.savefig(
        figures_dir / "isentropic_nozzle_profiles.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(x_mm, initial["mach"], label="Initialized StanShock")
    ax1.plot(x_mm, final["mach"], label="Advanced StanShock")
    ax1.plot(x_mm, np.asarray(theory["mach"]), "--", label="Theory")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("Mach [-]")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(x_mm, area_ratio, linestyle=":")
    ax2.set_ylabel(r"$A/A^*$ [-]")

    fig.tight_layout()
    fig.savefig(
        figures_dir / "isentropic_nozzle_mach.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def write_summary(
    output_dir: Path,
    metrics: dict[str, float],
    metadata: dict[str, float | int | str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    fieldnames = list(metadata.keys()) + list(metrics.keys())
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({**metadata, **metrics})


def main(
    *,
    plot_results_flag: bool = True,
    write_csv: bool = True,
    advance_time: float | None = None,
    n_preservation_steps: int = 5,
    n_cells: int = 401,
    area_ratio_max: float = 4.0,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the isentropic nozzle example.

    This case demonstrates StanShock's geometry and area-change features by
    constructing a smooth converging-diverging nozzle, initializing the flow
    using quasi-1D isentropic nozzle theory, and then advancing the solution a
    short time to verify that the solver preserves the quasi-steady balance.
    """
    case_dir = Path(__file__).resolve().parent
    if output_dir is None:
        output_dir = case_dir / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mechanism_path = (
        Path(__file__).resolve().parents[2] / "data" / "mechanisms" / "Nitrogen.yaml"
    )

    gas = ct.Solution(mechanism_path)
    gas.TP = 300.0, 5.0e5
    physics = ThermoTable(gas)

    length = 1.0
    xf = np.linspace(0.0, length, n_cells + 1)
    area = make_nozzle_area(xf, area_ratio_max=area_ratio_max)
    geometry = Box(xf=xf, h=area, w=1.0)

    face_area = geometry.area(0.0, geometry.xf)
    assert isinstance(face_area, np.ndarray)
    throat_area = float(np.min(face_area))

    initialization = InitializeIsentropic(
        geometry=geometry,
        physics=physics,
        inflow_state=gas,
        throat_area=throat_area,
        subsonic_inflow=True,
        subsonic_outflow=False,
    )
    boundary_conditions: BCInput = {
        "left": FreezeCells(location="left"),
        "right": FreezeCells(location="right"),
    }

    ss = Combustor(
        geometry=geometry,
        initialization=initialization,
        boundary_conditions=boundary_conditions,
        physics=physics,
        include_diffusion=False,
        output_every=100,
        use_double_flux=False,
    )

    idx = geometry.idx_cells
    x = geometry.xc[idx]

    theory = build_theory_profiles(
        geometry=geometry,
        gamma=float(gas.cp / gas.cv),
        inflow_pressure=float(gas.P),
        inflow_density=float(gas.density_mass),
        throat_area=throat_area,
        subsonic_inflow=True,
        subsonic_outflow=False,
    )

    ss.advance_simulation(0.0)
    numerical_initial = extract_numerical_profiles(ss, idx)

    if advance_time is None:
        advance_time = float(n_preservation_steps * ss.get_time_step())
    if advance_time > 0.0:
        ss.advance_simulation(float(advance_time))

    numerical_final = extract_numerical_profiles(ss, idx)

    metrics: dict[str, float] = {}
    metrics.update(
        compute_error_metrics(x, numerical_initial, theory, prefix="initial")
    )
    metrics.update(compute_error_metrics(x, numerical_final, theory, prefix="final"))
    metrics.update(compute_drift_metrics(x, numerical_initial, numerical_final))

    variable_info_map = get_variable_info_map(physics)
    area_interior = np.asarray(theory["area"])
    area_ratio_interior = np.asarray(theory["area_ratio"])
    variable_info_map["area"] = VariableInfo(
        short_name="A",
        plot_label=r"$A~[\mathrm{m^2}]$",
        fun=lambda _state, arr=area_interior: arr,
    )
    variable_info_map["area ratio"] = VariableInfo(
        short_name="A_Astar",
        plot_label=r"$A/A^*~[-]$",
        fun=lambda _state, arr=area_ratio_interior: arr,
    )

    if write_csv:
        csv_writer = CSVWriter(
            combustor=ss,
            filename=output_dir / "csv" / "isentropic_nozzle_state.csv",
            interval=0,
            variables=["x", "mach", "p", "rho", "u", "T", "area", "area ratio"],
            variable_info_map=variable_info_map,
        )
        csv_writer.write_current_state()

    if plot_results_flag:
        plot_results(output_dir, theory, numerical_initial, numerical_final)

    metadata: dict[str, float | int | str] = {
        "case": "02_isentropic_nozzle",
        "n_cells": n_cells,
        "advance_time": float(advance_time),
        "n_preservation_steps": n_preservation_steps,
        "area_ratio_max": area_ratio_max,
        "throat_area": throat_area,
        "inflow_mach": float(theory["inflow_mach"]),
    }
    write_summary(output_dir, metrics, metadata)

    print("Isentropic nozzle case complete.")
    print(f"Output directory: {output_dir}")
    print(
        "Final L1 errors: "
        + ", ".join(
            [
                f"Ma={metrics['final_l1_mach']:.3e}",
                f"p={metrics['final_l1_pressure']:.3e}",
                f"rho={metrics['final_l1_density']:.3e}",
                f"u={metrics['final_l1_velocity']:.3e}",
            ]
        )
    )
    print(
        "Short-time drift L1: "
        + ", ".join(
            [
                f"Ma={metrics['drift_l1_mach']:.3e}",
                f"p={metrics['drift_l1_pressure']:.3e}",
                f"rho={metrics['drift_l1_density']:.3e}",
                f"u={metrics['drift_l1_velocity']:.3e}",
            ]
        )
    )

    return {
        "combustor": ss,
        "theory": theory,
        "numerical_initial": numerical_initial,
        "numerical_final": numerical_final,
        "metrics": metrics,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    main()
