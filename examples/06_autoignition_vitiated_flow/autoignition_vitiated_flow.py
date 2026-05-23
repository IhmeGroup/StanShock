from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.physics.flamelet import FPVTable
from stanshock.physics.fluid_base import FluidState


@dataclass
class CaseConfig:
    """Shipped chemistry-comparison case based on Caban & Tyliszczak (2024)."""

    pressure: float = 101325.0
    fuel_temperature: float = 300.0
    oxidizer_temperature: float = 1400.0
    fuel_composition: dict[str, float] | None = None
    oxidizer_composition: dict[str, float] | None = None
    progress_variable_definition: dict[str, float] | None = None
    t_final: float = 2.0e-2
    ignition_temperature_rise: float = 0.01
    verbose: bool = True
    progress_bar_width: int = 24
    reference_xi_mr: float = 7.1e-3
    reference_tign_ms: float = 0.034

    def __post_init__(self) -> None:
        if self.fuel_composition is None:
            self.fuel_composition = {"H2": 1.0}
        if self.oxidizer_composition is None:
            self.oxidizer_composition = {"O2": 0.233, "N2": 0.767}
        if self.progress_variable_definition is None:
            self.progress_variable_definition = {"H2O": 1.0}


def get_example_root() -> Path:
    return Path(__file__).resolve().parent


def get_default_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "figures": root / "figures",
        "output": root / "output",
        "reference": root / "reference_data" / "ignition_delay_reference.csv",
        "table": root / "table_input" / "flamelet_results" / "H2_O2N2_p01_0_tf0300_to1400_200x2x200.h5",
        "mechanism": root.parent.parent / "data" / "mechanisms" / "h2_boivin_9sp_12r_mod.yaml",
    }


def print_progress(model_name: str, index: int, total: int, z: float, tign_ms: float | None) -> None:
    progress = index / total
    width = 24
    n_fill = int(round(width * progress))
    bar = "#" * n_fill + "-" * (width - n_fill)
    if tign_ms is None or not np.isfinite(tign_ms):
        status = "no ignition"
    else:
        status = f"t_ign = {tign_ms:.4f} ms"
    print(
        f"[{model_name}] [{bar}] {100.0 * progress:5.1f}%  "
        f"point {index:02d}/{total:02d}  Z = {z:.5f}  {status}"
    )


def load_reference_curve(reference_file: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(reference_file, delimiter=",", names=True)
    z = np.asarray(data[data.dtype.names[0]], dtype=float)
    tign = np.asarray(data[data.dtype.names[1]], dtype=float)
    z = np.clip(z, 0.0, None)
    tign = np.abs(tign)
    mask = np.isfinite(z) & np.isfinite(tign)
    z = z[mask]
    tign = tign[mask]
    order = np.argsort(z)
    z = z[order]
    tign = tign[order]
    # drop near-duplicate Z points after clipping
    keep = np.ones_like(z, dtype=bool)
    keep[1:] = np.diff(z) > 1.0e-7
    return z[keep], tign[keep]


def mixed_initial_gas(config: CaseConfig, mechanism: Path, z: float) -> ct.Solution:
    """Create an inertly mixed H2/oxidizer parcel at constant pressure."""
    gas = ct.Solution(mechanism)
    gas_fuel = ct.Solution(mechanism)
    gas_ox = ct.Solution(mechanism)

    gas_fuel.TPX = config.fuel_temperature, config.pressure, config.fuel_composition
    gas_ox.TPX = config.oxidizer_temperature, config.pressure, config.oxidizer_composition

    y_mix = z * gas_fuel.Y + (1.0 - z) * gas_ox.Y
    y_mix /= y_mix.sum()
    h_mix = z * gas_fuel.enthalpy_mass + (1.0 - z) * gas_ox.enthalpy_mass

    gas.HPY = h_mix, config.pressure, y_mix
    return gas


def frc_ignition_delay_for_z(
    z: float,
    config: CaseConfig,
    mechanism: Path,
) -> tuple[float, float, dict[str, float]]:
    """Return ignition delay [ms], initial T [K], and final diagnostic scalars for one Z."""
    gas = mixed_initial_gas(config, mechanism, z)
    T_initial = gas.T
    T_threshold = (1.0 + config.ignition_temperature_rise) * T_initial

    reactor = ct.IdealGasConstPressureReactor(gas)
    network = ct.ReactorNet([reactor])

    tign_ms = np.nan
    while network.time < config.t_final:
        network.step()
        if reactor.T >= T_threshold:
            tign_ms = 1.0e3 * network.time
            break

    diag = {
        "temperature_final": reactor.T,
        "Y_H2_final": reactor.thermo["H2"].Y[0],
        "Y_H2O_final": reactor.thermo["H2O"].Y[0],
        "Y_OH_final": reactor.thermo["OH"].Y[0],
        "Y_O2_final": reactor.thermo["O2"].Y[0],
    }
    return tign_ms, T_initial, diag


def fpv_state_from_zc(fpv: FPVTable, pressure: float, z: float, c: float) -> FluidState:
    """Construct a one-point FluidState for standalone FPV lookup/integration."""
    c_max = float(fpv.lookup_direct("PROG", z, 0.0, 1.0))
    c_clipped = float(np.clip(c, 0.0, c_max))
    composition = np.array([[1.0, z, c_clipped]], dtype=float)

    # In the shipped table workflow without pressure/temperature corrections,
    # T0 is the natural table temperature coordinate at the given (Z, L).
    l = 0.0 if c_max <= 0.0 else c_clipped / c_max
    T = float(fpv.lookup_direct("T0", z, 0.0, l))
    R = float(fpv.lookup_direct("ROM", z, 0.0, l))
    rho = pressure / (R * T)

    return FluidState(
        shape=(1,),
        pressure=np.array([pressure]),
        density=np.array([rho]),
        temperature=np.array([T]),
        composition=composition,
    )


def fpv_temperature_from_zc(fpv: FPVTable, pressure: float, z: float, c: float) -> float:
    state = fpv_state_from_zc(fpv, pressure, z, c)
    return float(state.temperature[0])


def fpv_rhs(fpv: FPVTable, pressure: float, z: float, c: float) -> float:
    state = fpv_state_from_zc(fpv, pressure, z, c)
    source = fpv.get_source_terms(state)
    factor = fpv.get_source_progress_variable_compressibility_factor(state)
    return float(factor[0] * state.density[0] * source[0])


def fpv_ignition_delay_for_z(
    z: float,
    config: CaseConfig,
    mechanism: Path,
    table_file: Path,
) -> tuple[float, float, dict[str, float]]:
    gas = mixed_initial_gas(config, mechanism, z)
    fpv = FPVTable(
        table_file,
        ct.Solution(mechanism),
        ox_def=config.oxidizer_composition,
        fuel_def=config.fuel_composition,
        prog_def=config.progress_variable_definition,
    )
    c_initial = float(fpv.get_progress_variable(gas.Y[None, :])[0])
    T_initial = fpv_temperature_from_zc(fpv, config.pressure, z, c_initial)
    T_threshold = (1.0 + config.ignition_temperature_rise) * T_initial

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([fpv_rhs(fpv, config.pressure, z, float(y[0]))], dtype=float)

    def ignition_event(t: float, y: np.ndarray) -> float:
        return fpv_temperature_from_zc(fpv, config.pressure, z, float(y[0])) - T_threshold

    ignition_event.terminal = True
    ignition_event.direction = 1.0

    sol = solve_ivp(
        rhs,
        (0.0, config.t_final),
        y0=np.array([c_initial], dtype=float),
        method="BDF",
        max_step=min(1.0e-6, config.t_final / 20.0),
        events=ignition_event,
        rtol=1.0e-6,
        atol=1.0e-9,
    )

    if sol.t_events and len(sol.t_events[0]) > 0:
        tign_ms = 1.0e3 * float(sol.t_events[0][0])
        c_final = float(sol.y_events[0][0][0])
    else:
        tign_ms = np.nan
        c_final = float(sol.y[0, -1])

    state_final = fpv_state_from_zc(fpv, config.pressure, z, c_final)
    diag = {
        "temperature_final": float(state_final.temperature[0]),
        "progress_variable_final": c_final,
        "Y_H2_final": float(fpv.lookup("H2", state_final)[0]),
        "Y_H2O_final": float(fpv.lookup("H2O", state_final)[0]),
        "Y_OH_final": float(fpv.lookup("OH", state_final)[0]),
        "Y_O2_final": float(fpv.lookup("O2", state_final)[0]),
    }
    return tign_ms, T_initial, diag


def run_branch(
    model_name: str,
    z_points: np.ndarray,
    config: CaseConfig,
    mechanism: Path,
    table_file: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ignition_delay_ms = np.full_like(z_points, np.nan, dtype=float)
    temperature_initial = np.full_like(z_points, np.nan, dtype=float)

    diagnostics: dict[str, list[float]] = {
        "temperature_final": [],
        "Y_H2_final": [],
        "Y_H2O_final": [],
        "Y_OH_final": [],
        "Y_O2_final": [],
    }
    if model_name == "fpv":
        diagnostics["progress_variable_final"] = []

    for i, z in enumerate(z_points, start=1):
        if model_name == "frc":
            tign_ms, T_initial, diag = frc_ignition_delay_for_z(z, config, mechanism)
        elif model_name == "fpv":
            assert table_file is not None
            tign_ms, T_initial, diag = fpv_ignition_delay_for_z(z, config, mechanism, table_file)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        ignition_delay_ms[i - 1] = tign_ms
        temperature_initial[i - 1] = T_initial
        for key, value in diag.items():
            diagnostics[key].append(value)

        if config.verbose:
            print_progress(model_name, i, len(z_points), float(z), tign_ms)

    runtime = time.perf_counter() - t0
    if np.any(np.isfinite(ignition_delay_ms)):
        i_min = int(np.nanargmin(ignition_delay_ms))
        xi_mr = float(z_points[i_min])
        tign_ms_scalar = float(ignition_delay_ms[i_min])
    else:
        xi_mr = np.nan
        tign_ms_scalar = np.nan

    return {
        "model": model_name,
        "runtime_s": runtime,
        "z": z_points,
        "ignition_delay_ms": ignition_delay_ms,
        "temperature_initial": temperature_initial,
        "xi_mr": xi_mr,
        "tign_ms": tign_ms_scalar,
        "diagnostics": {k: np.array(v, dtype=float) for k, v in diagnostics.items()},
    }


def l1_error(x_ref: np.ndarray, y_ref: np.ndarray, x_model: np.ndarray, y_model: np.ndarray) -> float:
    mask_ref = np.isfinite(x_ref) & np.isfinite(y_ref)
    mask_model = np.isfinite(x_model) & np.isfinite(y_model)
    if mask_ref.sum() < 2 or mask_model.sum() < 2:
        return np.nan
    y_interp = np.interp(x_ref[mask_ref], x_model[mask_model], y_model[mask_model])
    return float(np.mean(np.abs(y_interp - y_ref[mask_ref])))


def write_curve_csv(filename: Path, z: np.ndarray, ignition_delay_ms: np.ndarray) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mixture_fraction", "ignition_delay_ms"])
        for zi, ti in zip(z, ignition_delay_ms, strict=True):
            writer.writerow([zi, ti])


def write_summary_csv(rows: list[dict[str, Any]], filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with filename.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_plots(
    results: dict[str, dict[str, Any]],
    z_ref: np.ndarray,
    tign_ref: np.ndarray,
    config: CaseConfig,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(z_ref, tign_ref, "k:", linewidth=2.5, label="Reference (Caban & Tyliszczak, 2024)")
    for name, result in results.items():
        mask = np.isfinite(result["ignition_delay_ms"])
        ax.semilogy(result["z"][mask], result["ignition_delay_ms"][mask], linewidth=2, marker="o", label=name.upper())
        if np.isfinite(result["xi_mr"]) and np.isfinite(result["tign_ms"]):
            ax.semilogy(result["xi_mr"], result["tign_ms"], "o", ms=7)
    ax.semilogy(config.reference_xi_mr, config.reference_tign_ms, "rs", ms=6, label="Reference scalar")
    ax.set_xlabel("Mixture fraction, Z [-]")
    ax.set_ylabel("Ignition delay [ms]")
    ax.set_title("H$_2$–O$_2$–N$_2$ autoignition at $T_O=1400$ K, $Y_{N_2}=0.767$")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "oxygen_nitrogen_1400K_ignition_delay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for ax, key, ylabel in [
        (axes[0], "temperature_final", "Final temperature [K]"),
        (axes[1], "Y_H2_final", r"Final $Y_{H_2}$ [-]"),
        (axes[2], "Y_H2O_final", r"Final $Y_{H_2O}$ [-]"),
    ]:
        for name, result in results.items():
            ax.plot(result["z"], result["diagnostics"][key], linewidth=2, marker="o", label=name.upper())
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("Mixture fraction, Z [-]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / "oxygen_nitrogen_1400K_final_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(
    run_frc: bool = True,
    run_fpv: bool = True,
) -> dict[str, Any]:
    config = CaseConfig()
    paths = get_default_paths(get_example_root())
    paths["figures"].mkdir(exist_ok=True)
    paths["output"].mkdir(exist_ok=True)

    z_ref, tign_ref = load_reference_curve(paths["reference"])
    z_points = z_ref.copy()

    results: dict[str, dict[str, Any]] = {}

    if run_frc:
        results["frc"] = run_branch("frc", z_points, config, paths["mechanism"])

    if run_fpv:
        if paths["table"].exists():
            results["fpv"] = run_branch("fpv", z_points, config, paths["mechanism"], table_file=paths["table"])
        else:
            print(
                f"FPV table not found at {paths['table']}. Skipping FPV branch. "
                "Generate the table first and place it there."
            )

    for name, result in results.items():
        write_curve_csv(paths["output"] / f"{name}_ignition_delay_curve.csv", result["z"], result["ignition_delay_ms"])

        diag_file = paths["output"] / f"{name}_diagnostics.csv"
        with diag_file.open("w", newline="") as f:
            writer = csv.writer(f)
            keys = ["mixture_fraction", "ignition_delay_ms", "temperature_initial"] + list(result["diagnostics"].keys())
            writer.writerow(keys)
            for i in range(len(result["z"])):
                row = [result["z"][i], result["ignition_delay_ms"][i], result["temperature_initial"][i]]
                row += [result["diagnostics"][k][i] for k in result["diagnostics"]]
                writer.writerow(row)

    summary_rows: list[dict[str, Any]] = []
    for name, result in results.items():
        summary_rows.append(
            {
                "model": name,
                "runtime_s": result["runtime_s"],
                "xi_mr": result["xi_mr"],
                "tign_ms": result["tign_ms"],
                "xi_mr_ref": config.reference_xi_mr,
                "tign_ref_ms": config.reference_tign_ms,
                "abs_error_xi_mr": abs(result["xi_mr"] - config.reference_xi_mr) if np.isfinite(result["xi_mr"]) else np.nan,
                "abs_error_tign_ms": abs(result["tign_ms"] - config.reference_tign_ms) if np.isfinite(result["tign_ms"]) else np.nan,
                "l1_tign_curve_vs_reference": l1_error(z_ref, tign_ref, result["z"], result["ignition_delay_ms"]),
            }
        )

    if "frc" in results and "fpv" in results:
        summary_rows.append(
            {
                "model": "fpv_minus_frc",
                "l1_tign_curve": l1_error(
                    results["frc"]["z"],
                    results["frc"]["ignition_delay_ms"],
                    results["fpv"]["z"],
                    results["fpv"]["ignition_delay_ms"],
                ),
                "abs_delta_xi_mr": abs(results["fpv"]["xi_mr"] - results["frc"]["xi_mr"]),
                "abs_delta_tign_ms": abs(results["fpv"]["tign_ms"] - results["frc"]["tign_ms"]),
            }
        )

    write_summary_csv(summary_rows, paths["output"] / "oxygen_nitrogen_1400K_summary.csv")
    make_plots(results, z_ref, tign_ref, config, paths["figures"])

    return {"results": results, "summary": summary_rows}


if __name__ == "__main__":
    main()
