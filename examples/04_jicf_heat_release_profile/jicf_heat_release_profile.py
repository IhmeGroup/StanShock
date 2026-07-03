from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

from stanshock.components.combustor import Combustor
from stanshock.models.jicf import JICModel
from stanshock.numerics.boundary_conditions import BCInput, SpecifiedFace
from stanshock.physics.flamelet import FPVTable
from stanshock.processing.csv_writer import CSVWriter
from stanshock.processing.initialize import InitializeConstant
from stanshock.processing.plot import XTDiagram
from stanshock.system.geometry import Box


def get_root() -> Path:
    return Path(__file__).resolve().parent


def get_paths(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "mechanism": root.parent.parent
        / "data"
        / "mechanisms"
        / "h2_boivin_9sp_12r_mod.yaml",
        "table": root
        / "h2_table"
        / "flamelet_results"
        / "H2_O2N2_p01_0_tf0247_to1413_200x2x200.h5",
        "table_input": root / "h2_table" / "input.toml",
        "cache_dir": root / "data",
        "reference_dir": root / "reference_data",
        "figures_dir": root / "figures",
        "output_dir": root / "output",
    }


def ensure_dirs(paths: dict[str, Path]) -> dict[str, Path]:
    paths["figures_dir"].mkdir(parents=True, exist_ok=True)
    (paths["figures_dir"] / "xt").mkdir(exist_ok=True)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    (Path.cwd() / "data").mkdir(parents=True, exist_ok=True)
    return {"figures": paths["figures_dir"], "output": paths["output_dir"]}


# ------------------------- Micka & Driscoll Case 2 -------------------------
D_F = 2.49e-3
H_TEST = 25.4e-3
W_TEST = 38.1e-3
L_UPSTREAM = 1.0e-3
L_DOMAIN = 450.0e-3
X_INJ = L_UPSTREAM

P_IN = 101325.0
RHO_IN = 0.62
U_IN = 487.0
T_IN = 1413.0
RHO_F = 0.504
U_F = 1198.0
T_F = 247.0

X_OX = {"O2": 0.21, "N2": 0.79}
X_F = {"H2": 1.0}

N_INJ = 1
N_X = 300
CFL = 0.5
OUTPUT_EVERY = 100
T_END_MULTIPLIER = 2.0
Q_NORMALIZATION_LENGTH = 300.0e-3  # m; match experimental measurement window


def build_geometry() -> Box:
    xf = np.linspace(0.0, L_DOMAIN, N_X + 1)
    h = np.full_like(xf, H_TEST)
    return Box(xf=xf, h=h, w=W_TEST, n_ghost_layers=3)


def load_reference_curve(filename: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not filename.exists():
        return None
    data = np.genfromtxt(filename, delimiter=",", names=True)
    if getattr(data, "size", 0) == 0:
        return None
    x = np.atleast_1d(np.asarray(data[data.dtype.names[0]], dtype=float))
    y = np.atleast_1d(np.asarray(data[data.dtype.names[1]], dtype=float))
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    return x[order], y[order]


def load_scalar_reference(filename: Path) -> dict[str, float]:
    if not filename.exists():
        return {}
    out: dict[str, float] = {}
    with filename.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[str(row["metric"]).strip()] = float(row["value"])
            except Exception:
                continue
    return out


def l1_error(
    x_ref: np.ndarray, y_ref: np.ndarray, x_model: np.ndarray, y_model: np.ndarray
) -> float:
    if len(x_ref) < 2 or len(x_model) < 2:
        return np.nan
    y_interp = np.interp(x_ref, x_model, y_model)
    return float(np.mean(np.abs(y_interp - y_ref)))


def write_summary_csv(rows: list[dict[str, Any]], filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with filename.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_fpv_table(paths: dict[str, Path]) -> FPVTable:
    if not paths["table"].exists():
        msg = (
            f"Missing FPV table: {paths['table']}\n"
            "Generate the flamelet table first using h2_table/input.toml."
        )
        raise FileNotFoundError(msg)
    gas = ct.Solution(str(paths["mechanism"]))
    return FPVTable(
        str(paths["table"]),
        gas,
        ox_def=X_OX,
        fuel_def=X_F,
        prog_def={"H2O": 1.0},
        p_correction=False,
        T_correction=False,
    )


def build_injector(
    geometry: Box, physics: FPVTable, paths: dict[str, Path], t_end: float
) -> JICModel:
    t_inj = np.array([0.0, t_end])
    rho_inj = np.array([RHO_F, RHO_F])

    cache_dir = paths["cache_dir"]
    return JICModel(
        fuel="H2",
        x_inj=X_INJ,
        x_noz=L_DOMAIN,
        n_inj=N_INJ,
        d_inj=D_F,
        t_inj=t_inj,
        rho_inj=rho_inj,
        u_inj=U_F,
        T_inj=T_F,
        rho=RHO_IN,
        u=U_IN,
        T=T_IN,
        alpha=1.0e6,
        datadir=cache_dir,
        geometry=geometry,
        physics=physics,
    )


def collect_profiles(combustor: Combustor) -> dict[str, np.ndarray]:
    idx = combustor.geometry.idx_cells
    state = combustor.state[idx]
    x = combustor.geometry.xc[idx]
    physics = combustor.physics

    profiles: dict[str, np.ndarray] = {
        "x": np.asarray(x),
        "pressure": np.asarray(state.pressure),
        "temperature": np.asarray(physics.get_temperature(state)),
        "velocity": np.asarray(state.velocity),
        "mach": np.asarray(
            physics.get_velocity(state) / physics.get_sound_speed(state)
        ),
    }

    if state.composition is not None:
        profiles["mixture_fraction"] = np.asarray(state.composition[:, 1])
        profiles["progress_variable"] = np.asarray(state.composition[:, 2])

    for species in ["H2", "H2O", "OH", "O2"]:
        profiles[f"Y_{species}"] = np.asarray(physics.lookup(species, state))

    return profiles


def save_profiles_csv(filename: Path, profiles: dict[str, np.ndarray]) -> None:
    keys = list(profiles.keys())
    n = len(profiles[keys[0]])
    with filename.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([profiles[k][i] for k in keys])


def integrate_profile_to_x(x: np.ndarray, y: np.ndarray, x_cutoff: float) -> float:
    """Integrate y(x) from the first x location through x_cutoff.

    If x_cutoff falls between grid points, linearly interpolate y at the cutoff
    so the normalization window is exactly the requested physical length.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return 0.0

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if x_cutoff <= x[0]:
        return 0.0

    if x_cutoff >= x[-1]:
        return float(np.trapezoid(y, x))

    keep = x < x_cutoff
    x_int = np.append(x[keep], x_cutoff)
    y_int = np.append(y[keep], np.interp(x_cutoff, x, y))

    if len(x_int) < 2:
        return 0.0
    return float(np.trapezoid(y_int, x_int))


def compute_q_proxy(combustor: Combustor) -> dict[str, np.ndarray]:
    """Compute line-integrated heat-release quantities for comparison to Micka & Driscoll.

    The paper reports q(x) in kJ mm^-1 s^-1 and gives Q = integral q dx in kJ/s.
    Internally, x is in m. The model quantity formed here is a line-integrated
    heat-release proxy, q_line = A * source_rC. If source_rC has units W/m^3,
    q_line has units W/m, so q_line / 1e6 is kJ mm^-1 s^-1.
    """
    idx = combustor.geometry.idx_cells
    state = combustor.state[idx]
    x = combustor.geometry.xc[idx]
    physics = combustor.physics

    source_prog = np.asarray(physics.get_source_terms(state))
    factor = np.asarray(
        physics.get_source_progress_variable_compressibility_factor(state)
    )
    source_rC = np.maximum(0.0, factor * state.density * source_prog)

    try:
        area = np.asarray(combustor.geometry.area(combustor.t, x))
    except Exception:
        area = np.full_like(x, H_TEST * W_TEST)

    q_line_W_per_m = np.maximum(0.0, area * source_rC)
    Q_model_W = float(np.trapezoid(q_line_W_per_m, x))
    Q_model_300mm_W = integrate_profile_to_x(x, q_line_W_per_m, Q_NORMALIZATION_LENGTH)
    q_line_kJ_per_mm_s = q_line_W_per_m / 1.0e6

    # Normalize over the first 300 mm to match the experimental measurement window.
    if Q_model_300mm_W > 0.0:
        q_over_Q_per_m = q_line_W_per_m / Q_model_300mm_W
    else:
        q_over_Q_per_m = np.zeros_like(q_line_W_per_m)

    # Use this version when the x-axis is plotted in mm; it integrates to 1 over 0--300 mm.
    q_over_Q_per_mm = q_over_Q_per_m / 1.0e3

    # Shape-only normalization: divide by the peak heat release.
    q_max_W_per_m = float(np.max(q_line_W_per_m)) if len(q_line_W_per_m) > 0 else 0.0
    if q_max_W_per_m > 0.0:
        q_over_qmax = q_line_W_per_m / q_max_W_per_m
    else:
        q_over_qmax = np.zeros_like(q_line_W_per_m)

    peak = float(np.max(q_over_Q_per_m)) if len(q_over_Q_per_m) > 0 else 0.0
    liftoff_threshold = 0.01 * peak
    liftoff_idx = (
        int(np.argmax(q_over_Q_per_m > liftoff_threshold))
        if np.any(q_over_Q_per_m > liftoff_threshold)
        else -1
    )
    x_liftoff = float(x[liftoff_idx]) if liftoff_idx >= 0 else np.nan

    cumulative = np.zeros_like(x)
    if len(x) > 1:
        cumulative[1:] = np.cumsum(
            0.5 * (q_over_Q_per_m[1:] + q_over_Q_per_m[:-1]) * np.diff(x)
        )
    x90 = np.nan
    # With 300 mm normalization, x90 means the location where 90% of Q_300 has been released.
    if cumulative[-1] >= 0.9:
        x90 = float(np.interp(0.9, cumulative, x))

    return {
        "x": np.asarray(x),
        "q_line_W_per_m": q_line_W_per_m,
        "q_line_kJ_per_mm_s": q_line_kJ_per_mm_s,
        "Q_model_W": np.array([Q_model_W]),
        "Q_model_kJ_s": np.array([Q_model_W / 1.0e3]),
        "Q_model_300mm_W": np.array([Q_model_300mm_W]),
        "Q_model_300mm_kJ_s": np.array([Q_model_300mm_W / 1.0e3]),
        "q_normalization_length_mm": np.array([1.0e3 * Q_NORMALIZATION_LENGTH]),
        "q_max_W_per_m": np.array([q_max_W_per_m]),
        "q_max_kJ_per_mm_s": np.array([q_max_W_per_m / 1.0e6]),
        "q_over_Q_per_m": q_over_Q_per_m,
        "q_over_Q_per_mm": q_over_Q_per_mm,
        "q_over_Q": q_over_Q_per_m,
        "q_over_qmax": q_over_qmax,
        "x_liftoff": np.array([x_liftoff]),
        "x90": np.array([x90]),
    }


def make_xt_diagrams(combustor: Combustor, figdir: Path) -> None:
    variables = [
        "pressure",
        "temperature",
        "mixture fraction",
        "progress variable",
        "mach",
    ]
    combustor.xt_diagrams = [
        XTDiagram(combustor, variable, skip_steps=10) for variable in variables
    ]
    for diagram in combustor.xt_diagrams:
        diagram.plot(figdir=figdir)


def make_comparison_plots(
    profiles: dict[str, np.ndarray],
    q_proxy: dict[str, np.ndarray],
    q_ref: tuple[np.ndarray, np.ndarray] | None,
    scalar_ref: dict[str, float] | None,
    figdir: Path,
) -> float:
    x_mm = 1.0e3 * profiles["x"]
    q_x_mm = 1.0e3 * q_proxy["x"]

    # Reference file is assumed to be digitized from the paper:
    # x_ref_mm [mm], y_ref [kJ mm^-1 s^-1]. For normalized comparison,
    # normalize over the same 0--300 mm window used for the model.
    x_ref_mm, y_ref = np.array([]), np.array([])
    y_ref_norm_per_mm = np.array([])
    y_ref_over_qmax = np.array([])
    Q_ref_300mm_kJ_s = np.nan
    if q_ref is not None:
        x_ref_mm, y_ref = q_ref
        Q_ref_300mm_kJ_s = integrate_profile_to_x(
            x_ref_mm, y_ref, 1.0e3 * Q_NORMALIZATION_LENGTH
        )
        if np.isfinite(Q_ref_300mm_kJ_s) and Q_ref_300mm_kJ_s > 0.0:
            y_ref_norm_per_mm = y_ref / Q_ref_300mm_kJ_s

        q_ref_max = float(np.max(y_ref)) if len(y_ref) > 0 else 0.0
        if q_ref_max > 0.0:
            y_ref_over_qmax = y_ref / q_ref_max

    # Dimensional comparison: q(x), matching the paper axis directly.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if q_ref is not None:
        ax.plot(x_ref_mm, y_ref, "o", ms=4, label="Reference")
    ax.plot(q_x_mm, q_proxy["q_line_kJ_per_mm_s"], linewidth=2.5, label="StanShock")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"$q$ [kJ mm$^{-1}$ s$^{-1}$]")
    ax.set_title("Micka & Driscoll Case 2: heat-release profile")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.0, right=300.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        figdir / "case2_q_dimensional_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Normalized comparison: q/Q. Since x is in mm, both curves are 1/mm.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(y_ref_norm_per_mm) > 0:
        ax.plot(x_ref_mm, y_ref_norm_per_mm, "o", ms=4, label="Reference")
    ax.plot(q_x_mm, q_proxy["q_over_Q_per_mm"], linewidth=2.5, label="StanShock")
    if "liftoff_mm" in scalar_ref:
        ax.axvline(
            scalar_ref["liftoff_mm"],
            color="red",
            linestyle="--",
            linewidth=1,
            label="Ref liftoff",
        )
    if "x90_mm" in scalar_ref:
        ax.axvline(
            scalar_ref["x90_mm"],
            color="purple",
            linestyle=":",
            linewidth=1,
            label="Ref x90",
        )
    if np.isfinite(q_proxy["x_liftoff"][0]):
        ax.axvline(
            1.0e3 * float(q_proxy["x_liftoff"][0]),
            color="k",
            linestyle="--",
            linewidth=1,
            label="Model liftoff",
        )
    if np.isfinite(q_proxy["x90"][0]):
        ax.axvline(
            1.0e3 * float(q_proxy["x90"][0]),
            color="gray",
            linestyle=":",
            linewidth=1,
            label="Model x90",
        )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"$q/Q$ [mm$^{-1}$]")
    ax.set_title(
        "Micka & Driscoll Case 2: normalized heat-release profile (Q over 0-300 mm)"
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.0, right=300.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "case2_q_over_Q_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Shape-only comparison: q/q_max.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(y_ref_over_qmax) > 0:
        ax.plot(x_ref_mm, y_ref_over_qmax, "o", ms=4, label="Reference")
    ax.plot(q_x_mm, q_proxy["q_over_qmax"], linewidth=2.5, label="StanShock")
    if "liftoff_mm" in scalar_ref:
        ax.axvline(
            scalar_ref["liftoff_mm"],
            color="red",
            linestyle="--",
            linewidth=1,
            label="Ref liftoff",
        )
    if np.isfinite(q_proxy["x_liftoff"][0]):
        ax.axvline(
            1.0e3 * float(q_proxy["x_liftoff"][0]),
            color="k",
            linestyle="--",
            linewidth=1,
            label="Model liftoff",
        )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"$q/q_{\max}$ [-]")
    ax.set_title("Micka & Driscoll Case 2: peak-normalized heat-release profile")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0.0, right=300.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        figdir / "case2_q_over_qmax_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(7, 9), sharex=True)
    variables = [
        ("pressure", "Pressure [Pa]"),
        ("temperature", "Temperature [K]"),
        ("mixture_fraction", "Mixture fraction [-]"),
        ("progress_variable", "Progress variable [-]"),
    ]
    for ax, (key, ylabel) in zip(axes, variables, strict=True):
        ax.plot(x_mm, profiles[key], linewidth=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("x [mm]")
    fig.tight_layout()
    fig.savefig(figdir / "case2_state_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if len(y_ref_norm_per_mm) > 0:
        return l1_error(x_ref_mm, y_ref_norm_per_mm, q_x_mm, q_proxy["q_over_Q_per_mm"])
    return np.nan


def run_case(verbose: bool = True) -> dict[str, Any]:
    root = get_root()
    paths = get_paths(root)
    local = ensure_dirs(paths)

    geometry = build_geometry()
    fpv_table = build_fpv_table(paths)
    gas_init = ct.Solution(str(paths["mechanism"]))
    gas_init.TPX = T_IN, P_IN, X_OX

    bc_inlet = SpecifiedFace(
        reference_state=(gas_init.density, U_IN, gas_init.P, (1.0, 0.0, 0.0))
    )
    boundary_conditions: BCInput = {"left": bc_inlet, "right": "outflow"}

    t_end = T_END_MULTIPLIER * (L_DOMAIN / U_IN)
    injector = build_injector(geometry, fpv_table, paths, t_end)

    combustor = Combustor(
        geometry=geometry,
        physics=fpv_table,
        initialization=InitializeConstant(geometry, fpv_table, gas_init, U_IN),
        boundary_conditions=boundary_conditions,
        reacting=True,
        injector=injector,
        include_diffusion=False,
        cfl=CFL,
        output_every=OUTPUT_EVERY,
        verbose=verbose,
        plot_state_interval=-1,
        use_double_flux=False,
    )

    csv_writer = CSVWriter(
        combustor=combustor,
        filename=local["output"] / "case2_state.csv",
        interval=0,
        variables=[
            "x",
            "pressure",
            "temperature",
            "mach",
            "mixture fraction",
            "progress variable",
            "Y_H2",
            "Y_H2O",
            "Y_OH",
        ],
    )
    combustor.csv_writers = [csv_writer]

    import time

    t0 = time.perf_counter()
    combustor.advance_simulation(t_end)
    runtime_s = time.perf_counter() - t0

    csv_writer.write_current_state()
    profiles = collect_profiles(combustor)
    save_profiles_csv(local["output"] / "case2_profiles.csv", profiles)

    q_proxy = compute_q_proxy(combustor)
    with (local["output"] / "case2_heat_release_profiles.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "x_mm",
                "q_kJ_mm^-1_s^-1",
                "q_over_Q_mm^-1",
                "q_over_Q_m^-1",
                "q_over_qmax",
            ]
        )
        for xi, qi_dim, qi_norm_mm, qi_norm_m, qi_qmax in zip(
            1.0e3 * q_proxy["x"],
            q_proxy["q_line_kJ_per_mm_s"],
            q_proxy["q_over_Q_per_mm"],
            q_proxy["q_over_Q_per_m"],
            q_proxy["q_over_qmax"],
            strict=True,
        ):
            writer.writerow([xi, qi_dim, qi_norm_mm, qi_norm_m, qi_qmax])

    with (local["output"] / "case2_q_over_Q.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_mm", "q_over_Q_mm^-1"])
        for xi, qi in zip(
            1.0e3 * q_proxy["x"], q_proxy["q_over_Q_per_mm"], strict=True
        ):
            writer.writerow([xi, qi])

    with (local["output"] / "case2_q_over_qmax.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_mm", "q_over_qmax"])
        for xi, qi in zip(1.0e3 * q_proxy["x"], q_proxy["q_over_qmax"], strict=True):
            writer.writerow([xi, qi])

    q_ref = load_reference_curve(paths["reference_dir"] / "case2_q_over_Q.csv")
    scalar_ref = load_scalar_reference(
        paths["reference_dir"] / "case2_scalar_metrics.csv"
    )
    l1_q = make_comparison_plots(profiles, q_proxy, q_ref, scalar_ref, local["figures"])

    summary_rows = [
        {
            "runtime_s": runtime_s,
            "x_liftoff_mm": 1.0e3 * float(q_proxy["x_liftoff"][0]),
            "x90_mm": 1.0e3 * float(q_proxy["x90"][0]),
            "q_over_Q_l1_error": l1_q,
            "Q_model_kJ_s": float(q_proxy["Q_model_kJ_s"][0]),
            "Q_model_300mm_kJ_s": float(q_proxy["Q_model_300mm_kJ_s"][0]),
            "q_normalization_length_mm": float(q_proxy["q_normalization_length_mm"][0]),
            "q_max_kJ_mm^-1_s^-1": float(q_proxy["q_max_kJ_per_mm_s"][0]),
            "reference_curve_loaded": q_ref is not None,
            "reference_liftoff_mm": scalar_ref.get("liftoff_mm", np.nan),
            "reference_x90_mm": scalar_ref.get("x90_mm", np.nan),
            "abs_error_liftoff_mm": abs(
                1.0e3 * float(q_proxy["x_liftoff"][0]) - scalar_ref["liftoff_mm"]
            )
            if "liftoff_mm" in scalar_ref
            else np.nan,
            "abs_error_x90_mm": abs(
                1.0e3 * float(q_proxy["x90"][0]) - scalar_ref["x90_mm"]
            )
            if "x90_mm" in scalar_ref
            else np.nan,
            "table_file": str(paths["table"]),
        }
    ]
    write_summary_csv(summary_rows, local["output"] / "case2_summary.csv")

    return {
        "runtime_s": runtime_s,
        "profiles": profiles,
        "q_proxy": q_proxy,
        "reference_curve_loaded": q_ref is not None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Micka & Driscoll JICF heat-release example."
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Reduce terminal solver logging."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_case(verbose=not args.quiet)


if __name__ == "__main__":
    main()
