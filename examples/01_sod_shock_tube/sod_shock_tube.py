from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

from stanshock.components.shocktube import ShockTube
from stanshock.numerics.boundary_conditions import BCInput
from stanshock.numerics.face_extrapolation import FifthOrderWeno, FirstOrder
from stanshock.numerics.inviscid_flux import hllc_flux_vectorized, lax_friedrichs_flux
from stanshock.physics.fluid_base import FluidState
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.csv_writer import CSVWriter
from stanshock.processing.initialize import InitializeRiemannProblem
from stanshock.system.backend import Array
from stanshock.system.geometry import Geometry


@dataclass(frozen=True)
class SchemeCase:
    slug: str
    display_name: str
    use_double_flux: bool
    flux_function: Any
    inviscid_face_extrapolator: type


SCHEME_CASES: tuple[SchemeCase, ...] = (
    SchemeCase(
        slug="weno5_hllc",
        display_name="WENO5, HLLC",
        use_double_flux=False,
        flux_function=hllc_flux_vectorized,
        inviscid_face_extrapolator=FifthOrderWeno,
    ),
    SchemeCase(
        slug="weno5_hllc_df",
        display_name="WENO5, HLLC, +DF",
        use_double_flux=True,
        flux_function=hllc_flux_vectorized,
        inviscid_face_extrapolator=FifthOrderWeno,
    ),
    SchemeCase(
        slug="o1_hllc",
        display_name="O1, HLLC",
        use_double_flux=False,
        flux_function=hllc_flux_vectorized,
        inviscid_face_extrapolator=FirstOrder,
    ),
    SchemeCase(
        slug="o1_lf",
        display_name="O1, LF",
        use_double_flux=False,
        flux_function=lax_friedrichs_flux,
        inviscid_face_extrapolator=FirstOrder,
    ),
)


SOD_CASE: dict[str, float] = {
    "t_final": 2.0,
    "L": 10.0,
    "rhoL": 1.0,
    "uL": 0.0,
    "pL": 1.0,
    "rhoR": 0.125,
    "uR": 0.0,
    "pR": 0.1,
}


def analytical_sod_solution(
    t: float,
    x: Array,
    gamma: float = 1.4,
) -> tuple[Array, Array, Array]:
    """Return analytical pressure, density, and velocity for the Sod problem."""
    x0 = 0.0

    rho4 = SOD_CASE["rhoL"]
    p4 = SOD_CASE["pL"]
    u4 = SOD_CASE["uL"]

    rho1 = SOD_CASE["rhoR"]
    p1 = SOD_CASE["pR"]
    u1 = SOD_CASE["uR"]

    c4 = np.sqrt(gamma * p4 / rho4)
    c1 = np.sqrt(gamma * p1 / rho1)

    def resid(y: Array) -> Array:
        c1_factor = np.sqrt((gamma + 1.0) / (2.0 * gamma) * (y - 1.0) + 1)
        c2_factor = (gamma - 1.0) / (2.0 * c4) * (u4 - u1 - c1 / gamma * (y - 1.0) / c1_factor)
        exponent: float = -2.0 * gamma / (gamma - 1)
        return np.array(y * (1.0 + c2_factor) ** exponent - p4 / p1)

    y0 = 0.5 * p4 / p1
    Y = float(fsolve(resid, y0)[0])

    p2 = Y * p1
    u2 = u1 + c1 / gamma * (p2 / p1 - 1) / np.sqrt(
        (gamma + 1) / (2 * gamma) * (p2 / p1 - 1) + 1
    )
    num = (gamma + 1) / (gamma - 1) + p2 / p1
    den = 1 + (gamma + 1) / (gamma - 1) * (p2 / p1)
    c2 = c1 * np.sqrt(p2 / p1 * num / den)
    shock_speed = u1 + c1 * np.sqrt((gamma + 1) / (2 * gamma) * (p2 / p1 - 1) + 1)
    rho2 = gamma * p2 / c2**2

    p3 = p2
    u3 = u2
    c3 = (gamma - 1) / 2 * (u4 - u3 + 2 / (gamma - 1) * c4)
    rho3 = gamma * p3 / c3**2

    xe1 = (u4 - c4) * t + x0
    xe2 = t * ((gamma + 1) / 2 * u3 - (gamma - 1) / 2 * u4 - c4) + x0
    xs = shock_speed * t + x0
    xc = u2 * t + x0

    u = np.full_like(x, u4)
    p = np.full_like(x, p4)
    rho = np.full_like(x, rho4)

    idx = np.where(np.logical_and(x > xe1, x <= xe2))[0]
    u[idx] = 2 / (gamma + 1) * ((x[idx] - x0) / t + (gamma - 1) / 2 * u4 + c4)
    c = u[idx] - (x[idx] - x0) / t
    p[idx] = p4 * (c / c4) ** (2 * gamma / (gamma - 1))
    rho[idx] = gamma * p[idx] / c**2

    idx = np.where(np.logical_and(x > xe2, x <= xc))[0]
    u[idx] = u3
    p[idx] = p3
    rho[idx] = rho3

    idx = np.where(np.logical_and(x > xc, x <= xs))[0]
    u[idx] = u2
    p[idx] = p2
    rho[idx] = rho2

    idx = np.where(x > xs)[0]
    u[idx] = u1
    p[idx] = p1
    rho[idx] = rho1

    return p, rho, u


def l1_error(numerical: Array, exact: Array) -> float:
    return float(np.mean(np.abs(numerical - exact)))


def format_table(rows: list[dict[str, float | str]]) -> str:
    headers = ["Case", "Runtime [s]", "L1(rho)", "L1(p)", "L1(u)"]
    table_rows = [
        [
            str(row["case"]),
            f"{row['runtime_s']:.3f}",
            f"{row['l1_rho']:.4e}",
            f"{row['l1_p']:.4e}",
            f"{row['l1_u']:.4e}",
        ]
        for row in rows
    ]
    widths = [max(len(headers[i]), *(len(r[i]) for r in table_rows)) for i in range(len(headers))]
    line = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join(
        "  ".join(r[i].ljust(widths[i]) for i in range(len(headers))) for r in table_rows
    )
    return f"{line}\n{sep}\n{body}"


def save_summary_csv(rows: list[dict[str, float | str]], filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "runtime_s", "l1_rho", "l1_p", "l1_u"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_reference_states(mech_file: Path) -> tuple[ct.Solution, ct.Solution, ct.Solution, float, float, float, float]:
    gas = ct.Solution(mech_file)
    tref = 600.0
    pref = 101325.0
    gas.TP = tref, pref
    gamma = gas.cp / gas.cv
    mean_mw = gas.mean_molecular_weight
    gas_constant = ct.gas_constant / mean_mw
    rho_ref = pref / (gas_constant * tref)
    length_ref = np.sqrt(gamma / 1.4 * pref / rho_ref)

    gas_left = ct.Solution(mech_file)
    gas_left.DP = SOD_CASE["rhoL"] * rho_ref, SOD_CASE["pL"] * pref

    gas_right = ct.Solution(mech_file)
    gas_right.DP = SOD_CASE["rhoR"] * rho_ref, SOD_CASE["pR"] * pref

    return gas, gas_left, gas_right, gamma, pref, rho_ref, length_ref


def run_scheme_case(
    case: SchemeCase,
    geometry: Geometry,
    physics: ThermoTable,
    initialization: InitializeRiemannProblem,
    boundary_conditions: BCInput,
    t_final: float,
    output_dir: Path,
    write_csv: bool,
    rho_ref: float,
    p_ref: float,
    length_ref: float,
    gamma: float,
) -> tuple[dict[str, float | str], tuple[Array, FluidState]]:
    shock_tube = ShockTube(
        geometry=geometry,
        physics=physics,
        initialization=initialization,
        boundary_conditions=boundary_conditions,
        cfl=0.9,
        output_every=100,
        use_double_flux=case.use_double_flux,
        flux_function=case.flux_function,
        inviscid_face_extrapolator=case.inviscid_face_extrapolator,
    )

    if write_csv:
        writer = CSVWriter(
            shock_tube,
            output_dir / "csv" / f"{case.slug}.csv",
            interval=0,
            variables=["x", "rho", "u", "p", "T", "mach"],
        )
        shock_tube.csv_writers.append(writer)
    else:
        writer = None

    start = time.perf_counter()
    shock_tube.advance_simulation(t_final)
    runtime_s = time.perf_counter() - start

    if writer is not None:
        writer.write_current_state()

    idx = geometry.idx_cells
    x = geometry.xc[idx] / length_ref
    state = shock_tube.state[idx]

    p_exact, rho_exact, u_exact = analytical_sod_solution(t_final, x, gamma)
    assert state.density is not None
    assert state.pressure is not None
    assert state.velocity is not None

    row: dict[str, float | str] = {
        "case": case.slug,
        "runtime_s": runtime_s,
        "l1_rho": l1_error(state.density / rho_ref, rho_exact),
        "l1_p": l1_error(state.pressure / p_ref, p_exact),
        "l1_u": l1_error(state.velocity / length_ref, u_exact),
    }
    return row, (x, state)


def plot_results(
    final_states: dict[str, tuple[Array, FluidState]],
    x_analytical: Array,
    rho_analytical: Array,
    p_analytical: Array,
    u_analytical: Array,
    rho_ref: float,
    p_ref: float,
    length_ref: float,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = (
        ("rho", r"$\rho$", rho_analytical),
        ("p", "p", p_analytical),
        ("u", "u", u_analytical),
    )

    fig_combined, axes = plt.subplots(3, 1, sharex=True, figsize=(6.0, 7.5))

    for ax, (key, ylabel, y_exact) in zip(axes, plot_specs, strict=True):
        ax.plot(x_analytical, y_exact, "k--", lw=2, label="Analytical")
        for scheme in SCHEME_CASES:
            x, state = final_states[scheme.slug]
            if key == "rho":
                assert state.density is not None
                y_num = state.density / rho_ref
            elif key == "p":
                assert state.pressure is not None
                y_num = state.pressure / p_ref
            else:
                assert state.velocity is not None
                y_num = state.velocity / length_ref
            ax.plot(x, y_num, lw=2, label=scheme.display_name)

        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(alpha=0.25)

        fig_single, ax_single = plt.subplots(figsize=(8.0, 6.4))
        ax_single.plot(x_analytical, y_exact, "k--", lw=2, label="Analytical")
        for scheme in SCHEME_CASES:
            x, state = final_states[scheme.slug]
            if key == "rho":
                assert state.density is not None
                y_num = state.density / rho_ref
            elif key == "p":
                assert state.pressure is not None
                y_num = state.pressure / p_ref
            else:
                assert state.velocity is not None
                y_num = state.velocity / length_ref
            ax_single.plot(x, y_num, lw=2, label=scheme.display_name)

        ax_single.set_xlabel("x", fontsize=14)
        ax_single.set_ylabel(ylabel, fontsize=14)
        ax_single.tick_params(axis="both", which="major", labelsize=12)
        ax_single.grid(alpha=0.25)
        ax_single.legend(loc="best", fontsize=10)
        fig_single.tight_layout()
        fig_single.savefig(figures_dir / f"sod_{key}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_single)

    axes[-1].set_xlabel("x", fontsize=14)
    axes[0].legend(loc="best", fontsize=9)
    fig_combined.tight_layout()
    fig_combined.savefig(figures_dir / "sod_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig_combined)


def main(
    plot_results_flag: bool = True,
    write_csv: bool = True,
    output_dir: str | Path | None = None,
    selected_cases: list[str] | None = None,
    n_cells: int = 200,
) -> dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    mech_file = repo_root / "data" / "mechanisms" / "Nitrogen.yaml"
    output_root = script_dir / "output" if output_dir is None else Path(output_dir)
    figures_dir = output_root / "figures"
    csv_dir = output_root / "csv"

    gas, gas_left, gas_right, gamma, p_ref, rho_ref, length_ref = build_reference_states(mech_file)

    left_state = (gas_left, SOD_CASE["uL"])
    right_state = (gas_right, SOD_CASE["uR"])

    domain_length = SOD_CASE["L"]
    xf = np.linspace(-0.5 * domain_length * length_ref, 0.5 * domain_length * length_ref, n_cells + 1)
    geometry = Geometry(xf, area=1.0)

    x_analytical = np.linspace(-0.5 * domain_length, 0.5 * domain_length, 2001)
    p_analytical, rho_analytical, u_analytical = analytical_sod_solution(
        SOD_CASE["t_final"], x_analytical, gamma
    )

    boundary_conditions: BCInput = {"left": "reflecting", "right": "reflecting"}
    physics = ThermoTable(gas)
    initialization = InitializeRiemannProblem(
        geometry, physics, left_state, right_state, 0.0
    )

    requested = set(selected_cases) if selected_cases is not None else None
    final_states: dict[str, tuple[Array, FluidState]] = {}
    summary_rows: list[dict[str, float | str]] = []

    cases_to_run = [case for case in SCHEME_CASES if requested is None or case.slug in requested]
    if not cases_to_run:
        msg = "No matching scheme cases were selected."
        raise ValueError(msg)

    print("Solving Sod shock tube problem for the following configurations:")
    for case in cases_to_run:
        print(f"  - {case.display_name} ({case.slug})")

    for case in cases_to_run:
        print(f"\nRunning {case.display_name}...")
        row, final_state = run_scheme_case(
            case=case,
            geometry=geometry,
            physics=physics,
            initialization=initialization,
            boundary_conditions=boundary_conditions,
            t_final=SOD_CASE["t_final"],
            output_dir=output_root,
            write_csv=write_csv,
            rho_ref=rho_ref,
            p_ref=p_ref,
            length_ref=length_ref,
            gamma=gamma,
        )
        summary_rows.append(row)
        final_states[case.slug] = final_state

    summary_csv = output_root / "summary.csv"
    save_summary_csv(summary_rows, summary_csv)
    print("\nError summary:")
    print(format_table(summary_rows))
    print(f"\nSaved summary table to {summary_csv}")

    if plot_results_flag:
        plot_results(
            final_states=final_states,
            x_analytical=x_analytical,
            rho_analytical=rho_analytical,
            p_analytical=p_analytical,
            u_analytical=u_analytical,
            rho_ref=rho_ref,
            p_ref=p_ref,
            length_ref=length_ref,
            figures_dir=figures_dir,
        )
        print(f"Saved figures to {figures_dir}")

    if write_csv:
        print(f"Saved final-state CSV files to {csv_dir}")

    return {
        "summary_rows": summary_rows,
        "final_states": final_states,
        "summary_csv": summary_csv,
        "figures_dir": figures_dir,
        "csv_dir": csv_dir,
    }


if __name__ == "__main__":
    main()
