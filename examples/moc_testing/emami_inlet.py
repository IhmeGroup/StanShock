from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inlet_moc.moc_solution import MOCSolution
from inlet_moc.planar_inlet import PlanarInlet
from inlet_moc.plot.solution import plot_stream_thrust_average
from inlet_moc.processing import process_solution
from inlet_moc.shock_solvers.char_shock import obl_shock_angle

###############################################################################
#              Emami, 1995, NASA Technical Paper 3502                         #
###############################################################################


###### GEOMETRY #####
# (all units m and rad) #

Liso_Hth = 12.7  # isolator/throat ratio
Lc_Hth = 6.25  # cowl/throat ratio
H_th = 0.01016  # throat height

# CENTERBODY #
L_rp = 0.248158  # m
H_rp = 0.048237  # m
tan_theta_rp = H_rp / L_rp
theta_rp = np.arctan(tan_theta_rp)

L_iso = Liso_Hth * H_th

x_final = 1.0  # (L_rp + L_iso normally.)

# COWL #
theta_c = np.radians(2.2)
L_c = Lc_Hth * H_th  # m
dy_c = L_c * np.sin(theta_c)
dx_c = L_c * np.cos(theta_c)

x_cle = L_rp - dx_c  # cowl leading edge
y_cle = (H_th + H_rp) - dy_c

x_cent = [0.0, L_rp, x_final]
y_cent = [0.0, H_rp, H_rp]

x_cowl = [x_cle, L_rp, x_final]
y_cowl = [y_cle, H_rp + H_th, H_rp + H_th]

centerbody = np.column_stack((x_cent, y_cent))
cowl = np.column_stack((x_cowl, y_cowl))


def H_cap(tan_beta):
    return (
        H_rp
        - dy_c
        + H_th
        + ((tan_theta_rp * tan_beta) / (tan_theta_rp - tan_beta))
        * (
            L_rp * (1 - (tan_theta_rp / tan_beta))
            + L_c * (np.sin(theta_c) / tan_beta - np.cos(theta_c))
            - H_th / tan_beta
        )
    )


def streamtube_fxn(
    M1: float,
    gamma: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    d1 = theta_rp - theta
    tan_beta_1 = np.tan(obl_shock_angle(M1, gamma, d1))

    H_cap_st = H_cap(tan_beta_1)
    x_si = H_cap_st / tan_beta_1

    x_cowl_st = np.insert(x_cowl, 0, [0.0, x_si])
    y_cowl_st = np.insert(y_cowl, 0, [H_cap_st, H_cap_st])

    lower = np.column_stack(
        (
            x_cowl_st,
            np.interp(x_cowl_st, x_cent, y_cent),
            np.zeros_like(x_cowl_st),
        )
    )
    cowl_st = np.column_stack((x_cowl_st, y_cowl_st, np.zeros_like(x_cowl_st)))
    return cowl_st, lower


datadir = Path(__file__).resolve().parent
figdir = datadir / "01_figs"
figdir.mkdir(parents=True, exist_ok=True)
su2_st_file = datadir / "emami" / "00_data" / "emami_streamthrust_su2.csv"

inlet = PlanarInlet(centerbody, cowl)

Mach = 4.03
theta = 0.0
T_amb = 70.0
p_amb = 8290.0
N_idl = 100
x_stop = 0.35


def build_solution(
    N_idl_in: int = N_idl,
    *,
    x_stop_in: float = x_stop,
    verbose: bool = False,
) -> MOCSolution:
    return MOCSolution(
        inlet=inlet,
        Mach=Mach,
        theta=np.radians(theta),
        T_amb=T_amb,
        p_amb=p_amb,
        N_idl=N_idl_in,
        x_stop=x_stop_in,
        verbose=verbose,
        plot_during_solve=False,
        figdir=figdir,
    )


def main() -> None:
    soln = build_solution()
    soln.solve_inlet()

    result = process_solution(
        soln,
        mode="analytical",
        nx=200,
        figdir=figdir,
        plot_vars=("rho", "p", "M", "T"),
    )
    print(f"[main] final_state={result.final_state}")
    print(f"[main] eta_inlet={result.eta_inlet:.6g}")
    print(f"[main] p0_loss={result.p0_loss:.6g}")

    x_num, y_cowl_num, y_cent_num = soln.streamtube
    cowl_st_a, _ = streamtube_fxn(Mach, soln.gamma, np.radians(theta))

    fig, ax = inlet.plot_inlet()
    ax.plot(cowl_st_a[:, 0], cowl_st_a[:, 1], c="r", ls="--", label="Analytical")
    ax.plot(x_num, y_cowl_num, c="b", label="Numerical")

    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(0.0, 0.06)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    legend = ax.legend(loc="lower right", fontsize=8)
    legend.set_zorder(100)
    fig.tight_layout()

    fig_comp_path = figdir / "emami_streamtube_bounds.png"
    fig.savefig(fig_comp_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    """ COMPARE RESULTS TO SU2 """

    su2 = np.genfromtxt(su2_st_file, delimiter=",", names=True)
    su2 = su2[su2["x"] <= x_stop]

    fig, axes = plot_stream_thrust_average(
        inlet,
        plot_vars=("rho", "u", "p", "mach", "t"),
        bounds=(x_num, y_cent_num, y_cowl_num),
        dataset=su2,
        global_legend="SU2 Euler",
        color="k",
    )
    fig, axes = plot_stream_thrust_average(
        inlet,
        result.average_profile,
        plot_vars=("rho", "u", "p", "mach", "t"),
        fig=fig,
        axes=axes,
        bounds=(x_num, y_cent_num, y_cowl_num),
        global_legend=f"Rotational ({soln.N_idl} IDL)",
        color="tab:red",
    )
    streamthrust_path = figdir / f"stream_thrust_overlay_{soln.N_idl}_IDL.png"
    fig.savefig(streamthrust_path, dpi=500, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
