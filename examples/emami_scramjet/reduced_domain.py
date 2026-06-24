from __future__ import annotations

import time
import traceback
from datetime import datetime
from pathlib import Path

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from stanshock.components.combustor import Combustor
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.plot import XTDiagram
from stanshock.processing.probe import Probe

root_dir = Path(__file__).resolve().parent / ".." / ".."
plt.style.use(root_dir / "data" / "stylelib" / "publication.mplstyle")

# Paths
datadir = Path("./data")
datadir.mkdir(exist_ok=True)
figdir = Path("./figures")
figdir.mkdir(exist_ok=True)
animdir: Path = figdir / "anim"
animdir.mkdir(exist_ok=True)
resultsdir = Path("./xt_raw")
resultsdir.mkdir(exist_ok=True)

for ext in ("*.png", "*.mp4"):
    for file in figdir.glob(pattern=ext):
        file.unlink(missing_ok=True)

    for file in animdir.glob(pattern=ext):
        file.unlink(missing_ok=True)

# TESTING
closing = True
AR_i = 4.15
pseudoshock = True
desired_img_count = 200
t_sim = 0.025
N_x = 300
"""
SIMULATION TIME PARAMETERS
"""
t_stab = t_sim / 100

t_close = 0.5 * t_sim
tFinal = t_sim + t_stab

AR_f: float = AR_i if not closing else 1

dt_estimate = 7e-7


plot_interval = int(
    10 ** np.round(np.log10(tFinal / (dt_estimate * desired_img_count)))
)


# Chemistry
mech = "data/mechanisms/N2O2HeAr.yaml"
X_amb = "O2:0.21 N2:0.79"
gas = ct.Solution(mech)
"""
GEOMETRY INPUTS
    from ("Experimental Investigation of Inlet-Combustor Isolators for a Dual-Mode Scramjet...")
"""
Liso_Hth = 12.7  # ratio of isolator length to throat height
H_th = 0.01016  # throat height
# ~~~~~RELEVANT DIMENSIONS~~~~~~
# ~~Lengths~~
lx = 0.15
L_rp = 0.248158  # Ramp length, m
H_rp = 0.048237  # Ramp height, m
d1 = np.atan2(H_rp, L_rp) + np.radians(1)  # Ramp angle,  rad
L_iso = Liso_Hth * H_th  # Isolator length

L_dd = 0.129032  # Lower diffuser length
L_cd = 0.272  # Lower combustor length
L_fp = 0.150  # Total flap length

L_fd = 0.080431  # flat section before diffuser
L_du = 0.048601  # upper diffuser length
L_cu = 0.267637  # upper comb chamber wall length
L_nu = 0.076583  # upper nozzle narrowing section len
L_fu = 0.078055  # upper nozzle flat section length

H_rp = 0.048237
H_bt = 0.012392  # bottom ramp height
H_cc = 0.067106  # combustor total height
H_noz = 0.009435  # upper nozzle height


# ~~Cowl Geometry~~
theta_cowl = np.radians(2.2)
hyp_cowl = 0.0635  # m
h_cowl = hyp_cowl * np.sin(theta_cowl)
L_cowl = h_cowl / np.tan(theta_cowl)

x_cle = L_rp - L_cowl * np.cos(theta_cowl)
y_cle = (H_th + H_rp) - h_cowl

# Upper Wall (cowl and after)
xu1 = 0
xu2 = xu1 + L_iso + L_fd
xu3 = xu2 + L_du
xu4 = xu3 + L_cu
xu5 = xu4 + L_nu
xu6 = xu5 + L_fu
x_u = np.array([xu1, xu2, xu3, xu4, xu5, xu6]) - xu1

L_tot = x_u[-1] - x_u[0]

yu1 = H_rp + H_th
yu2 = yu1
yu3 = H_cc
yu4 = H_cc
yu5 = H_cc - H_noz
yu6 = yu5
y_u = np.array([yu1, yu2, yu3, yu4, yu5, yu6])


# Lower Wall is time dependent-- see H(x,t)
xd1 = 0
xd2 = L_iso
xd3 = xd2 + L_dd
xd4 = xd3 + L_cd
x_d_incomp = np.array([xd1, xd2, xd3, xd4]) - xd1

yd1 = H_rp
yd2 = H_rp
yd3 = 0
yd4 = 0
y_d_incomp = np.array([yd1, yd2, yd3, yd4])

W = 0.0508  # Constant Scramjet Width (m)

regions = {"isolator": (x_u[0], x_u[5])}

"""
FLOW PROPERTIES
"""
gas1 = ct.Solution(mech)
gas2 = ct.Solution(mech)
# ISOLATOR INLET CONDITIONS
M_amb = 4.03  # freestream inlet Mach number (w/o ramp/cowl = 2.1993)
T_amb = 70.618476  # freestream inlet static temp, K (w/o ramp/cowl = 152.48)
p_amb = 8278.763  # freestream inlet static pressure, Pa (w/o ramp/cowl = 81741.125)

M1 = 2.05
p1 = 85515.38
T1 = 182.74


gas1.TPX = T1, p1, X_amb  # isolator inlet solution/flow initialization
u1 = M1 * gas1.sound_speed
state1 = gas1, u1  # isolator inlet velocity, m/s
# POST-SHOCK ISOLATOR CONDITIONS
gas2.TPX = T1 * 1.5, p1 * 1.5, X_amb
u2 = (M1 * 0.6) * gas2.sound_speed
state2 = gas2, u2
# NOZZLE EXIT CONDITIONS
p2 = 8278.763  # nozzle exit static pressure, Pa
physics_model = ThermoTable(gas1)

"""
BOUNDARY CONDITIONS
"""
BC_inlet = gas1.density, u1, gas1.P, None
BC_outlet = "outflow"

BCs = (BC_inlet, BC_outlet)


def flap_coords(t):
    H_exit = H_th * AR(t)
    H_fp = H_cc - H_exit - H_noz
    theta_fp = np.asin(H_fp / L_fp)
    xf = (np.cos(theta_fp) * L_fp) + x_d_incomp[-1]
    yf = np.sin(theta_fp) * L_fp
    x_flap = np.array([xf, L_tot])
    y_flap = np.array([yf, yf])
    return x_flap, y_flap


def D_H(t, x):
    return (2 * W * H(t, x)) / (W + H(t, x))


def AR(t):  # Nozzle-to-throat area ratio from Deng et al.
    t = np.atleast_1d(t)  # ensures t is always an array (1D at minimum)

    AR_vals = np.ones_like(t, dtype=float) * AR_i

    stabilized = (t >= 0) & (t < t_stab)
    closing = (t >= t_stab) & (t <= t_stab + t_close)
    constant = t > t_stab + t_close

    slope = (AR_f - AR_i) / t_close
    AR_vals[stabilized] = AR_i
    AR_vals[closing] = slope * (t[closing] - t_stab) + AR_i
    AR_vals[constant] = AR_f

    return AR_vals[0] if AR_vals.size == 1 else AR_vals


def dAR_dt(t):  # nozzle to throat area time derivative ratio (from Deng et al)
    t = np.asarray(t)
    dAstar_At_dt = np.zeros_like(t, dtype=float)
    closing = (t >= t_stab) & (t <= t_stab + t_close)
    dAstar_At_dt[closing] = (AR_f - AR_i) / (t_close)
    return dAstar_At_dt


def H(t, x):
    x = np.asarray(x)
    x_fp, y_fp = flap_coords(t)

    x_d = np.append(x_d_incomp, x_fp)
    y_d = np.append(y_d_incomp, y_fp)

    y_d_arr = np.interp(x, x_d, y_d)
    y_u_arr = np.interp(x, x_u, y_u)

    return y_u_arr - y_d_arr


# def grid_gen(t, x):
#     x = np.asarray(x)
#     x_fp, y_fp = flap_coords(t)
#     x_d = np.append(x_d_incomp, x_fp)
#     y_d = np.append(y_d_incomp, y_fp)
#     y_d_arr = np.interp(x, x_d, y_d)
#     y_u_arr = np.interp(x, x_u, y_u)
#     Ny = 500
#     x_grid, y_grid = np.meshgrid(x, np.linspace(0, 1, Ny))  # y will be scaled next

#     y_lower = np.tile(y_d_arr, (Ny, 1))
#     y_upper = np.tile(y_u_arr, (Ny, 1))
#     y_scaled = y_lower + (y_upper - y_lower) * y_grid  # maps [0,1] → [y_d, y_u]

#     # Flatten arrays
#     x_flat = x_grid.ravel() + x_cle
#     y_flat = y_scaled.ravel()
#     z_flat = np.zeros_like(x_flat)

#     # Write to CSV
#     import pandas as pd
#     df = pd.DataFrame({'x': x_flat, 'y': y_flat, 'z': z_flat})
#     df.to_csv(os.path.join(resultsdir,"flowpath_coords.csv"), index=False)


def dHdx(t, x):
    x = np.asarray(x)
    # dH_dx = np.zeros_like(x, dtype=float)
    return np.gradient(H(t, x), x)


def dHdt(t, x):
    x = np.asarray(x)
    dH_dt = np.zeros_like(x)
    x_fp, _ = flap_coords(t)
    mask_noz = (x > x_fp[0]) & (x <= x_fp[-1])
    x_noz = x[mask_noz] - x_fp[0]
    dH_dt[mask_noz] = (x_noz / L_fp) * H_th * dAR_dt(t)
    return dH_dt


def A(t, x):
    return H(t, x) * W


def dAdx(t, x):
    return W * dHdx(t, x)


def dAdt(t, x):
    return W * dHdt(t, x)


def dlnAdx(t, x):
    dlnA_dx = np.gradient(np.log(A(t, x)), x)
    dlnA_dx_interp = interpolate.interp1d(x, dlnA_dx, kind="cubic")
    return dlnA_dx_interp(x)


def dlnAdt(t, x):
    return dAdt(t, x) / A(t, x)


# Define the grid
xShock = x_u[1]
x = np.linspace(x_u[0], x_u[-1], N_x)

# grid_gen(0,x)

try:
    ss = Combustor(
        n=N_x,
        x=x,
        dlnA_dx=dlnAdx,
        dlnA_dt=dlnAdt,
        h=H,
        w=W,
        d_outer=D_H,
        regions=regions,
        wall_temperature=330.0,
        include_boundary_layer=True,
        include_pseudoshock=pseudoshock,
        initialization=("riemann", state1, state2, xShock),
        boundary_conditions=BCs,
        physics=physics_model,
        cfl=1.0,
        include_diffusion=True,
        output_every=plot_interval,
        plot_state_interval=plot_interval,
    )
    ss.probes.append(Probe(ss, 0.01, skipSteps=10, probeName="isolator_inlet"))
    ss.probes.append(
        Probe(ss, x_d_incomp[1], skipSteps=10, probeName="isolator_outlet")
    )
    ss.probes.append(
        Probe(
            ss,
            ((x_d_incomp[2] + x_d_incomp[3]) / 2),
            skipSteps=10,
            probeName="backpressure",
        )
    )

    t0 = time.perf_counter()
    plot_variables = [
        "density",
        "velocity",
        "pressure",
        "temperature",
        "mach",
    ]
    ss.xt_diagrams = [
        XTDiagram(ss, variable, skipSteps=10) for variable in plot_variables
    ]
    ss.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)
except Exception as e:
    print("An error occurred:", e)
    print("Full traceback:")
    traceback.print_exc()
finally:
    for diagram in ss.xt_diagrams:
        diagram.plot(figdir=figdir)
        diagram.save_to_csv(output_dir=resultsdir)

        plt.figure(figsize=(4, 4))
        plt.plot(ss.pseudoshock.x_pred, ss.pseudoshock.t_us, c="r", label="Predicted")
        plt.plot(
            x[ss.pseudoshock.sf_array],
            ss.pseudoshock.time_array,
            c="b",
            label="Selected",
        )
        plt.ylabel("Time [s]")
        plt.xlabel(r"$x_s$ [m]")
        plt.xlim([x[0], x[-1]])
        plt.ylim([0, max(ss.pseudoshock.t_us)])
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(figdir / "shock_location.png", dpi=300, bbox_inches="tight")
        # plt.show()
    plt.figure(figsize=(4, 4))

    # Extract time and pressure data from probes, normalize by static pressure (originally at ramp inlet, but also as back pressure)
    t_AR = np.array(ss.probes[0].t)
    t_iso_in = np.array(ss.probes[0].t) / tFinal
    p_iso_in = np.array(ss.probes[0].p) / p_amb

    t_iso_out = np.array(ss.probes[1].t) / tFinal
    p_iso_out = np.array(ss.probes[1].p) / p_amb

    t_back_p = np.array(ss.probes[2].t) / tFinal
    p_back_p = np.array(ss.probes[2].p) / p_amb

    fig, ax1 = plt.subplots()

    ax1.plot(
        t_iso_out, p_iso_out, "r", label="$\\mathrm{Isolator\\ Outlet}$", linewidth=2.0
    )
    ax1.plot(
        t_iso_in, p_iso_in, "b", label="$\\mathrm{Isolator\\ Inlet}$", linewidth=2.0
    )
    ax1.plot(
        t_back_p, p_back_p, "g", label="$\\mathrm{Back\\ Pressure}$", linewidth=2.0
    )
    ax1.set_ylim(0, 55)
    ticks = np.arange(0, 57.5, 2.5)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([f"{t:.0f}" if t % 5 == 0 else "" for t in ticks])
    xticks = np.arange(0, max(t_iso_out), 0.1)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{x:.1f}" if round(x * 10) % 5 == 0 else "" for x in xticks])

    ax2 = ax1.twinx()
    ax2.set_ylabel("$A^*/A_{th}$")
    ax2.set_ylim(0, 4.5)

    ax2.plot(t_iso_out, AR(t_AR), "k", linewidth=2.0)
    ticks2 = np.arange(0, 4.25, 0.25)
    ax2.set_yticks(ticks2)
    ax2.set_yticklabels([f"{t:.1f}" if t % 1.0 == 0 else "" for t in ticks2])

    ax1.set_xlabel("$t\\ [\\mathrm{s}]$")
    ax1.set_ylabel("$p / p_1$")
    ax1.legend(loc="lower left", ncol=3, frameon=False)
    titlestr = "Pseudoshock Active" if pseudoshock else "Pseudoshock Inactive"
    fig.suptitle(titlestr)
    filename = f"deng_unstart_{titlestr}.png"
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        figdir / f"{filename}_{time_stamp}.png",
        dpi=300,
        bbox_inches="tight",
    )
