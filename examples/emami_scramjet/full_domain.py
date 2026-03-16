from __future__ import annotations

import time
from pathlib import Path

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

from stanshock.components.combustor import Combustor
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.plot import SnapshotDiagram, XTDiagram

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0

XSMALL_SIZE = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=XSMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Plotting utilities
scale = 1e3

# Data
datadir = Path("./data")
datadir.mkdir(exist_ok=True)
figdir = Path("./figures")
figdir.mkdir(exist_ok=True)
(figdir / "anim").mkdir(exist_ok=True)

# Chemistry
mech = "data/mechanisms/Nitrogen.yaml"

gas = ct.Solution(mech)
"""
GEOMETRY INPUTS
    from ("Experimental Investigation of Inlet-Combustor Isolators for a Dual-Mode Scramjet...")
        Note that the inlet conditions represent post shock properties (assuming 3 oblique shocks)
"""
H_ramp = 0.04826
theta_diff = 20  # diffuser angle
H_th = 0.01016  # throat height, m
L_H_th = 12.7  # isolator to throat height ratio
L_iso = L_H_th * H_th  # isolator length, m
L_diff = H_ramp / np.tan(np.radians(theta_diff))
H_diff = H_th + H_ramp
theta_diff2 = 14.11


H_cc = 0.06985  # total combustion chamber height
L_diff2 = (H_cc - H_diff) / np.tan(np.radians(theta_diff2))
L_cc = 0.2636  # total combustion chamber length
H_diff2 = L_diff2 * np.tan(np.radians(theta_diff2))

L_flap = 0.150  # nozzle length when theta_flap = 0
theta_flap = 10  # bottom flap deflection angle
theta_noz = np.degrees(np.atan2((H_cc - H_diff), L_flap))
H_noz = L_flap * np.tan(np.radians(theta_noz))
H_flap = L_flap * np.tan(np.radians(theta_flap))
theta_flap_max = np.degrees(
    np.atan2((H_cc - L_flap * np.tan(np.radians(theta_noz))), L_flap)
)
# print(theta_flap_max) 21.279 degrees
H_star = H_cc - H_noz - H_flap
print("A*/A_th = %0.2f" % (H_star / H_th))
print(L_diff + L_diff2)

x1 = L_iso  # Isolator End            (m)
x2 = x1 + L_diff  # Diffuser 1 End          (m)
x3 = x2 + L_diff2  # Diffuser 2 End          (m)
x4 = x3 + L_cc  # Combustion Chamber End  (m)
x5 = x4 + L_flap  # Nozzle End              (m)
L = x5  # Entire Scramjet Length  (m)

W = 0.0508  # Constant Scramjet Width (m)


"""
AMBIENT CONDITIONS
"""
gas1 = ct.Solution(mech)
gas2 = ct.Solution(mech)
# ISOLATOR INLET CONDITIONS
M1 = 2.1993  # isolator inlet Mach number
T1 = 152.48  # isolator inlet static temp, K
p1 = 81741.125  # isolator inlet static pressure, Pa
gas1.TP = T1, p1  # isolator inlet solution/flow initialization
u1 = M1 * gas1.sound_speed
state1 = gas1, u1  # isolator inlet velocity, m/s
# POST-SHOCK ISOLATOR CONDITIONS
gas2.TP = T1 * 1.770, p1 * 5.375
u2 = (M1 * 0.542) * gas2.sound_speed
state2 = gas2, u2
# NOZZLE EXIT CONDITIONS
p2 = 8278.763  # nozzle exit static pressure, Pa

# Time parameters
t_stab = 0.0025
t_close = 0.8  # duration of closing nozzle
tFinal = t_stab
AR_i = 4.15
AR_f = 1

physics_model = ThermoTable(gas1)

# Define the grid
N_x = 1000
xShock = 0.1 * L_iso


def D_H(t, x):
    return (2 * W * H(t, x)) / (W + H(t, x))


def AR(t):  # nozzle to throat area ratio (from Deng et al)
    t = np.asarray(t)
    AR = np.ones_like(t, dtype=float)
    stabilized = (t >= 0) & (t < t_stab)
    closing = (t >= t_stab) & (t <= t_stab + t_close)
    constant = t > t_stab + t_close
    slope = (AR_f - AR_i) / (t_close)
    AR[stabilized] = AR_i
    AR[closing] = slope * (t[closing] - t_stab) + AR_i
    AR[constant] = 1
    return AR


def dAR_dt(t):  # nozzle to throat area time derivative ratio (from Deng et al)
    # time1 = time.perf_counter()
    t = np.asarray(t)
    dAstar_At_dt = np.zeros_like(t, dtype=float)
    closing = (t >= t_stab) & (t <= t_stab + t_close)
    dAstar_At_dt[closing] = (AR_f - AR_i) / (t_close)
    # time2 = time.perf_counter()
    # if t >= t_stab: print("It took %0.2e sec"%(time2-time1))
    return dAstar_At_dt


def H(t, x):
    x = np.asarray(x)  # ensure x is an array
    heights = np.zeros_like(x, dtype=float)

    mask_iso = (x >= 0) & (x <= x1)
    mask_diff = (x > x1) & (x <= x2)
    mask_diff2 = (x > x2) & (x <= x3)
    mask_cc = (x > x3) & (x <= x4)
    mask_noz = (x > x4) & (x <= x5)

    heights[mask_iso] = H_th

    H_diff_x = H_th + (x[mask_diff] - x1) * np.tan(np.radians(theta_diff))
    heights[mask_diff] = H_diff_x

    H_diff2_x = H_diff + (x[mask_diff2] - x2) * np.tan(np.radians(theta_diff2))
    heights[mask_diff2] = H_diff2_x

    heights[mask_cc] = H_cc

    x_noz = x[mask_noz] - x4
    h_noz_up = x_noz * np.tan(np.radians(theta_noz))
    h_noz_down = x_noz * ((H_cc - H_noz - H_th * AR(t)) / L_flap)
    theta_flap = np.atan2((H_cc - H_noz - H_th * AR(t)), L_flap)
    # print("theta_flap = %0.2f" %(np.degrees(theta_flap)))
    h_noz_down = x_noz * np.tan(theta_flap)
    H_star = H_cc - h_noz_up - h_noz_down
    heights[mask_noz] = H_star
    return heights


def dHdx(t, x):
    x = np.asarray(x)  # ensure x is an array
    dH_dx = np.zeros_like(x, dtype=float)
    mask_iso = (x >= 0) & (x <= x1)
    mask_diff = (x > x1) & (x <= x2)
    mask_diff2 = (x > x2) & (x <= x3)
    mask_cc = (x > x3) & (x <= x4)
    mask_noz = (x > x4) & (x <= x5)

    dH_dx[mask_iso] = 0
    dH_dx[mask_diff] = (H(x2, t) - H(x1, t)) / (x2 - x1)
    dH_dx[mask_diff2] = (H(x3, t) - H(x2, t)) / (x3 - x2)
    dH_dx[mask_cc] = (H(x4, t) - H(x3, t)) / (x4 - x3)
    dH_dx[mask_noz] = (H(x5, t) - H(x4, t)) / (x5 - x4)
    return dH_dx


def dHdt(t, x):
    x = np.asarray(x)
    dH_dt = np.zeros_like(x)
    mask_noz = (x > x4) & (x <= x5)
    x_noz = x[mask_noz] - x4
    dH_dt[mask_noz] = (x_noz / L_flap) * H_th * dAR_dt(t)
    return dH_dt


x = np.linspace(0, x5, N_x)


def A(t, x):
    return H(t, x) * W


def dAdx(t, x):
    return W * dHdx(t, x)


def dAdt(t, x):
    return W * dHdt(t, x)


def dlnAdx(t, x):
    return dAdx(t, x) / A(t, x)


def dlnAdt(t, x):
    return dAdt(t, x) / A(t, x)


# Define the boundary conditions
BC_inlet = gas1.density, u1, gas1.P, None
BC_outlet = None, None, p2, None

BCs = (BC_inlet, BC_outlet)

# plt.figure()
# plt.plot(x, A(x,0))
# plt.show()

try:
    # Initialize and run the simulation
    ss = Combustor(
        n=N_x,
        x=x,
        dlnA_dx=dlnAdx,
        dlnA_dt=dlnAdt,
        d_outer=D_H,
        wall_temperature=330.0,
        include_boundary_layer=True,
        include_pseudoshock=True,
        initialization=("riemann", state1, state2, xShock),
        boundary_conditions=BCs,
        physics=physics_model,
        cfl=0.5,
        include_diffusion=True,
        output_every=1000,
        plot_state_interval=1000,
    )

    import traceback

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

    ss.snapshot_diagrams = [
        SnapshotDiagram(ss, variable, skipSteps=10) for variable in plot_variables
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
    # code.interact(local=locals())
