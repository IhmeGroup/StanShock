from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ambiance import Atmosphere as atm

from inlet_moc.moc_solution import MOCSolution
from inlet_moc.planar_inlet import PlanarInlet
from inlet_moc.processing import process_solution

datadir = Path(__file__).resolve().parent
figdir = datadir / "01_figs"
figdir.mkdir(parents=True, exist_ok=True)

t_s = np.array([538.103, 538.179, 538.734, 538.805])
M0 = np.array([7.828, 7.831, 7.938, 7.938])
q0_kPa = np.array([24.88, 25.33, 31.55, 32.20])
h = np.array([34.48, 34.31, 33.05, 32.89]) * 1000

p_atm_arr = atm(h).pressure
T_atm_arr = atm(h).temperature

# mech = "air.yaml"
# gas = ct.Solution(mech)
# sol = ct.SolutionArray(gas, (1,))


alpha_deg = np.array([-5.012, 5.540, -5.081, 4.617])


"""
INLET GEOMETRY
"""
L_c = 103.6
xy_ramp = (
    np.column_stack((np.array([-355.4, 0.0, L_c]), np.array([0.0, 115.5, 115.5])))
    / 1000.0
)

xy_cowl = (
    np.column_stack((np.array([-58.4, 0.0, L_c]), np.array([125.3, 125.3, 125.3])))
    / 1000.0
)

inlet = PlanarInlet(xy_ramp, xy_cowl)

i = 0
Mach = M0[i]
theta = np.radians(-alpha_deg[i])
# theta = 0
T_amb = T_atm_arr[i]
p_amb = p_atm_arr[i]
N_idl = 100
x_stop = 0.0


soln = MOCSolution(
    inlet=inlet,
    Mach=Mach,
    theta=theta,
    T_amb=T_amb,
    p_amb=p_amb,
    N_idl=N_idl,
    x_stop=None,
    verbose=True,
    plot_during_solve=True,
    figdir=figdir,
    case_name="hyshot2",
)

print("[main] Starting solve_inlet()...")
soln.solve_inlet()
print("[main] solve_inlet() complete.")
if soln.nets:
    process_solution(soln, figdir=figdir)
else:
    print("[main] No solved nets available; skipping plot export.")
plt.show()
plt.close("all")
