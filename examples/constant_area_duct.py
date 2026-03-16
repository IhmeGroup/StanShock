from __future__ import annotations

import time
import traceback
from pathlib import Path

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

from stanshock.components.combustor import Combustor
from stanshock.numerics.boundary_conditions import SpecifiedFace
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.initialize import InitializeRiemannProblem
from stanshock.processing.plot import XTDiagram
from stanshock.system.geometry import Box

# Data
data_dir = Path(__file__).resolve().parent / "../data"
figdir = Path("./figures")
figdir.mkdir(exist_ok=True)
(figdir / "anim").mkdir(exist_ok=True)

for ext in ("*.png", "*.mp4"):
    for file in Path("./figures/anim").glob(ext):
        file.unlink()


# Chemistry
mech = data_dir / "mechanisms/Nitrogen.yaml"

h1 = 0.0698
h2 = h1 / 1.2

w = 0.0572
L = 0.609  # m
N_x = 500
xShock = 4 * L
x = np.linspace(0, 5 * L, N_x)
# h_arr = np.ones_like(x) * h
h_pts = np.array([h1, h1, h2])
x_pts = np.array([0, 4 * L, 5 * L])
h = np.interp(x, x_pts, h_pts)
geometry = Box(xf=x, h=h, w=w)


"""
Boundary Conditions
"""
gas1 = ct.Solution(mech)
gas2 = ct.Solution(mech)
# INFLOW
M1 = 1.72
# M1 = 4.5
T1 = 300
# p1 = 81741.125
p1 = 16.0e3
gas1.TP = T1, p1  # isolator inlet solution/flow initialization
u1 = M1 * gas1.sound_speed
state1 = gas1, u1  # isolator inlet velocity, m/s

gas2.TP = T1 * 1.6, p1 * 3.0
u2 = gas2.sound_speed * 0.5
state2 = gas2, u2
limits = [T1, p1, gas1.density, [0, 3], [0, 5], [0, 3], [0, 0], [0, 3]]
# Mlims, plims, Tlims, ulims, rlims
tFinal = 0.1


physics_model = ThermoTable(gas1)
BC_inlet = SpecifiedFace(reference_state=(gas1.density, u1, gas1.P, (1.0,)))
BC_outlet = SpecifiedFace(
    reference_state=(gas2.density, None, gas2.P, (1.0,)), location="right"
)


# Initialize and run the simulation
ss = Combustor(
    geometry=geometry,
    wall_temperature=330.0,
    # include_pseudoshock=True,
    initialization=InitializeRiemannProblem(
        geometry, physics_model, state1, state2, xShock
    ),
    boundary_conditions={"left": BC_inlet, "right": BC_outlet},
    physics=physics_model,
    cfl=1.0,
    include_diffusion=True,
    output_every=200,
    plot_state_interval=50,
    # limits=limits,
)
try:
    t0 = time.perf_counter()
    plot_variables = [
        "density",
        "velocity",
        "pressure",
        "temperature",
        "mach",
    ]
    ss.xt_diagrams = [
        XTDiagram(ss, variable, skip_steps=10) for variable in plot_variables
    ]

    ss.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)
except Exception as e:
    print("An error occurred:", e)
    print("Full traceback:")
    traceback.print_exc()
finally:
    t_ps = np.array(ss.pseudoshock.t_ps).flatten()
    ind_s = np.array(ss.pseudoshock.sf_array).flatten()
    u_s = np.array(ss.pseudoshock.us).flatten()
    x_s = x[ind_s]
    t_ps_ms = t_ps * 1000
    plt.figure()
    plt.plot(x_s, t_ps_ms, c="r")
    plt.xlabel("x [m]")
    plt.ylabel("t [ms]")
    plt.xlim([x[0], x[-1]])
    plt.tight_layout()
    plt.show()
    for diagram in ss.xt_diagrams:
        diagram.plot(figdir=figdir)
    # code.interact(local=locals())
