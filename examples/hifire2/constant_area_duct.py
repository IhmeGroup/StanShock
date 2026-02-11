from __future__ import annotations

import code
from pathlib import Path

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
import time

from stanshock.components.combustor import Combustor
from stanshock.numerics.boundary_conditions import BCInput, SpecifiedFace
from stanshock.models.wall_models import SkinFriction_CompressibleReacting, HeatFlux_Compressible
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.plot import XTDiagram
from stanshock.system.geometry import Box
from stanshock.processing.initialize import InitializeConstant
import os
import glob

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

l = 0.609 #m

N_x = 500
x = np.linspace(0, l, N_x)

geometry = Box(xf=x,h=0.0698,w=0.0572)

"""
Boundary Conditions
"""
gas1 = ct.Solution(mech)
gas2 = ct.Solution(mech)
# INFLOW
M1 = 1.72
T1 = 300 
p1 = 16.0e3
gas1.TP = T1, p1  # isolator inlet solution/flow initialization
u1 = M1 * gas1.sound_speed
state1 = gas1, u1  # isolator inlet velocity, m/s

p2 = p1 * 2.5
# Mlims, plims, Tlims, ulims, rlims
tFinal = 0.01

physics_model = ThermoTable(gas1)
BC_inlet = SpecifiedFace(reference_state=(gas1.density, u1, gas1.P, (1.0)))
BC_outlet = SpecifiedFace(reference_state=(None, None, p2, None))
BCs: BCInput ={"left": BC_inlet, "right": BC_outlet}
init = InitializeConstant(geometry, physics_model, gas1, u1)
wall_models = (SkinFriction_CompressibleReacting, HeatFlux_Compressible)

try:
    ss = Combustor(
        geometry=geometry,
        wall_temperature=330.0,
        wall_models=wall_models,
        initialization=init,
        boundary_conditions=BCs,
        physics=physics_model,
        cfl=1.0,
        include_diffusion=True,
        output_every=100,
        plot_state_interval=100,
        use_double_flux=False
    )

    import traceback

    t0 = time.perf_counter()
    ss.advance_simulation(tFinal)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)
except Exception as e:
    print("An error occurred:", e)
    print("Full traceback:")
    traceback.print_exc()
finally:
    
    # ind_s = np.array(ss.pseudoshock.sf_array).flatten()
    
    # if len(ind_s) != 0:
    #     t_ps = np.array(ss.pseudoshock.t_ps).flatten()
    #     u_s = np.array(ss.pseudoshock.us).flatten()
    #     x_s = x[ind_s]
    #     sigma = np.array(ss.pseudoshock.sigma_ss).flatten()
    #     t_ps_ms = t_ps * 1000

    #     plt.figure()
    #     plt.plot(x_s, t_ps_ms,c='r')
    #     plt.xlabel('x [m]')
    #     plt.ylabel('t [ms]')
    #     plt.xlim([x[0],x[-1]])
    #     plt.tight_layout()
    #     plt.show()


    #     plt.figure()
    #     plt.plot(t_ps_ms, sigma,c='r')
    #     plt.ylabel(r"$\sigma  [(P_2 / P_1)_{SS} / (P_2 / P_1)_{NS}]$")
    #     plt.ylim([0, 1.01])
    #     plt.xlabel(r"$t$ $[\mathrm{ms}]$")
    #     plt.show()

    for diagram in ss.xt_diagrams:
        diagram.plot(figdir=figdir)

