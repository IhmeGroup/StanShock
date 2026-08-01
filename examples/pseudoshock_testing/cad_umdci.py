from __future__ import annotations

import traceback
from pathlib import Path

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

from stanshock.components.combustor import Combustor
from stanshock.models.wall_models import (
    CompressibleHeatFlux,
    CompressibleInertSkinFriction,
)
from stanshock.numerics.boundary_conditions import BCInput, SpecifiedFace
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.csv_writer import CSVWriter
from stanshock.processing.initialize import InitializeConstant
from stanshock.processing.plot import get_variable_info_map
from stanshock.system.geometry import Box

root_dir = Path(__file__).resolve().parents[2]
plt.style.use(root_dir / "data" / "stylelib" / "publication.mplstyle")

###

###


# Data
figdir = Path("./figures")
figdir.mkdir(exist_ok=True)
(figdir / "anim").mkdir(exist_ok=True)

# Chemistry
mech = Path(__file__).resolve().parent / "../../data/mechanisms/N2O2HeAr.yaml"


# Geometry
L = 0.609  # m
N_x = 500
x = np.linspace(0, L, N_x)

geometry = Box(xf=x, h=0.0698, w=0.0572)

"""
Boundary Conditions
"""
gas1 = ct.Solution(mech)
# INFLOW
M1 = 1.72
T1 = 300
p1 = 16.0e3
gas1.TP = T1, p1  # inlet solution/flow initialization
u1 = M1 * gas1.sound_speed  # inlet velocity, m/s

p2 = p1 * 2.5
tFinal = 0.01

physics_model = ThermoTable(gas1)
variable_info_map = get_variable_info_map(physics_model)
BC_inlet = SpecifiedFace(
    location="left", reference_state=(gas1.density, u1, gas1.P, (1.0,))
)
BC_outlet = SpecifiedFace(location="right", reference_state=(None, None, p2, None))
BCs: BCInput = {"left": [BC_inlet], "right": [BC_outlet]}
init = InitializeConstant(geometry, physics_model, gas1, u1)
wall_models = (CompressibleInertSkinFriction(), CompressibleHeatFlux())



ss = Combustor(
    geometry=geometry,
    wall_temperature=330.0,
    wall_models=wall_models,
    initialization=init,
    boundary_conditions=BCs,
    physics=physics_model,
    cfl=1.0,
    include_pseudoshock=True,
    output_every=100,
    plot_state_interval=100,
    plot_state_variables=["mach", "p", "T"],
    plot_state_variable_info_map=variable_info_map,
    use_double_flux=False,
)
ss.csv_writers = [
    CSVWriter(
        combustor=ss,
        filename=figdir / "csv" / "state.csv",
        interval=ss.plot_state_interval,
        variables=["x", "mach", "p", "T"],
        variable_info_map=variable_info_map,
    )
]
try:
    ss.advance_simulation(tFinal)
except Exception as e:
    print("An error occurred:", e)
    print("Full traceback:")
    traceback.print_exc()
finally:
    
    ind_s = np.array(ss.pseudoshock.sf_array).flatten()
    
    if len(ind_s) != 0:
        t_ps = np.array(ss.pseudoshock.t_ps).flatten()
        u_s = np.array(ss.pseudoshock.us).flatten()
        x_s = x[ind_s]
        sigma = np.array(ss.pseudoshock.sigma_ss).flatten()
        t_ps_ms = t_ps * 1000

        plt.figure()
        plt.plot(x_s, t_ps_ms,c='r')
        plt.xlabel('x [m]')
        plt.ylabel('t [ms]')
        plt.xlim([x[0],x[-1]])
        plt.tight_layout()
        plt.show()


        plt.figure()
        plt.plot(t_ps_ms, sigma,c='r')
        plt.ylabel(r"$\sigma  [(P_2 / P_1)_{SS} / (P_2 / P_1)_{NS}]$")
        plt.ylim([0, 1.01])
        plt.xlabel(r"$t$ $[\mathrm{ms}]$")
        plt.show()

    for diagram in ss.xt_diagrams:
        diagram.plot(figdir=figdir)
