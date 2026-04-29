from __future__ import annotations

import time
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
from stanshock.processing.initialize import InitializeConstant, InitializeRiemannProblem
from stanshock.processing.plot import XTDiagram, get_variable_info_map
from stanshock.system.backend import Array
from stanshock.system.geometry import Box

"""
Benchmark case for future implementation of pseudoshock model.
Inflow, outflow, and geometry based on UMDCI facility.
Data-Driven One Dimensional Modeling of Pseudoshocks
"""
test_pseudoshock = True

# Data
figdir = Path("./figures")
figdir.mkdir(exist_ok=True)
(figdir / "anim").mkdir(exist_ok=True)

# Chemistry
mech = Path(__file__).resolve().parent / "../../data/mechanisms/Nitrogen.yaml"

h: tuple[Array, Array] | float = 0.0698  # m
w = 0.0572  # m
L = 0.609  # m
N_x = 500

if test_pseudoshock:
    x = np.linspace(0, 5 * L, N_x)

    h_pts = np.array([h, h, h / 1.2])
    x_pts = np.array([0, 4 * L, 4 * L])
    h = (x_pts, h_pts)
else:
    x = np.linspace(0, L, N_x)

geometry = Box(xf=x, h=h, w=w)

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

physics_model = ThermoTable(gas1)
variable_info_map = get_variable_info_map(physics_model)
wall_models = (CompressibleInertSkinFriction(), CompressibleHeatFlux())

if test_pseudoshock:
    t_final = 0.1
    p2 = p1 * 3.0
    gas2 = ct.Solution(mech)
    gas2.TP = T1 * 1.6, p2
    u2 = gas2.sound_speed * 0.5
    x_shock = 4 * L
    init = InitializeRiemannProblem(
        geometry, physics_model, (gas1, u1), (gas2, u2), x_shock
    )
else:
    t_final = 0.01
    p2 = p1 * 2.5
    init = InitializeConstant(geometry, physics_model, gas1, u1)

BC_inlet = SpecifiedFace(
    location="left", reference_state=(gas1.density, u1, gas1.P, (1.0,))
)
BC_outlet = SpecifiedFace(location="right", reference_state=(None, None, p2, None))
BCs: BCInput = {"left": [BC_inlet], "right": [BC_outlet]}
init = InitializeConstant(geometry, physics_model, gas1, u1)
wall_models = (CompressibleInertSkinFriction(), CompressibleHeatFlux())

plot_variables = ["mach", "pressure", "temperature"]

ss = Combustor(
    geometry=geometry,
    wall_temperature=330.0,
    include_pseudoshock=True,
    wall_models=wall_models,
    include_pseudoshock=test_pseudoshock,
    initialization=init,
    boundary_conditions=BCs,
    physics=physics_model,
    cfl=1.0,
    include_diffusion=False,
    output_every=100,
    plot_state_interval=100,
    plot_state_variables=plot_variables,
    plot_state_variable_info_map=variable_info_map,
    use_double_flux=False,
)
ss.csv_writers = [
    CSVWriter(
        combustor=ss,
        filename=figdir / "csv" / "state.csv",
        interval=ss.plot_state_interval,
        variables=["x", *plot_variables],
        variable_info_map=variable_info_map,
    ),
]

plot_variables = [
    "density",
    "velocity",
    "pressure",
    "temperature",
    "mach",
]
ss.xt_diagrams = [XTDiagram(ss, variable, skip_steps=10) for variable in plot_variables]

try:
    t0 = time.perf_counter()
    ss.advance_simulation(t_final)
    t1 = time.perf_counter()
    print("The process took ", t1 - t0)
except Exception as e:
    print("An error occurred:", e)
    print("Full traceback:")
    traceback.print_exc()
finally:
    # Plot the pseudoshock movement
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

    # Plot the spatiotemporal contours
    for diagram in ss.xt_diagrams:
        diagram.plot(figdir=figdir)

    if test_pseudoshock:
        t_ps = np.array(ss.pseudoshock.t_ps).flatten()
        ind_s = np.array(ss.pseudoshock.sf_array).flatten()
        u_s = np.array(ss.pseudoshock.us).flatten()
        x_s = x[ind_s]
        t_ps_ms = t_ps * 1000

        fig, ax = plt.subplots()
        ax.plot(x_s, t_ps_ms, c="r")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("t [ms]")
        ax.set_xlim((x[0], x[-1]))
        fig.tight_layout()
        plt.show()
