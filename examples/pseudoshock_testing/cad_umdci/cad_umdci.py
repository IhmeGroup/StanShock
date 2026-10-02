from __future__ import annotations

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
from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.processing.initialize import InitializeRiemannProblem
from stanshock.processing.plot import get_variable_info_map
from stanshock.system.geometry import Box

###############################################################################
#              Edelman, 2023, Assessment of Pseudoshock Models... JPP         #
###############################################################################
### Data ###
data_dir = Path(__file__).resolve().parents[3] / "data"
plt.style.use(data_dir / "stylelib" / "publication.mplstyle")
mech = data_dir / "mechanisms" / "N2O2HeAr.yaml"
piv_data = data_dir / "validation" / "cad_umdci_piv.csv"

### Simulation Output ###
case_dir = Path()
figdir = case_dir / "figures"
animdir = figdir / "anim"
csvdir = figdir / "csv"
figdir.mkdir(exist_ok=True)
animdir.mkdir(exist_ok=True)
csvdir.mkdir(exist_ok=True)
test_figdir = case_dir / "test_figs"


### Geometry ###
N_x = 120
L, H, W = 0.609, 0.0698, 0.0572
x = np.linspace(0, L, N_x)
geometry = Box(xf=x, h=H, w=W)

### Boundary Conditions ###
gas = ct.Solution(mech)
physics = CanteraInterface(gas)
X_air = {"N2": 0.79, "O2": 0.21}


# inlet
M1, T1, p1 = 1.86, 174.0, 13.7e3
gas1 = ct.Solution(mech)
gas1.TPX = T1, p1, X_air
u1 = M1 * gas1.sound_speed

# outlet
M2, T2, p2 = 0.740, 266.0, p1 * 3.1
gas2 = ct.Solution(mech)
gas2.TPX = T2, p2, X_air
u2 = M2 * gas2.sound_speed

BC_inlet = SpecifiedFace(
    location="left", reference_state=(gas1.density, u1, gas1.P, gas1.Y)
)
BC_outlet = SpecifiedFace(location="right", reference_state=(None, None, p2, None))
BCs: BCInput = {"left": [BC_inlet], "right": [BC_outlet]}


### Initialization ###
x_shock = L / 2
init = InitializeRiemannProblem(geometry, physics, (gas1, u1), (gas2, u2), x_shock)


### Sim params ###
interval = 10
tFinal = 2e-3

variable_info_map = get_variable_info_map(physics)
wall_models = (CompressibleInertSkinFriction(), CompressibleHeatFlux())

plot_variables = ["mach", "density", "pressure", "temperature"]

ss = Combustor(
    geometry=geometry,
    wall_temperature=None,
    wall_models=wall_models,
    include_pseudoshock=True,
    initialization=init,
    boundary_conditions=BCs,
    physics=physics,
    cfl=1.0,
    output_every=interval,
    plot_state_interval=interval,
    plot_state_variables=plot_variables,
    plot_state_variable_info_map=variable_info_map,
    use_double_flux=False,
)


ss.advance_simulation(tFinal)

### Validation plot ###
col_exp, lw_exp, label_exp = "r", 1.0, "Measured"
col_ss, lw_ss, label_ss = "k", 1.0, "Stanshock"

ylabels = [r"$p~[\mathrm{kPa}]$", r"$Ma~[\mathrm{-}]$"]
ylims = [[10.0, 80.0], [0.5, 2.0]]
xlabel, xlims, xticks = r"$x/H~[\mathrm{-}]$", [0.0, 9.0], np.arange(0, 10.0)


# Measured data
pseudoshock = ss.pseudoshock
x_sf = np.array(pseudoshock.x_sf).flatten()

p_sf = physics.get_pressure(pseudoshock.state_0)
xH_local_exp, p_p1_exp, M_exp, _ = np.genfromtxt(
    piv_data,
    delimiter=",",
    names=True,
    unpack=True,
)
p_exp = p_sf * p_p1_exp / 1e3
xH_exp = xH_local_exp + x_sf[-1] / H

# Stanshock data
idx = ss.geometry.idx_cells
xH = ss.geometry.xc[idx] / H

state = ss.state[idx]
p = 1e-3 * physics.get_pressure(state)
M = physics.get_velocity(state) / physics.get_sound_speed(state)

fig, axs = plt.subplots(1, 2, figsize=(6, 3))
stanshock_data = [p, M]
measured_data = [p_exp, M_exp]

for ax, y_ss, y_exp, ylabel, ylim in zip(
    axs, stanshock_data, measured_data, ylabels, ylims, strict=False
):
    ax.plot(xH, y_ss, c=col_ss, lw=lw_ss, label=label_ss)
    ax.plot(xH_exp, y_exp, c=col_exp, lw=lw_exp, label=label_exp)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlims)
    ax.set_xticks(xticks)
    ax.legend(loc="best")
    ax.grid(which="major", axis="x", lw=0.8, alpha=0.6)
    ax.grid(True, which="both", axis="y", lw=0.4, alpha=0.3)

fig.tight_layout()
fig.savefig(animdir / "umdci_validation.png", bbox_inches="tight", dpi=300)

for diagram in ss.xt_diagrams:
    diagram.plot(figdir=animdir)
