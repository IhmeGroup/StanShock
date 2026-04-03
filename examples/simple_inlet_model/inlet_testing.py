from __future__ import annotations

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

from stanshock.models.inlet_diffuser import InletDiffuser
from stanshock.physics.fluid_base import FluidState
from stanshock.physics.thermotable import ThermoTable
from stanshock.system.geometry import AsymmetricBox

# Geometry definition
h_const = 9.8e-3  # m
w = 75.0e-3  # m
L_const = 300.0e-3  # m
L_exhaust = 100.0e-3  # m
x_inj = 57.5e-3  # m
theta_ramp = np.deg2rad(18.0)  # rad
theta_exhaust = np.deg2rad(12.0)  # rad
L = L_const + L_exhaust  # m
h_nozzle = h_const + L_exhaust * np.tan(theta_exhaust)

x_ramp_start = -350.0e-3  # m
y_ramp_start = -120.0e-3  # m
x_cowl_start = -55.0e-3  # m
# Adjust the combustor start location to remove the boundary layer bleed
# x_combustor_start = 0.0  # m
x_combustor_start = -y_ramp_start / np.tan(theta_ramp) + x_ramp_start

x_sim_start = 5.0e-3  # m
x_combustor_end = L_const  # m
regions = {
    "domain": (x_sim_start, x_combustor_end),
    "external": (x_ramp_start, x_sim_start),
}

# Define the 1D simulation grid
N_x = 200
xf = np.linspace(x_sim_start, x_combustor_end, N_x + 1)


xy_lower_wall = (
    np.array([x_ramp_start, x_combustor_start, L]),
    np.array([y_ramp_start, 0.0, 0.0]),
)
xy_upper_wall = (
    np.array([x_cowl_start, x_combustor_start, x_combustor_end, L]),
    np.array([h_const, h_const, h_const, h_nozzle]),
)

# geometry = AsymmetricBox(xf=xf,regions=regions, lower_wall=(xf,lower_wall_y_values), upper_wall=(xf,upper_wall_y_values)) # Geometry definition goes in here
geometry = AsymmetricBox(
    xf=xf, regions=regions, lower_wall=xy_lower_wall, upper_wall=xy_upper_wall
)

# Reference inflow boundary conditions from RANS results
theta = 3.6  # deg.
P_in = 127.444  # kPa
rho_in = 0.323551  # kg/m^3
U_in = 1791.05  # m/s
T_in = 1366.81  # K
M_in = 2.48942  # -
mdot_a = rho_in * U_in * h_const * w

# mech = "data/mechanisms/h2_boivin_9sp_12r_mod.yaml"
mech = "air.yaml"
gas = ct.Solution(mech)
sol = ct.SolutionArray(gas, (1,))
physics = ThermoTable(gas)

sol.TPY = 263.6, 2024.0, {"N2": 0.752, "O2": 0.216, "NO": 0.032}

freestream_state = FluidState(
    shape=sol.shape,
    temperature=sol.T,
    density=sol.density_mass,
    velocity=np.array([2398.0]),
    composition=sol.Y,
    sound_speed=sol.sound_speed,
)

inlet = InletDiffuser(
    angle_of_attack=np.deg2rad(theta),
    freestream=freestream_state,
    geometry=geometry,
    physics=physics,
)

n_aoa = 101
aoa_range = np.linspace(-10.0, 10, n_aoa)
temperatures = np.zeros((n_aoa,))
pressures = np.zeros((n_aoa,))
density = np.zeros((n_aoa,))
velocity = np.zeros((n_aoa,))
mach = np.zeros((n_aoa,))

for i in range(n_aoa):
    inlet.angle_of_attack = np.deg2rad(aoa_range[i])
    state = inlet.compute_combustor_inlet_properties()
    temperatures[i] = physics.get_temperature(state)[0]
    pressures[i] = physics.get_pressure(state)[0] / 1000  # kPa conversion
    density[i] = physics.get_density(state)[0]
    velocity[i] = physics.get_velocity(state)[0]
    mach[i] = velocity[i] / physics.get_sound_speed(state)[0]


plots = [
    (temperatures, T_in, "Temperature [$K$]", "T"),
    (pressures, P_in, "Pressure [$kPa$]", "P"),
    (density, rho_in, "Density [$kg/m^3$]", "rho"),
    (velocity, U_in, "Velocity [$K$]", "u"),
    (mach, M_in, "Mach Number [$-$]", "Ma"),
]

for y, y_ref, ylabel, vname in plots:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.plot(aoa_range, y)
    ax.scatter([theta], [y_ref], s=36, marker="+")

    ax.set_title("Combustor Inflow")
    ax.set_xlabel(r"AoA [$\degree$]")
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    fig.savefig(f"{vname}_vs_AoA.png")
    plt.close()
