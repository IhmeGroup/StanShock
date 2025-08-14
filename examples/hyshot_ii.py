from __future__ import annotations

import code

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np

from stanshock.components.combustor import Combustor
from stanshock.numerics.boundary_conditions import Inflow
from stanshock.physics.thermotable import ThermoTable
from stanshock.system.base import RightHandSide
from stanshock.system.geometry import Box

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


def add_h_plot(ax):
    ax1 = ax.twinx()
    ax1.plot(x * scale, h * scale, "k", linestyle="--")
    ax1.axhline(0, color="k", linestyle="--")
    ax1.set_aspect("equal")
    ax1.set_ylabel("h [mm]")
    return ax1


# Chemistry
mech = "../data/mechanisms/h2_boivin_9sp_12r_mod.yaml"
gas = ct.Solution(mech)

# Specs from the HyShot II scramjet

# Geometry definition
h_const = 9.8e-3  # m
w = 75.0e-3  # m
L_const = 300.0e-3  # m
L_exhaust = 100.0e-3  # m
x_inj = 57.5e-3  # m
theta_exhaust = np.deg2rad(12)  # rad
L = L_const + L_exhaust  # m

# Define the boundary conditions
P_in = 127.444e3  # Pa
rho_in = 0.323551  # kg/m^3
U_in = 1791.05  # m/s
T_in = 1366.81  # K
M_in = 2.48942  # -
mdot_a = rho_in * U_in * h_const * w

# Define the grid
N_x = 200
x = np.linspace(0, L_const + L_exhaust, N_x)
h = np.zeros_like(x)
h[x < L_const] = h_const
h[x >= L_const] = h_const + (x[x >= L_const] - L_const) * np.tan(theta_exhaust)
geometry = Box(x=x, h=h, w=w)

# Time parameters
tau = L / U_in
t_end = 5 * tau
# t_end = 0.5 * tau
print(f"tau = {tau:.2e} s")
print(f"t_end = {t_end:.2e} s")

# Initialize the state
gas_init = ct.Solution(mech)
gas_init.TPX = T_in, P_in, "O2:1,N2:3.76"
W_in = gas_init.mean_molecular_weight

# Define the boundary conditions
BC_inlet = Inflow(reference_state=(gas_init.density, U_in, gas_init.P, gas_init.Y))
BC_outlet = "outflow"
BCs = (BC_inlet, BC_outlet)


# Define the fuel inflow
class HydrogenInjection(RightHandSide):
    def __init__(self, gas, geometry):
        self.geometry = geometry

        # Injector area
        r_f = 0.2e-3  # m
        N_f = 4  # -
        A_f = np.pi * r_f**2  # m^2
        A_f_tot = N_f * A_f  # m^2

        # Fuel properties
        T_f = 250.0  # K
        mdot_f = 4.4e-3  # kg/s
        # phi = 0.35  # -
        M_f = 1.0  # -
        gas.TPX = T_f, 101325.0, "H2:1"
        gamma_f = gas.cp / gas.cv
        R_f = ct.gas_constant / gas.mean_molecular_weight
        a_f = np.sqrt(gamma_f * R_f * T_f)
        U_f = M_f * a_f
        rho_f = mdot_f / (U_f * A_f_tot)
        gas.TDX = T_f, rho_f, "H2:1"
        P_f = gas.P
        # W_f = gas.mean_molecular_weight

        self.rho_f = rho_f
        self.rhoE_f = P_f / (gamma_f - 1) + 0.5 * rho_f * U_f**2
        self.rhoYH2_f = rho_f * gas.Y[gas.species_index("H2")]
        self.U_f = U_f
        self.A_f = A_f_tot

        # Define the source terms
        self.L_src = 30.0e-3
        self.scale_factor = 3.960715337483353

    def source(self, t, _state_array, _gamma, x):
        rho_f = self.rho_f
        U_f = self.U_f
        A_f = self.A_f
        rhoE_f = self.rhoE_f
        rhoYH2_f = self.rhoYH2_f
        L_src = self.L_src
        scale_factor = self.scale_factor

        nsp = gas.n_species
        rhs = np.zeros([len(x), 2 + nsp])
        index = np.logical_and(x >= x_inj, x < x_inj + L_src)
        dx = x[1] - x[0]
        area = self.geometry.area(t, x[index])

        rhs[index, 0] = rho_f * U_f * A_f / L_src * dx / (dx * area) * scale_factor
        rhs[index, 1] = rhoE_f * U_f * A_f / L_src * dx / (dx * area) * scale_factor
        rhs[index, 2 + gas.species_index("H2")] = (
            rhoYH2_f * U_f * A_f / L_src * dx / (dx * area) * scale_factor
        )
        return rhs


# Initialize and run the simulation
ss = Combustor(
    x=x,
    geometry=geometry,
    initialization=("constant", gas_init, U_in),
    boundary_conditions=BCs,
    source_terms=HydrogenInjection(gas, geometry),
    cfl=0.5,
    reacting=True,
    include_diffusion=False,
    output_every=10,
    physics=ThermoTable(gas),
)
ss.advance_simulation(t_end)


# Plot the results
def plot_sim(ss):
    x = ss.geometry.x

    idx = ss.idx_cells
    rho = ss.state.density[idx]
    u = ss.state.velocity[idx]
    p = ss.state.pressure[idx]
    Y = ss.state.composition[idx]
    T = ss.physics.get_temperature(ss.state)[idx]
    c = ss.physics.get_sound_speed(ss.state)[idx]
    M = u / c

    fig, ax = plt.subplots(7, 1, sharex=True, figsize=(6, 8))
    ax[0].plot(x * scale, rho)
    ax[0].set_ymargin(0.1)
    ax[0].set_ylabel(r"$\rho$ [kg/m$^3$]")
    add_h_plot(ax[0])

    ax[1].plot(x * scale, u)
    ax[1].set_ymargin(0.1)
    ax[1].set_ylabel(r"$u$ [m/s]")
    add_h_plot(ax[1])

    ax[2].plot(x * scale, p)
    ax[2].set_ymargin(0.1)
    ax[2].set_ylabel(r"$p$ [Pa]")
    add_h_plot(ax[2])

    ax[3].plot(x * scale, T)
    ax[3].set_ymargin(0.1)
    ax[3].set_ylabel(r"$T$ [K]")
    add_h_plot(ax[3])

    ax[4].plot(x * scale, M)
    ax[4].set_ymargin(0.1)
    ax[4].set_ylabel(r"$M$ [-]")
    add_h_plot(ax[4])

    ax[5].plot(x * scale, Y[:, gas.species_index("H2")])
    ax[5].set_ymargin(0.1)
    ax[5].set_ylabel(r"$Y_{\mathrm{H}_2}$ [-]")
    add_h_plot(ax[5])

    ax[6].plot(x * scale, Y[:, gas.species_index("H2O")])
    ax[6].set_ymargin(0.1)
    ax[6].set_ylabel(r"$Y_{\mathrm{H}_2\mathrm{O}}$ [-]")
    add_h_plot(ax[6])

    ax[6].set_xlabel("x [mm]")

    plt.tight_layout()
    plt.savefig("hyshot_ii.png", bbox_inches="tight", dpi=300)


plot_sim(ss)

code.interact(local=locals())
