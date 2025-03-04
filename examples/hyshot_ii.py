from __future__ import annotations

import code

import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from stanshock.components.combustor import Combustor

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
mech = "ohn.yaml"
gas = ct.Solution(mech)

# Specs from the HyShot II scramjet

# Geometry definition
h_const = 9.8e-3  # m
w = 75.0e-3  # m
L_const = 300.0e-3  # m
L_exhaust = 100.0e-3  # m
x_inj = 57.5e-3  # m
theta_exhaust = np.deg2rad(12)  # rad
r_f = 0.2e-3  # m
N_f = 4  # -

L = L_const + L_exhaust  # m
A_f = np.pi * r_f**2  # m^2
A_f_tot = N_f * A_f  # m^2

# Define the boundary conditions
P_in = 127.444e3  # Pa
rho_in = 0.323551  # kg/m^3
U_in = 1791.05  # m/s
T_in = 1366.81  # K
M_in = 2.48942  # -
mdot_a = rho_in * U_in * h_const * w

T_f = 250.0  # K
mdot_f = 4.4e-3  # kg/s
phi = 0.35  # -
M_f = 1.0  # -

# Define the grid
N_x = 200
x = np.linspace(0, L_const + L_exhaust, N_x)
h = np.zeros_like(x)
h[x < L_const] = h_const
h[x >= L_const] = h_const + (x[x >= L_const] - L_const) * np.tan(theta_exhaust)
A = h * w
lnA = np.log(A)
dlnAdx_data = np.gradient(lnA, x)
dlnAdx_interp = interpolate.interp1d(x, dlnAdx_data, kind="cubic")


def dlnAdx(x, t):
    return dlnAdx_interp(x)


# Time parameters
tau = L / U_in
t_end = 5 * tau
# t_end = 0.5 * tau
print(f"tau = {tau:.2e} s")
print(f"t_end = {t_end:.2e} s")

# Initialize the state
gas_init = ct.Solution(mech)
gas_init.TPX = T_in, P_in, "O2:1,N2:3.76"
initState = gas_init, U_in
W_in = gas_init.mean_molecular_weight

# Define the boundary conditions
BC_inlet = gas_init.density, U_in, gas_init.P, gas_init.Y
BC_outlet = "outflow"
BCs = (BC_inlet, BC_outlet)

# Define the fuel inflow
gas.TPX = T_f, 101325.0, "H2:1"
gamma_f = gas.cp / gas.cv
R_f = ct.gas_constant / gas.mean_molecular_weight
a_f = np.sqrt(gamma_f * R_f * T_f)
U_f = M_f * a_f
rho_f = mdot_f / (U_f * A_f_tot)
gas.TDX = T_f, rho_f, "H2:1"
P_f = gas.P
rhoE_f = P_f / (gamma_f - 1) + 0.5 * rho_f * U_f**2
rhoYH2_f = rho_f * gas.Y[gas.species_index("H2")]
W_f = gas.mean_molecular_weight

# Define the source terms
L_src = 30.0e-3
scale_factor = 3.960715337483353


def sourceTerms(rho, rhou, rhoE, rhoY, gamma, x, t):
    nsp = gas.n_species
    rhs = np.zeros([len(x), 3 + nsp])
    index = np.logical_and(x >= x_inj, x < x_inj + L_src)
    dx = x[1] - x[0]
    rhs[index, 0] = rho_f * U_f * A_f / L_src * dx / (dx * w * h[index]) * scale_factor
    rhs[index, 1] = 0.0 * U_f * A_f / L_src * dx / (dx * w * h[index]) * scale_factor
    rhs[index, 2] = rhoE_f * U_f * A_f / L_src * dx / (dx * w * h[index]) * scale_factor
    rhs[index, 3 + gas.species_index("H2")] = (
        rhoYH2_f * U_f * A_f / L_src * dx / (dx * w * h[index]) * scale_factor
    )
    return rhs


# Initialize and run the simulation
ss = Combustor(
    gas,
    dlnAdx=dlnAdx,
    initialization=("constant", initState, x),
    boundaryConditions=BCs,
    sourceTerms=sourceTerms,
    cfl=0.5,
    reacting=True,
    includeDiffusion=False,
    outputEvery=10,
    physics="FRC",
)
ss.advance_simulation(t_end)


# Plot the results
def plot_sim(ss):
    rho = ss.r
    u = ss.u
    p = ss.p
    Y = ss.Y
    T = ss.thermoTable.getTemperature(rho, p, Y)

    M = np.zeros_like(x)
    for i in range(len(x)):
        gas.TPY = T[i], p[i], Y[i, :]
        c = gas.sound_speed
        M[i] = u[i] / c

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
