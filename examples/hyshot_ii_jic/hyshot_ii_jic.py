from __future__ import annotations

import code
from pathlib import Path

import cantera as ct
import numpy as np
from scipy import optimize

from stanshock.components.combustor import Combustor
from stanshock.models.jicf import JICModel
from stanshock.models.wall_models import (
    CompressibleHeatFlux,
    CompressibleReactingSkinFriction,
)
from stanshock.numerics.boundary_conditions import BCInput, SpecifiedFace
from stanshock.physics.flamelet import FPVTable
from stanshock.processing.csv_writer import CSVWriter
from stanshock.processing.initialize import InitializeConstant
from stanshock.processing.plot import XTDiagram
from stanshock.system.geometry import Box

# Data
datadir = Path("./data")
datadir.mkdir(exist_ok=True)
figdir = Path("./figures")
figdir.mkdir(exist_ok=True)
(figdir / "anim").mkdir(exist_ok=True)

# Chemistry
mech = "../../data/mechanisms/h2_boivin_9sp_12r_mod.yaml"
table_file = "./h2_table/flamelet_results/H2_O2N2_p01_3_tf0300_to1367_200x2x200.h5"
gas = ct.Solution(mech)
X_ox = {"O2": 0.21, "N2": 0.79}
X_f = {"H2": 1.0}

# Specs from the HyShot II scramjet

# Geometry definition
h_const = 9.8e-3  # m
w = 75.0e-3  # m
L_const = 300.0e-3  # m
L_exhaust = 100.0e-3  # m
x_inj = 58.0e-3  # m
theta_exhaust = np.deg2rad(12)  # rad
r_f = 1.0e-3  # m
N_f = 4  # -

L = L_const + L_exhaust  # m
A_f = np.pi * r_f**2  # m^2
A_f_tot = N_f * A_f  # m^2

# Define the boundary conditions (from Tim Dawson email)
P_in = 127.444e3  # Pa
rho_in = 0.323551  # kg/m^3
U_in = 1791.05  # m/s
T_in = 1366.81  # K
M_in = 2.48942  # -
T0_f = 300.0  # K

# Time parameters
tau = L / U_in
print(f"tau = {tau:.2e} s")

# Compressible air properties
gas.TPX = T_in, P_in, X_ox
gamma_in = gas.cp / gas.cv
H_in = gas.enthalpy_mass
a_in = gas.sound_speed
M_in_comp = U_in / a_in

# State downstream of the bow shock on the injected jet
rho_2 = rho_in * (gamma_in + 1) * M_in_comp**2 / ((gamma_in - 1) * M_in_comp**2 + 2)
U_2 = U_in * rho_in / rho_2
P_2 = P_in * (2 * gamma_in * M_in_comp**2 - (gamma_in - 1)) / (gamma_in + 1)

# Stoichiometry
mdot_a = rho_in * U_in * h_const * w
mdot_O2 = 0.23291 * mdot_a
mdot_N2 = mdot_a - mdot_O2
gas.TP = 298.15, ct.one_atm
gas.set_equivalence_ratio(1.0, X_f, X_ox)
Yf_st = gas.Y[gas.species_index("H2")]
Z_st = gas.mixture_fraction(X_f, X_ox)

# Fuel inflow rate

# Get properties at combustor conditions
gas.TPX = T0_f, P_in, X_f
gamma_f = gas.cp / gas.cv
R_f = ct.gas_constant / gas.mean_molecular_weight


def f(M):
    tmp = ((gamma_f + 1) / 2) ** ((gamma_f + 1) / (2 * (gamma_f - 1)))
    return (
        tmp
        * M
        / (1 + (gamma_f - 1) / 2 * M**2) ** ((gamma_f + 1) / (2 * (gamma_f - 1)))
    )


def calc_M(P0, Pa):
    P0_choked = Pa * ((gamma_f + 1) / 2) ** (gamma_f / (gamma_f - 1))
    if P0_choked > P0:
        # Exit pressure is equal to the ambient pressure
        M = np.sqrt(2 / (gamma_f - 1) * ((P0 / Pa) ** ((gamma_f - 1) / gamma_f) - 1))
    else:
        # Exit pressure is no longer equal to the ambient pressure
        # We know based on geometry that the Mach number at the orifice is 1
        M = 1.0
    return M


def calc_mdot(P0, T0, A, Pa):
    M = calc_M(P0, Pa)
    gamma_term = gamma_f / ((gamma_f + 1) / 2) ** ((gamma_f + 1) / (2 * (gamma_f - 1)))
    return gamma_term * P0 * A / np.sqrt(gamma_f * R_f * T0) * f(M)


def P0_from_mdot(mdot, T0, A):
    def eqn(P0):
        return calc_mdot(P0, T0, A, P_in) - mdot

    result = optimize.root_scalar(eqn, x0=P_in)
    P0 = result.root
    M = calc_M(P0, P_in)
    return P0, M


def fuel_props_from_phi(phi_gl):
    if phi_gl == 0.0:
        return np.nan, 0.0, 300.0

    gas.set_equivalence_ratio(phi_gl, X_f, X_ox)
    # X_mix = gas.X
    Yf_gl = gas.Y[gas.species_index("H2")]
    mdot_f = (Yf_gl / gas.Y[gas.species_index("O2")]) * mdot_O2
    # Z_gl = gas.mixture_fraction(X_f, X_ox)

    # Compute the fuel plenum (stagnation) pressure to achieve mdot_f
    _, M_f = P0_from_mdot(mdot_f, T0_f, A_f_tot)
    T_f = T0_f * (1 + (gamma_f - 1) / 2 * M_f**2) ** (-1)
    a_f = np.sqrt(gamma_f * R_f * T_f)
    U_f = M_f * a_f
    rho_f = mdot_f / (U_f * A_f_tot)
    # gas.TDX = T_f, rho_f, X_f
    # P_f = gas.P
    # H_f = gas.enthalpy_mass

    # Compute the estimated temperature of the mixture
    # (Pressure will change but this doesn't affect the temperature)
    # gas.HPX = (mdot_a * H_in + mdot_f * H_f) / (mdot_a + mdot_f), P_in, X_mix
    # T_mix = gas.T
    # delta_T = T_mix - T_in

    return rho_f, U_f, T_f


eps_t = 1.0e-6
# t_phi_gl_schedule = np.array(
#     [
#         [0.0, 0.0],
#         [0.1 * tau, 0.0],
#         [8.0 * tau, 0.35],
#         [10.0 * tau, 0.35],
#         [14.0 * tau, 0.45],
#         [16.0 * tau, 0.45],
#     ]
# )
t_phi_gl_schedule = np.array(
    [
        [0.0, 0.0],
        [0.1 * tau - eps_t, 0.0],
        [0.1 * tau, 0.35],
        [3.0 * tau, 0.35],
        [8.0 * tau, 0.35],
        [13.0 * tau, 0.6],
        [15.0 * tau, 0.6],
    ]
)

t_f = t_phi_gl_schedule[:, 0]
phi_f = t_phi_gl_schedule[:, 1]
rho_f = np.zeros(t_phi_gl_schedule.shape[0])
U_f = np.zeros(t_phi_gl_schedule.shape[0])
T_f = np.zeros(t_phi_gl_schedule.shape[0])
for i in range(t_phi_gl_schedule.shape[0]):
    t_f[i] = t_phi_gl_schedule[i, 0]
    rho_f[i], U_f[i], T_f[i] = fuel_props_from_phi(t_phi_gl_schedule[i, 1])
# NOTE: Assuming perfect gas & isentropic choked flow, only rho_f changes with phi/mdot

# Define the grid
N_x = 200
xf = np.linspace(0, L_const + L_exhaust, N_x + 1)
xc = 0.5 * (xf[1:] + xf[:-1])
h = np.zeros_like(xf)
h[xf < L_const] = h_const
h[xf >= L_const] = h_const + (xf[xf >= L_const] - L_const) * np.tan(theta_exhaust)
geometry = Box(xf=xf, h=h, w=w, n_ghost_layers=3)


# PDF sampling parameters
dx_Z_pdf = 5.0e-3
dZ_pdf = 1.0e-2
n_bins_Z_pdf = int(np.ceil(1.0 / dZ_pdf))

# Initialize the state
gas_init = ct.Solution(mech)
gas_init.TPX = T_in, P_in, X_ox

# Define the boundary conditions
BC_inlet = SpecifiedFace(
    reference_state=(gas_init.density, U_in, gas_init.P, (1.0, 0.0, 0.0))
)
BC_outlet = "outflow"
BCs: BCInput = {"left": BC_inlet, "right": BC_outlet}

# Load the FPV table
fpv_table = FPVTable(
    table_file,
    gas,
    ox_def=X_ox,
    fuel_def=X_f,
    prog_def={"H2O": 1.0},
    p_correction=False,
    T_correction=False,
)

# #################################################################

# Build the injector model
jic = JICModel(
    x_inj=x_inj,
    x_noz=L_const,
    n_inj=N_f,
    d_inj=2 * r_f,
    t_inj=t_f,
    phi_inj=phi_f,
    rho_inj=rho_f,
    u_inj=U_f,
    T_inj=T_f,
    rho=rho_in,
    u=U_in,
    T=T_in,
    alpha=1e6,
    load_Z_3D=(datadir / "Z_3D.npy").exists(),
    load_Z_avg_var_profiles=(datadir / "Z_var_profile.npy").exists(),
    load_chemical_sources=(datadir / "omega_C_int.npy").exists(),
    load_MIB_profile=(datadir / "C_profile_MIB.npy").exists(),
    geometry=geometry,
    physics=fpv_table,
)

# Initialize and run the simulation
ss = Combustor(
    geometry=geometry,
    wall_temperature=300.0,
    wall_models=(CompressibleReactingSkinFriction(), CompressibleHeatFlux()),
    initialization=InitializeConstant(geometry, fpv_table, gas_init, U_in),
    boundary_conditions=BCs,
    source_terms=None,
    injector=jic,
    cfl=0.5,
    physics=fpv_table,
    reacting=True,
    include_diffusion=False,
    output_every=100,
    plot_state_interval=100,
    use_double_flux=False,
)

# Update CSV writer initialization to match plot_state_interval
csv_writer = CSVWriter(
    combustor=ss,
    filename=figdir / "data.csv",  # Will become test_00000.csv, test_00001.csv, etc.
    interval=100,  # Same as plot_state_interval=100
)
ss.csv_writers = [csv_writer]

plot_variables = [
    "density",
    "velocity",
    "pressure",
    "temperature",
    "mixture fraction",
    "progress variable",
    "mach",
]
ss.xt_diagrams = [XTDiagram(ss, variable, skip_steps=10) for variable in plot_variables]
ss.advance_simulation(t_f[-1])
for diagram in ss.xt_diagrams:
    diagram.plot(figdir=figdir)

code.interact(local=locals())
