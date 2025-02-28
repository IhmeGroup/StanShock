#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
    Copyright 2017 Kevin Grogan
    Copyright 2024 Matthew Bonanni

    This file is part of StanScram.

    StanScram is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License.

    StanScram is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with StanScram.  If not, see <https://www.gnu.org/licenses/>.
'''
import os
import pickle
from tqdm import tqdm
from typing import Optional
import numpy as np
from scipy import interpolate, integrate, optimize
import cantera as ct
import matplotlib.pyplot as plt

from StanScram.stanScram import stanScram
from StanScram.fpv_table import FPVTable
from StanScram.jet_in_crossflow import JICModel
from StanScram.monte_carlo_sampler_2d import MonteCarloSampler2D

import matplotlib.font_manager
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

XSMALL_SIZE = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=XSMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Plotting utilities
scale = 1e3
def add_h_plot(ax):
    ax1 = ax.twinx()
    ax1.plot(x*scale, h*scale, 'k', linestyle='--')
    ax1.axhline(0, color='k', linestyle='--')
    ax1.set_aspect('equal')
    ax1.set_ylabel('h [mm]')
    return ax1

# Data
datadir = "./data"
figdir = "./figures"

# Chemistry
# mech = "./HydrogenAirNOx_15sp-47st.yaml"
mech = "./h2_boivin_9sp_12r_mod.yaml"
table_file = "./H2_O2_p01_3_tf0250_to1367_200x2x200.h5"
gas = ct.Solution(mech)
X_ox = "O2:0.21,N2:0.79"
X_f = "H2:1"

# Specs from the HyShot II scramjet

# Geometry definition
h_const       =   9.8e-3       # m
w             =  75.0e-3       # m
L_const       = 300.0e-3       # m
L_exhaust     = 100.0e-3       # m
x_inj         =  58.0e-3       # m
theta_exhaust = np.deg2rad(12) # rad
r_f           =   1.0e-3       # m
N_f           =   4            # -

L       = L_const + L_exhaust # m
A_f     = np.pi * r_f**2      # m^2
A_f_tot = N_f * A_f           # m^2

# Define the boundary conditions (from Tim Dawson email)
P_in   =  127.444e3  # Pa
rho_in =    0.323551 # kg/m^3
U_in   = 1791.05     # m/s
T_in   = 1366.81     # K
M_in   =    2.48942  # -
T0_f   = 300.0       # K

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
Yf_st = gas.Y[gas.species_index('H2')]
Z_st = gas.mixture_fraction(X_f, X_ox)

# Fuel inflow rate

# Get properties at combustor conditions
gas.TPX = T0_f, P_in, X_f
gamma_f = gas.cp / gas.cv
R_f = ct.gas_constant / gas.mean_molecular_weight

def f(M):
    tmp = ((gamma_f + 1) / 2)**((gamma_f + 1) / (2 * (gamma_f - 1)))
    return tmp * M / (1 + (gamma_f - 1) / 2 * M**2)**((gamma_f + 1) / (2 * (gamma_f - 1)))

def calc_M(P0, Pa):
    P0_choked = Pa * ((gamma_f + 1) / 2)**(gamma_f / (gamma_f - 1))
    if P0 < P0_choked:
        # Exit pressure is equal to the ambient pressure
        M = np.sqrt(2 / (gamma_f - 1) * ((P0 / Pa)**((gamma_f - 1) / gamma_f) - 1))
    else:
        # Exit pressure is no longer equal to the ambient pressure
        # We know based on geometry that the Mach number at the orifice is 1
        M = 1.0
    return M

def calc_mdot(P0, T0, A, Pa):
    P0_choked = Pa * ((gamma_f + 1) / 2)**(gamma_f / (gamma_f - 1))
    M = calc_M(P0, Pa)
    gamma_term = gamma_f / ((gamma_f + 1) / 2)**((gamma_f + 1) / (2 * (gamma_f - 1)))
    return gamma_term * P0 * A / np.sqrt(gamma_f * R_f * T0) * f(M)

def P0_from_mdot(mdot, T0, A):
    P0_choked = P_in * ((gamma_f + 1) / 2)**(gamma_f / (gamma_f - 1))
    eqn = lambda P0 : calc_mdot(P0, T0, A, P_in) - mdot
    result = optimize.root_scalar(eqn, x0=P_in)
    P0 = result.root
    M = calc_M(P0, P_in)
    return P0, M

def fuel_props_from_phi(phi_gl):
    if phi_gl == 0.0:
        return np.nan, 0.0, 300.0

    gas.set_equivalence_ratio(phi_gl, X_f, X_ox)
    X_mix = gas.X
    Yf_gl = gas.Y[gas.species_index('H2')]
    mdot_f = (Yf_gl / gas.Y[gas.species_index('O2')]) * mdot_O2
    Z_gl = gas.mixture_fraction(X_f, X_ox)

    # Compute the fuel plenum (stagnation) pressure to achieve mdot_f
    P0_f, M_f = P0_from_mdot(mdot_f, T0_f, A_f_tot)
    T_f = T0_f * (1 + (gamma_f - 1) / 2 * M_f**2)**(-1)
    a_f = np.sqrt(gamma_f * R_f * T_f)
    U_f = M_f * a_f
    rho_f = mdot_f / (U_f * A_f_tot)
    gas.TDX = T_f, rho_f, X_f
    P_f = gas.P
    H_f = gas.enthalpy_mass

    # Compute the estimated temperature of the mixture
    # (Pressure will change but this doesn't affect the temperature)
    gas.HPX = (mdot_a * H_in + mdot_f * H_f) / (mdot_a + mdot_f), P_in, X_mix
    T_mix = gas.T
    delta_T = T_mix - T_in

    return rho_f, U_f, T_f

eps_t = 1.0e-6
t_phi_gl_schedule = np.array(
    [[ 0.0    , 0.0 ],
     [ 0.1*tau, 0.0 ],
     [ 8.0*tau, 0.35],
     [10.0*tau, 0.35],
     [14.0*tau, 0.6 ],
     [16.0*tau, 0.6 ]])
# t_phi_gl_schedule = np.array(
#     [[ 0.0,           0.0 ],
#      [ 0.1*tau-eps_t, 0.0 ],
#      [ 0.1*tau,       0.35],
#      [ 3.0*tau,       0.35],
#      [ 8.0*tau,       0.35],
#      [13.0*tau,       0.6 ],
#      [15.0*tau,       0.6 ]])

t_f = np.zeros(t_phi_gl_schedule.shape[0])
rho_f = np.zeros(t_phi_gl_schedule.shape[0])
U_f = np.zeros(t_phi_gl_schedule.shape[0])
T_f = np.zeros(t_phi_gl_schedule.shape[0])
for i in range(t_phi_gl_schedule.shape[0]):
    t_f[i] = t_phi_gl_schedule[i, 0]
    rho_f[i], U_f[i], T_f[i] = fuel_props_from_phi(t_phi_gl_schedule[i, 1])
# NOTE: Assuming perfect gas & isentropic choked flow, only rho_f changes with phi/mdot
U_f = U_f[-1]
T_f = T_f[-1]

# Define the grid
N_x = 200
x = np.linspace(0, L_const + L_exhaust, N_x)
h = np.zeros_like(x)
h[x < L_const] = h_const
h[x >= L_const] = h_const + (x[x >= L_const] - L_const) * np.tan(theta_exhaust)
A = h * w
lnA = np.log(A)
dlnAdx_data = np.gradient(lnA, x)
dlnAdx_interp = interpolate.interp1d(x, dlnAdx_data, kind='cubic')
dlnAdx = lambda x, t : dlnAdx_interp(x)

# PDF sampling parameters
dx_Z_pdf = 5.0e-3
dZ_pdf = 1.0e-2
n_bins_Z_pdf = int(np.ceil(1.0 / dZ_pdf))

# Initialize the state
gas_init = ct.Solution(mech)
gas_init.TPX = T_in, P_in, "O2:1,N2:3.76"
initState = gas_init, U_in

# Define the boundary conditions
BC_inlet = gas_init.density, U_in, gas_init.P, (0.0, 0.0)
BC_outlet = 'outflow'
BCs = (BC_inlet, BC_outlet)

# Load the FPV table
fpv_table = FPVTable(table_file)

# #################################################################

# Build the injector model
jic = JICModel(gas, "H2",
               x, x_inj, L_const, w, h[0], N_f, 2*r_f,
               t_f, rho_f, U_f, T_f,
               rho_in, U_in, T_in,
               alpha=1e6,
               fpv_table=fpv_table,
               load_Z_3D=True,
               load_Z_avg_var_profiles=True,
               load_chemical_sources=False,
               load_MIB_profile=False)

###################################################################

# i_m = 1
# z_plot = jic.z_inj[1]
# i_z = np.argmin(np.abs(jic.z_3D_data - z_plot))

# fig, ax = plt.subplots()
# c = ax.contourf(jic.x_3D_data*scale,
#                 jic.y_3D_data*scale,
#                 jic.Z_3D_data[i_m,:,:,i_z].T,
#                 levels=np.linspace(0, 0.3, 100))
# y_cl_max = jic.y_cl(L - x_inj)[i_m]
# y_cl_arr = np.linspace(0, y_cl_max, 1000)
# x_cl_arr = jic.x_cl_from_y_cl(y_cl_arr[:,np.newaxis])[:,i_m]
# ax.plot((x_cl_arr + x_inj)*scale, y_cl_arr*scale, 'r')
# ax.set_xlabel(r'$x$ [mm]')
# ax.set_ylabel(r'$y$ [mm]')
# ax.set_xlim((50.0, 100.0))
# cbar = plt.colorbar(c, ticks=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
# cbar.set_label(r'$Z$')
# ax.set_aspect('equal')
# plt.savefig(os.path.join(figdir, "injector_xy.png"), bbox_inches='tight', dpi=300)


# x_plot = x_inj + np.array([0.0e-3,
#                            25.0e-3,
#                            50.0e-3])
# i_plot = np.array([np.argmin(np.abs(x - x_i)) for x_i in x_plot])

# fig, axs = plt.subplots(len(x_plot), 1, sharex=True, figsize=(6, 8))
# for i in range(len(x_plot)):
#     i_x = i_plot[i]
#     c = axs[i].contourf(jic.z_3D_data*scale,
#                         jic.y_3D_data*scale,
#                         jic.Z_3D_data[i_m,i_x,:,:],
#                         levels=np.linspace(0, 0.3, 100))
#     for i_inj in range(N_f):
#         axs[i].scatter(jic.z_inj[i_inj]*scale,
#                        jic.y_cl(jic.x_3D_data[i_x] - x_inj)[i_m]*scale,
#                        color='r', marker='x')
#     axs[i].set_ylabel(r'$y$ [mm]')
#     axs[i].set_aspect('equal')
#     # plt.colorbar(c, ax=axs[i_x])
# axs[-1].set_xlabel(r'$z$ [mm]')
# plt.tight_layout()
# plt.savefig(os.path.join(figdir, "injector_zy.png"), bbox_inches='tight', dpi=300)

# breakpoint()

###################################################################

# # Evaluate the mixture fraction PDFs along the axial direction
# print("Evaluating the mixture fraction PDFs along the axial direction...")
# Z_pdf = []
# Z_pdf_samples = []
# Z_pdf_weights = []
# Z_avg_simple = np.zeros_like(x)
# Z_avg = np.zeros_like(x)
# Z_var = np.zeros_like(x)
# i_last_const = np.where(x <= L_const)[0][-1]

# Z_plot = np.linspace(0, 1, 1000)
# fig, ax = plt.subplots()
# ax.set_xlabel(r'$Z$')
# ax.set_ylabel(r'$p(Z)$')

# for i_x, x_i in enumerate(tqdm(x[:i_last_const])):
#     func = lambda z, y : jic.Z_3D(x_i, y, z)
#     grad_func = lambda z, y : jic.grad_Z_3D(x_i, y, z)
#     ranges = ((-w/2, w/2), (0, h[i_x]))
#     grid_dims = (int(np.ceil((ranges[0][1] - ranges[0][0]) / dx_Z_pdf)),
#                  int(np.ceil((ranges[1][1] - ranges[1][0]) / dx_Z_pdf)))
#     sampler = MonteCarloSampler2D(
#         ranges, func, grad_func, grid_dims,
#         alpha=0.0, num_samples_per_iter=5000, max_iters=10)
#     Z_pdf_i, z_samples, y_samples, Z_pdf_samples_i, Z_pdf_weights_i = sampler.compute_scalar_pdf()
#     Z_pdf.append(Z_pdf_i)
#     Z_pdf_samples.append(Z_pdf_samples_i)
#     Z_pdf_weights.append(Z_pdf_weights_i)

#     # Plot, mapping color to x coordinate
#     ax.plot(Z_plot, Z_pdf_i(Z_plot), c=plt.cm.viridis(i_x / (len(x) - 1)))
#     fig.savefig(os.path.join(figdir, "Z_pdf.png"), bbox_inches='tight', dpi=300)
    
# for i_x in range(i_last_const, len(x)):
#     Z_pdf.append(Z_pdf[i_last_const-1])
#     Z_pdf_samples.append(Z_pdf_samples[i_last_const-1])
#     Z_pdf_weights.append(Z_pdf_weights[i_last_const-1])

#     # Plot, mapping color to x coordinate
#     ax.plot(Z_plot, Z_pdf[i_last_const-1](Z_plot), c=plt.cm.viridis(i_x / (len(x) - 1)))
#     fig.savefig(os.path.join(figdir, "Z_pdf.png"), bbox_inches='tight', dpi=300)

# print("Computing the mean and variance of the PDFs...")
# for i_x in range(len(x)):
#     Z_avg[i_x] = integrate.quad(lambda Z : Z * Z_pdf[i_x](Z), 0, 1)[0]
#     Z_var[i_x] = integrate.quad(lambda Z : (Z - Z_avg[i_x])**2 * Z_pdf[i_x](Z), 0, 1)[0]

# np.save(os.path.join(datadir, "Z_avg.npy"), Z_avg)
# np.save(os.path.join(datadir, "Z_var.npy"), Z_var)
# with open(os.path.join(datadir, "Z_pdf.pkl"), 'wb') as f:
#     pickle.dump((Z_pdf_samples, Z_pdf_weights), f)

###################################################################

# fig, ax = plt.subplots(figsize=(4, 3.2))
# ax.plot(x*scale, jic.Z_avg_profile, 'b')
# ax.axhline(Z_gl, color='b', linestyle='--')
# ax.set_ymargin(0.1)
# ax.set_xlabel(r'$x$ [mm]')
# ax.set_ylabel(r'$\langle Z \rangle$', color='b')
# ax.tick_params(axis='y', labelcolor='b')
# ax1 = ax.twinx()
# ax1.semilogy(x*scale, jic.Z_var_profile, 'r')
# ax1.set_ymargin(0.1)
# ax1.set_ylabel(r"$\langle Z''^2 \rangle$", color='r')
# ax1.tick_params(axis='y', labelcolor='r')
# plt.savefig(os.path.join(figdir, "Z_avg_var.png"), bbox_inches='tight', dpi=300)

###################################################################

# Initialize and run the simulation
ss = stanScram(gas,
               h = h,
               w = w,
               dlnAdx = dlnAdx,
               Tw = 300.0,
               includeBoundaryLayerTerms = True,
               initialization = ("constant", initState, x),
               boundaryConditions = BCs,
               sourceTerms = None,
               injector = jic,
               ox_def = X_ox,
               fuel_def = X_f,
               prog_def = {"H2O" : 1.0},
               cfl = 0.5,
               physics = "FPV",
               fpv_table = fpv_table,
               reacting = True,
               includeDiffusion = False,
               outputEvery = 10,
               plotStateInterval = 10)
ss.addXTDiagram("density", skipSteps=10)
ss.addXTDiagram("velocity", skipSteps=10)
ss.addXTDiagram("pressure", skipSteps=10)
ss.addXTDiagram("temperature", skipSteps=10)
ss.addXTDiagram("mixture fraction", skipSteps=10)
ss.addXTDiagram("progress variable", skipSteps=10)
ss.addXTDiagram("mach", skipSteps=10)
ss.advanceSimulation(t_f[-1])
ss.plotXTDiagrams(figdir=figdir)

import code; code.interact(local=locals())