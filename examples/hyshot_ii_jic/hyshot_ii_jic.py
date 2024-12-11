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
from scipy import interpolate, integrate
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
x_inj         =  57.5e-3       # m
theta_exhaust = np.deg2rad(12) # rad
r_f           =   0.2e-3       # m
N_f           =   4            # -

L       = L_const + L_exhaust # m
A_f     = np.pi * r_f**2      # m^2
A_f_tot = N_f * A_f           # m^2

# Define the boundary conditions
P_in   =  127.444e3  # Pa
rho_in =    0.323551 # kg/m^3
U_in   = 1791.05     # m/s
T_in   = 1366.81     # K
M_in   =    2.48942  # -
mdot_a = rho_in * U_in * h_const * w

T_f    = 250.0    # K
mdot_f =   4.4e-3 # kg/s
phi    =   0.35   # -
M_f    =   1.0    # -

# Compressible air properties
gas.TPX = T_in, P_in, X_ox
gamma_in = gas.cp / gas.cv
a_in = gas.sound_speed
M_in_comp = U_in / a_in

# State downstream of the bow shock on the injected jet
rho_2 = rho_in * (gamma_in + 1) * M_in_comp**2 / ((gamma_in - 1) * M_in_comp**2 + 2)
U_2 = U_in * rho_in / rho_2
P_2 = P_in * (2 * gamma_in * M_in_comp**2 - (gamma_in - 1)) / (gamma_in + 1)

# Stoichiometry
mdot_O2 = 0.23291 * mdot_a
mdot_N2 = mdot_a - mdot_O2
gas.TPY = 298.15, ct.one_atm, "H2:{0},O2:{1},N2:{2}".format(mdot_f, mdot_O2, mdot_N2)
Yf_gl = gas.Y[gas.species_index('H2')]
phi_gl = gas.equivalence_ratio(X_f, X_ox)
Z_gl = gas.mixture_fraction(X_f, X_ox)
gas.set_equivalence_ratio(1.0, X_f, X_ox)
Yf_st = gas.Y[gas.species_index('H2')]
Z_st = gas.mixture_fraction(X_f, X_ox)

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

# Define the boundary conditions
BC_inlet = gas_init.density, U_in, gas_init.P, gas_init.Y
BC_outlet = 'outflow'
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
rhoYH2_f = rho_f * gas.Y[gas.species_index('H2')]

# Load the FPV table
fpv_table = FPVTable(table_file)

# ##############################################################

# # TEST CASE (Torrez 2011)
# x_inj = 0.358
# w = 0.0381
# h = 0.0254
# N_f = 1
# T_f = 248.0
# p_f = 438.0e3
# U_f = 1200.0
# gas.TPX = T_f, p_f, "H2:1"
# rho_f = gas.density
# r_f = 2.49e-3 / 2

# T_in = 1280
# p_in = 261.0e3
# U_in = 458.0
# gas.TPY = T_in, p_in, "O2:0.251,N2:0.611,H2O:0.138"
# rho_in = gas.density

# #################################################################

# Build the injector model
jic = JICModel(gas, "H2",
               x, x_inj, w, h[0], N_f, 2*r_f,
               rho_f, U_f, T_f,
               rho_in, U_in, T_in,
               alpha=1e6,
               time_delay=1e-3,
               fpv_table=fpv_table,
               load_adjustment_factor=True,
               load_Z_avg_var_profiles=True)

###################################################################

# # ######
# # # Test plot

# x_plot = x_inj + np.linspace(-5.0e-3, 0.05, 100)
# y_plot = np.linspace(0, h[0], 50)
# # x_plot = x_inj + np.linspace(-1.0e-3, 5.0e-3, 100)
# # y_plot = np.linspace(0, 3.0e-3, 50)
# z_plot = jic.z_inj[0]

# X_plot, Y_plot = np.meshgrid(x_plot, y_plot, indexing='ij')
# Z_plot = np.zeros_like(X_plot)
# for i_x in range(len(x_plot)):
#     for i_y in range(len(y_plot)):
#         Z_plot[i_x, i_y] = jic.Z_3D_adjusted(X_plot[i_x, i_y], Y_plot[i_x, i_y], z_plot)

# fig, ax = plt.subplots()
# c = ax.contourf(X_plot*scale, Y_plot*scale, Z_plot, np.linspace(0, 1.0, 101))
# x_plot_ycl = np.linspace(x_inj, x_plot[-1], 1000)
# ax.plot(x_plot_ycl*scale, jic.y_cl(x_plot_ycl - x_inj)*scale, 'r')
# ax.set_xlabel(r'$x$ [mm]')
# ax.set_ylabel(r'$y$ [mm]')
# cbar = plt.colorbar(c)
# cbar.set_label(r'$Z$')
# ax.set_aspect('equal')
# plt.savefig(os.path.join(figdir, "injector_xy.png"), bbox_inches='tight', dpi=300)

# # # breakpoint()

# # ######
# # # Test plot

# x_plot = x_inj + np.array([0.0e-3,
#                            5.0e-3,
#                            10.0e-3,
#                            50.0e-3])
# # x_plot = x_inj + np.linspace(-1.0e-3, 5.0e-3, 40)
# y_plot = np.linspace(0, h[0], 100)
# z_plot = np.linspace(-w/2, w/2, 200)

# Z_plot, Y_plot = np.meshgrid(z_plot, y_plot, indexing='ij')
# Z_avg_plot = np.zeros_like(x_plot)

# fig, axs = plt.subplots(len(x_plot), 1, sharex=True, figsize=(6, 8))

# for i_x, x_i in enumerate(tqdm(x_plot)):
#     ZBilger_plot = np.zeros_like(Z_plot)
#     for i_z in range(len(z_plot)):
#         for i_y in range(len(y_plot)):
#             ZBilger_plot[i_z, i_y] = jic.Z_3D_adjusted(x_i, Y_plot[i_z, i_y], Z_plot[i_z, i_y])
    
#     Z_avg_plot[i_x] = np.mean(ZBilger_plot)
    
#     c = axs[i_x].contourf(Z_plot*scale, Y_plot*scale, ZBilger_plot, levels=np.linspace(0, 0.3, 101))
#     for i_inj in range(N_f):
#         axs[i_x].scatter(jic.z_inj[i_inj]*scale, jic.y_cl(x_i - x_inj)*scale, color='r', marker='x')
#     axs[i_x].set_ylabel(r'$y$ [mm]')
#     axs[i_x].set_aspect('equal')
#     # plt.colorbar(c, ax=axs[i_x])

#     # breakpoint()

# axs[-1].set_xlabel(r'$z$ [mm]')
# plt.tight_layout()
# plt.savefig(os.path.join(figdir, "injector_zy.png"), bbox_inches='tight', dpi=300)

# fig, ax = plt.subplots()
# ax.plot(x_plot * scale, Z_avg_plot)
# ax.axhline(Z_gl, color='r', linestyle='--')
# ax.set_xlabel(r'$x$ [mm]')
# ax.set_ylabel(r'$\langle Z \rangle$')
# plt.savefig(os.path.join(figdir, "injector_z_avg.png"), bbox_inches='tight', dpi=300)

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

fig, ax = plt.subplots(figsize=(4, 3.2))
ax.plot(x*scale, jic.Z_avg_profile, 'b')
ax.axhline(Z_gl, color='b', linestyle='--')
ax.set_ymargin(0.1)
ax.set_xlabel(r'$x$ [mm]')
ax.set_ylabel(r'$\langle Z \rangle$', color='b')
ax.tick_params(axis='y', labelcolor='b')
ax1 = ax.twinx()
ax1.semilogy(x*scale, jic.Z_var_profile, 'r')
ax1.set_ymargin(0.1)
ax1.set_ylabel(r"$\langle Z''^2 \rangle$", color='r')
ax1.tick_params(axis='y', labelcolor='r')
plt.savefig(os.path.join(figdir, "Z_avg_var.png"), bbox_inches='tight', dpi=300)

###################################################################

# Initialize and run the simulation
ss = stanScram(gas,
               dlnAdx = dlnAdx,
               initializeConstant = (initState, x),
               boundaryConditions = BCs,
               sourceTerms = None,
               injector = jic,
               ox_def = X_ox,
               fuel_def = X_f,
               prog_def = {"H2O" : 1.0},
               cfl = 0.5,
               reacting = False,
               includeDiffusion = False,
               outputEvery = 10)
ss.advanceSimulation(t_end)

# Plot the results
def plot_sim(ss):
    rho = ss.r
    u = ss.u
    p = ss.p
    Y = ss.Y
    T = ss.thermoTable.getTemperature(rho, p, Y)

    fig, ax = plt.subplots(6, 1, sharex=True, figsize=(6, 8))
    ax[0].plot(x*scale, rho)
    ax[0].set_ymargin(0.1)
    ax[0].set_ylabel(r'$\rho$ [kg/m$^3$]')
    add_h_plot(ax[0])

    ax[1].plot(x*scale, u)
    ax[1].set_ymargin(0.1)
    ax[1].set_ylabel(r'$u$ [m/s]')
    add_h_plot(ax[1])

    ax[2].plot(x*scale, p)
    ax[2].set_ymargin(0.1)
    ax[2].set_ylabel(r'$p$ [Pa]')
    add_h_plot(ax[2])

    ax[3].plot(x*scale, T)
    ax[3].set_ymargin(0.1)
    ax[3].set_ylabel(r'$T$ [K]')
    add_h_plot(ax[3])

    ax[4].plot(x*scale, Y[:, gas.species_index('H2')])
    ax[4].set_ymargin(0.1)
    ax[4].set_ylabel(r'$Y_{\mathrm{H}_2}$')
    add_h_plot(ax[4])

    ax[5].plot(x*scale, Y[:, gas.species_index('H2O')])
    ax[5].set_ymargin(0.1)
    ax[5].set_ylabel(r'$Y_{\mathrm{H}_2\mathrm{O}}$')
    add_h_plot(ax[5])

    ax[5].set_xlabel('x [mm]')

    plt.tight_layout()
    plt.savefig("hyshot_ii.png", bbox_inches='tight', dpi=300)

plot_sim(ss)

import code; code.interact(local=locals())