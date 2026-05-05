from __future__ import annotations

from typing import Unpack

import cantera as ct
import numpy as np

from stanshock.components.combustor import Combustor
from stanshock.numerics.boundary_conditions import BCInput, SpecifiedFace
from stanshock.physics.fluid_base import FluidState
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.initialize import InitializeConstant
from stanshock.processing.plot import plot_state
from stanshock.system.backend import Array
from stanshock.system.base import PrecomputeSteps, RightHandSide
from stanshock.system.geometry import Box

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
xf = np.linspace(0, L_const + L_exhaust, N_x + 1)
h = np.zeros_like(xf)
h[xf < L_const] = h_const
h[xf >= L_const] = h_const + (xf[xf >= L_const] - L_const) * np.tan(theta_exhaust)
geometry = Box(xf=xf, h=h, w=w)

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
BC_inlet = SpecifiedFace(
    reference_state=(gas_init.density, U_in, gas_init.P, gas_init.Y)
)
BC_outlet = "outflow"
BCs: BCInput = {"left": [BC_inlet], "right": [BC_outlet]}


# Define the fuel inflow
class HydrogenInjection(RightHandSide):
    def __init__(self, **precompute_steps: Unpack[PrecomputeSteps]) -> None:
        super().__init__(**precompute_steps)
        assert self.geometry is not None
        assert self.physics is not None
        gas = self.physics.gas
        n_variables = self.physics.n_scalars + 2

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

        # rhoE_f = P_f / (gamma_f - 1) + 0.5 * rho_f * U_f**2
        rhoE_f = rho_f * (gas.int_energy_mass + 0.5 * U_f**2)
        rhoYH2_f = rho_f * gas.Y[gas.species_index("H2")]
        A_f = A_f_tot

        # Define the source terms
        L_src = 30.0e-3
        scale_factor = 1.0  # 3.960715337483353

        # Get geometry information
        xf = self.geometry.xf
        idx_faces = np.where(np.logical_and(xf >= x_inj, xf < x_inj + L_src))[0]
        self.idx_input = idx_faces + self.geometry.n_ghost_layers
        self.xc = self.geometry.xc[self.idx_input]
        n_cells = len(self.xc)

        idx_faces = np.concatenate((idx_faces, [idx_faces[-1] + 1]))
        self.xf = self.geometry.xf[idx_faces]

        self.shape_input = (n_cells, n_variables)
        self.shape_output = (n_cells, 3)
        self.idx_source = np.array([0, 1, 2 + gas.species_index("H2")])

        self.rhs = np.zeros(self.shape_output)
        self.rhs[:, 0] = rho_f * U_f
        self.rhs[:, 1] = rhoE_f
        self.rhs[:, 2] = rhoYH2_f

        dx = self.geometry.dx
        if isinstance(dx, np.ndarray):
            dx = dx[self.idx_input, None]
        self.rhs *= U_f * A_f * (dx / L_src) * scale_factor

        # If the volume is constant with time, go ahead and precompute it:
        self.compute_volume: bool = True
        if self.geometry.dlnA_dt is None:
            self.compute_volume = False
            vol = self.geometry.volume(0.0, self.xf)[:, None]
            self.rhs = np.ravel(self.rhs / vol)

    def source_implementation(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        assert self.geometry is not None
        _ = time, state_array_local, state, face_states, avg_face_states, face_gradients
        if self.compute_volume:
            vol = self.geometry.volume(time, self.xf)[:, None]
            return np.ravel(self.rhs / vol)
        return self.rhs


# Initialize and run the simulation
physics = ThermoTable(gas)
ss = Combustor(
    geometry=geometry,
    initialization=InitializeConstant(geometry, physics, gas_init, U_in),
    boundary_conditions=BCs,
    source_terms=HydrogenInjection(geometry=geometry, physics=physics),
    cfl=0.5,
    reacting=True,
    include_diffusion=False,
    output_every=10,
    physics=physics,
)
ss.advance_simulation(t_end)


# Plot the results
plot_state(ss, "hyshot_ii.png")
