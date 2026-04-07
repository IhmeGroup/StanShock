from __future__ import annotations

from pathlib import Path
from typing import Literal, Unpack

import cantera as ct
import numpy as np
from injector_models import fuel_props_from_phi

from stanshock.components.combustor import Combustor
from stanshock.models.inlet_diffuser import InletDiffuser
from stanshock.models.jicf import JICModel
from stanshock.models.wall_models import (
    CompressibleHeatFlux,
    CompressibleReactingSkinFriction,
)
from stanshock.numerics.boundary_conditions import BCInput, BCType, SpecifiedFace

# from stanshock.physics.thermotable import ThermoTable
from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.physics.flamelet import FPVTable
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.processing.initialize import InitializeConstant
from stanshock.system.backend import Array
from stanshock.system.base import PrecomputeSteps, RightHandSide
from stanshock.system.geometry import AsymmetricBox, Geometry

data_dir = Path(__file__).resolve().parent / "../../data"


# Specs from the HyShot II scramjet
def hyshot_ii_geometry(n_x: int = 200) -> AsymmetricBox:
    h_const = 9.8e-3  # m
    w = 75.0e-3  # m
    L_const = 300.0e-3  # m
    L_exhaust = 100.0e-3  # m
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

    x_sim_start = max(x_combustor_start, 5.0e-3)  # m
    x_combustor_end = L_const  # m
    x_nozzle_end = L
    regions = {
        "domain": (x_sim_start, x_nozzle_end),
        "combustor": (x_sim_start, x_combustor_end),
        "nozzle": (x_combustor_end, x_nozzle_end),
        "external": (x_ramp_start, x_sim_start),
    }

    # Define the 1D simulation grid
    xf = np.linspace(x_sim_start, x_nozzle_end, n_x + 1)

    xy_lower_wall = (
        np.array([x_ramp_start, x_combustor_start, L]),
        np.array([y_ramp_start, 0.0, 0.0]),
    )
    xy_upper_wall = (
        np.array([x_cowl_start, x_combustor_start, x_combustor_end, L]),
        np.array([h_const, h_const, h_const, h_nozzle]),
    )

    # geometry = AsymmetricBox(xf=xf,regions=regions, lower_wall=(xf,lower_wall_y_values), upper_wall=(xf,upper_wall_y_values)) # Geometry definition goes in here
    return AsymmetricBox(
        xf=xf,
        w=w,
        regions=regions,
        lower_wall=xy_lower_wall,
        upper_wall=xy_upper_wall,
        n_ghost_layers=3,
    )


# Define the default fluid physics
def default_frc_physics(
    mech: str | Path = data_dir / "mechanisms/h2_boivin_9sp_12r_mod.yaml",
) -> FluidPhysics:
    gas = ct.Solution(mech)
    # return ThermoTable(gas)
    return CanteraInterface(gas)


def default_fpv_physics(
    mech: str | Path = data_dir / "mechanisms/h2_boivin_9sp_12r_mod.yaml",
    table_file: str
    | Path = "./h2_table/flamelet_results/H2_O2N2_p01_3_tf0300_to1367_200x2x200.h5",
) -> FPVTable:
    gas = ct.Solution(mech)
    return FPVTable(
        table_file,
        gas,
        ox_def={"O2": 0.21, "N2": 0.79},
        fuel_def={"H2": 1.0},
        prog_def={"H2O": 1.0},
        p_correction=False,
        T_correction=False,
    )


# Define the inflow boundary condition
def stream_averaged_inflow(physics: FluidPhysics) -> SpecifiedFace:
    """Inflow conditions obtained via stream-averaging RANS results."""
    P_in = 127.444e3  # Pa
    # rho_in = 0.323551  # kg/m^3
    U_in = 1791.05  # m/s
    T_in = 1366.81  # K
    # M_in = 2.48942  # -

    # Initialize the state
    gas_init = physics.gas

    if physics.is_flamelet:
        gas_init.TPX = T_in, P_in, physics.ox_def
        composition = (1.0, 0.0, 0.0)
    else:
        gas_init.TPX = T_in, P_in, {"O2": 1, "N2": 3.76}
        composition = gas_init.Y

    return SpecifiedFace(
        reference_state=(gas_init.density, U_in, gas_init.P, composition)
    )


def inlet_diffuser_inflow(
    geometry: AsymmetricBox, physics: FluidPhysics, theta: float = 3.6
) -> InletDiffuser:
    gas = physics.gas
    sol = ct.SolutionArray(gas, (1,))
    sol.TPX = 263.6, 2024.0, physics.ox_def
    # sol.TPY = 263.6, 2024.0, {"N2": 0.752, "O2": 0.216, "NO": 0.032}

    freestream_state = FluidState(
        shape=sol.shape,
        temperature=sol.T,
        density=sol.density_mass,
        velocity=np.array([2398.0]),
        composition=physics.get_composition_from_mass_fractions(sol.Y),
        sound_speed=sol.sound_speed,
    )

    return InletDiffuser(
        angle_of_attack=np.deg2rad(theta),
        freestream=freestream_state,
        geometry=geometry,
        physics=physics,
    )


def get_inflow_conditions(
    bc: SpecifiedFace | InletDiffuser, physics: FluidPhysics
) -> tuple[ct.Solution, float]:
    """Get the inflow conditions from the boundary condition"""
    gas = physics.gas

    if isinstance(bc, InletDiffuser):
        state = bc.fluid_state_regions[-1]
        u = physics.get_velocity(state)[0]
        T = physics.get_temperature(state)[0]
        rho = physics.get_density(state)[0]
        Y = physics.get_mass_fractions(state)
        gas.TDY = T, rho, Y
    else:
        rho, u, p, _ = bc.reference_state
        assert rho is not None
        assert u is not None
        assert p is not None
        gas.DPY = rho, p, physics.ox_def

    return gas, u


# Define the fuel inflow source terms
class HydrogenInjectionFRC(RightHandSide):
    def __init__(self, **precompute_steps: Unpack[PrecomputeSteps]) -> None:
        super().__init__(**precompute_steps)
        assert self.geometry is not None
        assert self.physics is not None
        gas = self.physics.gas
        n_variables = self.physics.n_scalars + 2

        # Injector area
        x_inj = 57.5e-3  # m
        r_f = 0.2e-3  # m
        N_f = 4  # -
        A_f = np.pi * r_f**2  # m^2
        A_f_tot = N_f * A_f  # m^2

        # Fuel properties
        comp_f = {"H2": 1.0}
        T_f = 250.0  # K
        mdot_f = 4.4e-3  # kg/s
        # phi = 0.35  # -
        M_f = 1.0  # -
        gas.TPX = T_f, 101325.0, comp_f
        gamma_f = gas.cp / gas.cv
        R_f = ct.gas_constant / gas.mean_molecular_weight
        a_f = np.sqrt(gamma_f * R_f * T_f)
        U_f = M_f * a_f
        rho_f = mdot_f / (U_f * A_f_tot)
        gas.TDX = T_f, rho_f, comp_f

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


def get_injectors_fpv(
    geometry: Geometry,
    physics: FPVTable,
    gas_in: ct.Solution,
    U_in: float,
    mdot: Literal["constant", "schedule"] = "constant",
    fpv_dir: Path = Path("./data"),
) -> JICModel:
    # Freeze after constant cross section region of the combustor
    L_const = 300.0e-3  # m

    # Injector geometry
    x_inj = 58.0e-3  # m
    r_f = 1.0e-3  # m
    N_f = 4  # -

    A_f = np.pi * r_f**2  # m^2
    A_f_tot = N_f * A_f  # m^2

    T0_f = 300.0

    # Air flow properties
    T_in = gas_in.T
    P_in = gas_in.P
    rho_in = gas_in.density_mass
    # gamma_in = gas_in.cp / gas_in.cv
    # H_in = gas_init.enthalpy_mass
    # a_in = gas_in.sound_speed
    # M_in_comp = U_in / a_in

    # Stoichiometry
    A_in = float(geometry.area(0.0, geometry.xf[0]))
    mdot_ox = rho_in * U_in * A_in

    # # State downstream of the bow shock on the injected jet
    # rho_2 = rho_in * (gamma_in + 1) * M_in_comp**2 / ((gamma_in - 1) * M_in_comp**2 + 2)
    # U_2 = U_in * rho_in / rho_2
    # P_2 = P_in * (2 * gamma_in * M_in_comp**2 - (gamma_in - 1)) / (gamma_in + 1)

    # Time parameters
    L = float(geometry.xf[-1] - geometry.xf[0])
    tau = L / U_in
    print(f"tau = {tau:.2e} s")

    # Mass flow rate ramp:
    if mdot == "constant":
        t_phi_gl_schedule = np.array(
            [[0.0, 0.45], [1e3, 0.45 + 1e-15], [2e3, 0.45 + 2e-15]]
        )
    else:
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

    t_f = np.zeros(t_phi_gl_schedule.shape[0])
    rho_f = np.zeros(t_phi_gl_schedule.shape[0])
    U_f = np.zeros(t_phi_gl_schedule.shape[0])
    T_f = np.zeros(t_phi_gl_schedule.shape[0])
    for i in range(t_phi_gl_schedule.shape[0]):
        t_f[i] = t_phi_gl_schedule[i, 0]
        rho_f[i], U_f[i], T_f[i] = fuel_props_from_phi(
            physics, t_phi_gl_schedule[i, 1], mdot_ox, T0_f, P_in, A_f_tot
        )
    # # NOTE: Assuming perfect gas & isentropic choked flow, only rho_f changes with phi/mdot
    # U_f = U_f[-1]
    # T_f = T_f[-1]

    return JICModel(
        x_inj=x_inj,
        x_noz=L_const,
        n_inj=N_f,
        d_inj=2 * r_f,
        t_inj=t_f,
        phi_inj=t_phi_gl_schedule[:, 1],
        rho_inj=rho_f,
        u_inj=U_f,
        T_inj=T_f,
        rho=rho_in,
        u=U_in,
        T=T_in,
        alpha=1e6,
        load_Z_3D=(fpv_dir / "Z_3D.npy").exists(),
        load_Z_avg_var_profiles=(fpv_dir / "Z_var_profile.npy").exists(),
        load_chemical_sources=(fpv_dir / "omega_C_int.npy").exists(),
        load_MIB_profile=(fpv_dir / "C_profile_MIB.npy").exists(),
        geometry=geometry,
        physics=physics,
    )


class Hyshot2Interface:
    """Wrapper around HyShot-II scramjet to facilitate calling from 6-DOF trajectory simulations."""

    def __init__(
        self,
        chemistry: Literal["FRC", "FPV"] = "FPV",
        inflow: Literal["constant", "diffuser"] = "diffuser",
        mdot: Literal["constant", "schedule"] = "constant",
    ) -> None:
        # Get case setup
        geometry = hyshot_ii_geometry(200)

        source = None
        jic = None
        if chemistry == "FRC":
            physics = default_frc_physics()
            source = HydrogenInjectionFRC(geometry=geometry, physics=physics)
        else:
            physics = default_fpv_physics()

        if inflow == "constant":
            self.inflow_bc: BCType = stream_averaged_inflow(physics)
        else:
            self.inflow_bc = inlet_diffuser_inflow(geometry, physics)
        BCs: BCInput = {"left": self.inflow_bc, "right": "outflow"}

        # Set initial conditions
        gas_init, u_init = get_inflow_conditions(self.inflow_bc, physics)
        initialization = InitializeConstant(geometry, physics, gas_init, u_init)

        if chemistry == "FPV":
            # Initialize the injectors
            jic = get_injectors_fpv(geometry, physics, gas_init, u_init, mdot)

        # Initialize the simulation
        self.case = Combustor(
            geometry=geometry,
            physics=physics,
            initialization=initialization,
            boundary_conditions=BCs,
            wall_temperature=300.0,
            wall_models=(CompressibleReactingSkinFriction(), CompressibleHeatFlux()),
            source_terms=source,
            injector=jic,
            cfl=0.5,
            reacting=True,
            include_diffusion=False,
            verbose=False,
            use_double_flux=False,
        )

    def __call__(
        self,
        dt: float,
        p_ref: float,
        T_ref: float,
        mach: float,
        angle_of_attack: float | None = None,
    ) -> tuple[float, float]:
        if isinstance(self.inflow_bc, InletDiffuser):
            if angle_of_attack is not None:
                # Update angle of attack
                self.inflow_bc.angle_of_attack = angle_of_attack

            # Update freestream conditions
            physics = self.case.physics
            composition = self.inflow_bc.freestream.composition
            freestream = physics.set_state(
                FluidState(
                    shape=(1,),
                    temperature=np.array([T_ref]),
                    pressure=np.array([p_ref]),
                    composition=composition,
                )
            )
            freestream.velocity = mach * physics.get_sound_speed(freestream)
            self.inflow_bc.freestream = freestream

        # Integrate forward in time by dt
        self.case.advance_simulation(self.case.t + dt)

        # Compute and return the thrust and fuel consumption rate
        thrust = self.get_thrust()
        mdot = 4.4e-3  # kg/s

        return thrust, mdot

    def get_thrust(self) -> float:
        """Compute the thrust:

        (mdot * V)_e - (mdot * V)_i + (p_e - p_i)*A_e
        """
        geometry = self.case.geometry
        physics = self.case.physics

        x_i = geometry.xf[0]
        A_i = self.case.geometry.area(self.case.t, x_i)

        x_e = geometry.xf[-1]
        A_e = self.case.geometry.area(self.case.t, x_e)

        if isinstance(self.inflow_bc, InletDiffuser):
            p_inf = physics.get_pressure(self.inflow_bc.freestream)
            momentum_flux_i = self.inflow_bc.reference_flux[0]
        else:
            p_inf = 2024.0
            in_state = self.case.state[0]
            rho_i = physics.get_density(in_state)
            u_i = physics.get_velocity(in_state)
            momentum_flux_i = rho_i * u_i**2

        out_state = self.case.state[-1]
        rho_e = physics.get_density(out_state)
        u_e = physics.get_velocity(out_state)
        p_e = physics.get_pressure(out_state)
        momentum_flux_e = rho_e * u_e**2

        return momentum_flux_e * A_e - momentum_flux_i * A_i + (p_e - p_inf) * A_e


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test the use of Hyshot2Interface
    sim = Hyshot2Interface()

    physics = sim.case.physics
    freestream = sim.inflow_bc.freestream
    p_ref = physics.get_pressure(freestream)[0]
    T_ref = physics.get_temperature(freestream)[0]
    u_ref = physics.get_velocity(freestream)[0]
    mach_ref = u_ref / physics.get_sound_speed(freestream)[0]

    def angle_of_attack(t: float) -> float:
        return 2.0 * np.sin(100.0 * t)

    # Output data every 0.01 ms for 3 ms
    dt = 1e-5
    n = 300

    time = np.zeros((n,))
    aoa = np.zeros((n,))
    thrust = np.zeros((n,))

    for i in range(n):
        time[i] = sim.case.t * 1e3
        print(f"Sim Time = {time[i]:.3f} ms")
        aoa[i] = angle_of_attack(sim.case.t + 0.5 * dt)
        thrust[i], _ = sim(dt, p_ref, T_ref, mach_ref, aoa[i])

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    color = "tab:blue"
    ax.plot(time, thrust, color=color)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(r"Thrust [$N$]")
    ax.tick_params(axis="y", labelcolor=color)

    color = "tab:red"
    ax2 = ax.twinx()
    ax2.plot(time, aoa, color=color)
    ax2.set_ylabel(r"Angle of Attack [$\degree$]")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    fig.savefig("thrust_vs_time.png")
    plt.close()
