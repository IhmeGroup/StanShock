from __future__ import annotations

from case_setup import (
    HydrogenInjectionFRC,
    default_frc_physics,
    hyshot_ii_geometry,
    stream_averaged_inflow,
)

from stanshock.components.combustor import Combustor
from stanshock.numerics.boundary_conditions import BCInput
from stanshock.processing.initialize import InitializeConstant
from stanshock.processing.plot import plot_state

# Get case setup
geometry = hyshot_ii_geometry()
physics = default_frc_physics()
inflow_bc = stream_averaged_inflow(physics)
BCs: BCInput = {"left": inflow_bc, "right": "outflow"}

# Time parameters
U_in = inflow_bc.reference_state[1]
assert U_in is not None
L = geometry.regions["domain"][1] - geometry.regions["domain"][0]
tau = L / U_in
t_end = 5 * tau
# t_end = 0.5 * tau
print(f"tau = {tau:.2e} s")
print(f"t_end = {t_end:.2e} s")

# Initialize and run the simulation
ss = Combustor(
    geometry=geometry,
    physics=physics,
    initialization=InitializeConstant(geometry, physics, physics.gas, U_in),
    boundary_conditions=BCs,
    source_terms=HydrogenInjectionFRC(geometry=geometry, physics=physics),
    cfl=0.5,
    reacting=True,
    include_diffusion=False,
    output_every=10,
)
ss.advance_simulation(t_end)

# Plot the results
plot_state(ss, "hyshot_ii_frc.png")
