import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

from stanshock.models.inlet_diffuser import InletDiffuser
from stanshock.physics.fluid_base import FluidState
from stanshock.physics.thermotable import ThermoTable
from stanshock.system.geometry import AsymmetricBox,LinearInterpolator  # Or use the initialize_geometry helper function with upper and lower walls

# mech = "data/mechanisms/h2_boivin_9sp_12r_mod.yaml"
mech = "air.yaml"
gas = ct.Solution(mech)
physics = ThermoTable(gas)

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

x_ramp_start = -350.0e-3 #m
x_cowl_start = -55.0e-3 #m
y_ramp_start = -120.0e-3 #m
x_combustor_start = 0.0 #m
x_combustor_end = L_const #m
regions = {"domain": (x_combustor_start, x_combustor_end), "external": (x_ramp_start, x_combustor_start)}

xf = np.linspace(x_ramp_start,x_combustor_end,N_x+1)
ramp_lower_wall = LinearInterpolator(xp=[x_ramp_start,x_combustor_start],fp=[y_ramp_start,0])
combustor_lower_wall = LinearInterpolator(xp=[x_combustor_start,x_combustor_end],fp=[0,0])
ramp_upper_wall = LinearInterpolator(xp=[x_cowl_start,x_combustor_start],fp=[h_const,h_const])
combustor_upper_wall = LinearInterpolator(xp=[x_combustor_start,x_combustor_end],fp=[h_const,h_const])

lower_wall_ramp_y_values = ramp_lower_wall(_time=0,x=xf[xf <= x_combustor_start])
lower_wall_combustor_y_values = combustor_lower_wall(_time=0,x=xf[(xf>=x_combustor_start) & (xf <= x_combustor_end)])

lower_wall_y_values = np.concatenate((lower_wall_ramp_y_values,lower_wall_combustor_y_values))

upper_wall_ramp_y_values = ramp_upper_wall(_time=0,x=xf[(xf>=x_cowl_start) & (xf <= x_combustor_start)])
#fill open with height 1000, check if there is better way to do this
# upper_wall_ramp_y_values = np.concatenate((np.full(len(xf[xf <= x_combustor_start])-len(upper_wall_ramp_y_values),1000.0),upper_wall_ramp_y_values))
upper_wall_combustor_y_values = combustor_upper_wall(_time=0,x=xf[(xf>=x_combustor_start) & (xf <= x_combustor_end)])

# upper_wall_y_values = np.concatenate((upper_wall_ramp_y_values,upper_wall_combustor_y_values))

# import matplotlib.pyplot as plt
# plt.plot(xf,lower_wall_y_values)
# plt.plot(xf,upper_wall_y_values)
# plt.show()

# geometry = AsymmetricBox(xf=xf,regions=regions, lower_wall=(xf,lower_wall_y_values), upper_wall=(xf,upper_wall_y_values)) # Geometry definition goes in here
geometry = AsymmetricBox(xf=xf,regions=regions, lower_wall=(xf,lower_wall_y_values), upper_wall=(xf[(xf>=x_combustor_start)],upper_wall_combustor_y_values)) # Geometry definition goes in here


gas.TPY = 263.6, 2024.0, {"N2": 0.752, "O2": 0.216, "NO": 0.032}
freestream_state = FluidState(
    shape=(1,),
    temperature=np.array([gas.T]),
    density=np.array([gas.density_mass]),
    velocity=np.array([2398.0]),
    composition=gas.Y[None, :],
    sound_speed=343
)
past_M1_state = FluidState(
    shape=(1,),
    temperature=np.array([gas.T]),
    density=np.array([gas.density_mass]),
    velocity=np.array([2398.0]),
    composition=gas.Y[None, :],
    sound_speed=343
)
past_M2_state = FluidState(
    shape=(1,),
    temperature=np.array([gas.T]),
    density=np.array([gas.density_mass]),
    velocity=np.array([2398.0]),
    composition=gas.Y[None, :],
    sound_speed=343
)
past_M3_state = FluidState(
    shape=(1,),
    temperature=np.array([gas.T]),
    density=np.array([gas.density_mass]),
    velocity=np.array([2398.0]),
    composition=gas.Y[None, :],
    sound_speed=343
)

inlet = InletDiffuser(
    angle_of_attack=np.deg2rad(3.6),
    freestream=freestream_state,
    past_M1=past_M1_state,
    past_M2=past_M2_state,
    past_M3=past_M3_state,
    geometry=geometry,
    physics=physics,
)



# flow_deflection_angles = [18 + np.rad2deg(inlet.angle_of_attack) ,18,0]
# inlet.compute_combustor_inlet_properties(n_shocks=3,flow_deflection_angles=flow_deflection_angles)
# print(inlet.past_M3)
# print(inlet.past_M1)

aoa_range = np.linspace(0,10,100)
temperatures = []
pressures = []

for aoa in aoa_range:
    inlet.compute_combustor_inlet_properties(n_shocks=3,flow_deflection_angles=[18+aoa, 18, 0])
    temperatures.append(inlet.past_M3.temperature)
    pressures.append(inlet.past_M3.pressure/1000) #kPa conversion

fig,(ax1,ax2) = plt.subplots(1,2,figsize = (10,4))
ax1.plot(aoa_range,temperatures)
ax1.set_title('Combustor Inflow Temperature vs AoA')
ax1.set_xlabel('AoA (deg)')
ax1.set_ylabel('Combustor Inflow Temperature (K)')

ax2.plot(aoa_range,pressures)
ax2.set_title('Combustor Inflow Pressure vs AoA')
ax2.set_xlabel('AoA (deg)')
ax2.set_ylabel('Combustor Inflow Pressure (kPa)')

plt.tight_layout()
plt.show()


#RANS values
#Freestream: 7.3561
#Region1: 3.66349
#Region2: 2.64796
#Region3: 2.35554 right after shock, 2.61205 further downstream