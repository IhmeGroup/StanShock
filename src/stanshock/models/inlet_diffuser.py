from __future__ import annotations

from typing import Literal

import numpy as np

from stanshock.numerics.boundary_conditions import SpecifiedFlux
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.geometry import AsymmetricBox
from stanshock.utils.isentropic import compute_ratios_across_oblique_shock
class InletDiffuser(SpecifiedFlux):
    """Model of a supersonic inlet."""

    def __init__(
        self,
        angle_of_attack: float,
        freestream: FluidState,
        past_M1: FluidState,
        past_M2: FluidState,
        past_M3: FluidState,
        geometry: AsymmetricBox,
        physics: FluidPhysics,
        location: Literal["left", "right"] = "left",
    ) -> None:
        self.angle_of_attack = angle_of_attack
        self.freestream = freestream
        self.past_M1 = past_M1
        self.past_M2 = past_M2
        self.past_M3 = past_M3
        self.fluid_state_regions = [self.freestream,self.past_M1,self.past_M2,self.past_M3]
        self.geometry = geometry
        self.physics = physics

        self.reference_flux: Array = np.zeros((self.physics.n_scalars + 2,))
        self.reference_flux = self.compute_flux(time=0.0)

        super().__init__(self.reference_flux, location)
        if self.location == "left":
            self.idx_boundary_face = 0
        else:
            self.idx_boundary_face = -1

    def compute_flux(self, time: float) -> Array:
        _ = time
        rho = self.freestream.density[0]
        u = self.freestream.velocity[0]
        Y = self.freestream.composition[0]
        assert rho is not None
        assert u is not None
        assert Y is not None
        p = self.physics.get_pressure(self.freestream)[0]
        e_int = self.physics.get_internal_energy(self.freestream)[0]

        # Momentum flux
        self.reference_flux[0] = rho * u**2 + p
        # Energy flux
        self.reference_flux[1] = u * (rho * (e_int + 0.5 * u**2) + p)
        # Species fluxes
        self.reference_flux[2:] = rho * u * Y

        return self.reference_flux

    def update(self, time: float, target: Array) -> Array:
        target[self.idx_boundary_face] = self.compute_flux(time)

        return target

    def compute_combustor_inlet_properties(self,n_shocks,flow_deflection_angles):
        for i in range(n_shocks):
            if i==0:
                mach = self.fluid_state_regions[i].velocity / self.physics.get_sound_speed(self.freestream)
                print(f'Mach_Freestream: {mach}')
            rho = self.fluid_state_regions[i].density[0]
            pressure = self.fluid_state_regions[i].pressure[0]
            temp = self.fluid_state_regions[i].temperature[0]
            u = self.fluid_state_regions[i].velocity[0]
            Y = self.fluid_state_regions[i].composition[0]
            gamma = self.physics.get_gamma(self.fluid_state_regions[i])
            new_mach, density_ratio,pressure_ratio,temperature_ratio = compute_ratios_across_oblique_shock(mach=mach,gamma=gamma,theta=flow_deflection_angles[i])
            mach = new_mach
            print(f'Mach_{i+1}: {mach}')
            temp = temp*temperature_ratio
            pressure = pressure*pressure_ratio
            rho = rho*density_ratio
            self.fluid_state_regions[i+1].temperature = temp
            self.fluid_state_regions[i+1].pressure = pressure
            self.fluid_state_regions[i+1].density = rho
            self.fluid_state_regions[i+1].velocity = mach*self.physics.get_sound_speed(self.fluid_state_regions[i])
            
            combustor_inlet_properties = 1 #dummy placeholder
        return combustor_inlet_properties
    

