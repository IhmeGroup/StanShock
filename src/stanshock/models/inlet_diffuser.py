from __future__ import annotations

from typing import Literal

import numpy as np

from stanshock.numerics.boundary_conditions import SpecifiedFlux
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.geometry import AsymmetricBox, LinearInterpolator
from stanshock.utils.isentropic import compute_ratios_across_oblique_shock


class InletDiffuser(SpecifiedFlux):
    """Model of a supersonic inlet."""

    def __init__(
        self,
        angle_of_attack: float,
        freestream: FluidState,
        geometry: AsymmetricBox,
        physics: FluidPhysics,
        location: Literal["left", "right"] = "left",
    ) -> None:
        self.geometry = geometry
        self.physics = physics

        # Compute the (geometry-fixed) turn angles for the flow
        self.turn_angles = self.compute_turn_angles()
        self._angle_of_attack: float = 0.0
        self.angle_of_attack = angle_of_attack

        # Set the freestream properties, which updates the post-shock properties
        self.fluid_state_regions: list[FluidState] = []
        self.inflow_known: bool = False  # Whether inflow properties have been computed
        self.freestream = freestream
        self.reference_flux: Array = np.zeros((self.physics.n_scalars + 2,))
        self.reference_flux: Array = self.compute_flux(time=0.0)

        super().__init__(self.reference_flux, location)
        if self.location == "left":
            self.idx_boundary_face = 0
        else:
            self.idx_boundary_face = -1

    @property
    def angle_of_attack(self) -> float:
        return self._angle_of_attack

    @angle_of_attack.setter
    def angle_of_attack(self, angle_of_attack: float) -> None:
        if angle_of_attack != self._angle_of_attack:
            self._angle_of_attack = angle_of_attack
            self.inflow_known = False  # Invalidate old inflow values

    def compute_turn_angles(self) -> Array:
        """Enumerate the turn angles of the flow for zero angle of attack.

        Assumes only strong shocks at sharp angles, and that each flow region of
        interest remains parallel to the wall.

        Currently assumes fixed geometry, but could be updated for time-varying upper
        and lower walls.
        """
        # Extract the inlet geometry
        assert isinstance(self.geometry.lower_wall, LinearInterpolator)
        x_lower = self.geometry.lower_wall.xp
        y_lower = self.geometry.lower_wall.fp
        theta_lower = np.arctan2(np.diff(y_lower), np.diff(x_lower))

        assert isinstance(self.geometry.upper_wall, LinearInterpolator)
        x_upper = self.geometry.upper_wall.xp
        y_upper = self.geometry.upper_wall.fp
        theta_upper = np.arctan2(np.diff(y_upper), np.diff(x_upper))

        # Sort angles in x-direction and locate physical turns
        x = np.concatenate((x_lower[:-1], x_upper[:-1]))
        theta = np.concatenate((theta_lower, theta_upper))
        idx_sort = np.argsort(x)
        x = x[idx_sort]
        theta = theta[idx_sort]

        x_inlet = self.geometry.xf[0]
        turns: list[float] = []
        theta_current = 0.0
        for i in range(len(x)):
            if x[i] >= x_inlet:
                break

            if theta[i] != theta_current:
                turns += [theta[i]]
                theta_current = theta[i]

        return np.array(turns)

    @property
    def freestream(self) -> FluidState:
        return self._freestream

    @freestream.setter
    def freestream(self, freestream: FluidState) -> None:
        self._freestream = freestream
        self.inflow_known = False  # Invalidate old inflow values

    def compute_flux(self, time: float) -> Array:
        _ = time
        if self.inflow_known:
            # Reference flux doesn't need to be updated
            return self.reference_flux

        inflow_state = self.compute_combustor_inlet_properties()
        rho = self.physics.get_density(inflow_state)
        u = self.physics.get_velocity(inflow_state)
        p = self.physics.get_pressure(inflow_state)
        e_int = self.physics.get_internal_energy(inflow_state)
        Y = inflow_state.composition
        assert Y is not None

        # Momentum flux
        self.reference_flux[0] = rho * u**2 + p
        # Energy flux
        self.reference_flux[1] = u * (rho * (e_int + 0.5 * u**2) + p)
        # Species fluxes
        self.reference_flux[2:] = rho * u * Y

        # Set flag so the state and flux don't need to be recalculated
        self.inflow_known = True

        return self.reference_flux

    def update(self, time: float, target: Array) -> Array:
        target[self.idx_boundary_face] = self.compute_flux(time)

        return target

    def compute_combustor_inlet_properties(self) -> FluidState:
        n_shocks = len(self.turn_angles)

        # Start with freestream properties
        freestream = self.freestream
        self.fluid_state_regions = [freestream]
        rho = self.physics.get_density(freestream)
        T = self.physics.get_temperature(freestream)
        p = self.physics.get_pressure(freestream)
        gamma = float(self.physics.get_gamma(freestream)[0])
        u = self.physics.get_velocity(freestream)
        mach = float((u / self.physics.get_sound_speed(freestream))[0])
        # print(f"Freestream Mach: {mach[0]}, Angle of Attack: {np.rad2deg(self.angle_of_attack)}")

        # Freeze composition:
        composition = freestream.composition
        assert composition is not None

        # March across each shock
        flow_angle = -self.angle_of_attack
        for i in range(n_shocks):
            turn_angle = np.abs(self.turn_angles[i] - flow_angle)
            flow_angle = self.turn_angles[i]

            mach, density_ratio, pressure_ratio, temperature_ratio = (
                compute_ratios_across_oblique_shock(
                    mach=mach, gamma=gamma, theta=turn_angle
                )
            )
            # print(f"Mach_{i + 1}: {mach[0]}, Turn angle: {np.rad2deg(turn_angle)}")
            T = T * temperature_ratio
            p = p * pressure_ratio
            rho = rho * density_ratio
            new_state = FluidState(
                shape=(1,),
                temperature=T,
                pressure=p,
                density=rho,
                composition=composition,
            )
            new_state.velocity = mach * self.physics.get_sound_speed(new_state)
            gamma = float(self.physics.get_gamma(new_state)[0])

            self.fluid_state_regions.append(new_state)

        return self.fluid_state_regions[-1]
