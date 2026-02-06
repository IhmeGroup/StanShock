from __future__ import annotations

from typing import Literal

import numpy as np

from stanshock.numerics.boundary_conditions import SpecifiedFlux
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.geometry import AsymmetricBox


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
        self.angle_of_attack = angle_of_attack
        self.freestream = freestream
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
        rho = self.freestream.density
        u = self.freestream.velocity
        Y = self.freestream.composition
        assert rho is not None
        assert u is not None
        assert Y is not None
        p = self.physics.get_pressure(self.freestream)
        e_int = self.physics.get_internal_energy(self.freestream)

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
