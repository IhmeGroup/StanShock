from __future__ import annotations

import numpy as np

from stanshock.models.wall_models import (
    HeatFlux,
    SkinFriction,
    WallState,
)
from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Index, Unpack
from stanshock.system.base import PrecomputeStepName, PrecomputeSteps, RightHandSide


class BoundaryLayer(RightHandSide):
    REQUIRED_PRECOMPUTE_STEPS: tuple[PrecomputeStepName, ...] = ("geometry", "physics")

    def __init__(
        self,
        wall_models: tuple[SkinFriction, HeatFlux | None],
        wall_temperature: Array | float | None = None,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        super().__init__(**precompute_steps)
        assert self.physics is not None
        self.wall_temperature = wall_temperature

        if wall_models is None:
            msg = "Must provide at least a SkinFriction model"
            raise ValueError(msg)

        # Provides momentum and energy source terms
        self.skin_friction, self.heat_flux = wall_models
        self.idx_source: Index = np.array([0, 1])
        self.shape_output = (self.shape_output[0], 2)

    def source_implementation(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Boundary layer contribution to RHS."""
        _ = time, state_array_local, face_states, avg_face_states, face_gradients
        assert self.physics is not None
        assert self.geometry is not None
        assert state is not None
        rhs = np.zeros((*state.shape, 2))
        x = self.geometry.xc[self.idx_input]
        characteristic_length = self.geometry.characteristic_length(time, x)
        hydraulic_diameter = self.geometry.hydraulic_diameter(time, x)

        wall = WallState.from_state(
            state, self.physics, Lc=characteristic_length, Tw=self.wall_temperature
        )

        wall.Cf = self.skin_friction(wall)
        tau_wall = wall.Cf * wall.q
        rhs[:, 0] = -4.0 / hydraulic_diameter * tau_wall

        if self.heat_flux is not None:
            q_wall = self.heat_flux(wall)
            rhs[:, 1] = -4.0 / hydraulic_diameter * q_wall

        return np.ravel(rhs)
