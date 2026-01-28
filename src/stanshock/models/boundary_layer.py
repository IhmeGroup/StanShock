from __future__ import annotations

from typing import cast

import numpy as np
from scipy.optimize import root

from stanshock.physics.fluid_base import FluidState
from stanshock.models.wall_models import WallModel, WallState, get_wall_state, ShearStress_CompressibleReacting, HeatFlux_Compressible
from stanshock.system.backend import Array, Index, Unpack
from stanshock.system.base import PrecomputeStepName, PrecomputeSteps, RightHandSide


class BoundaryLayer(RightHandSide):
    REQUIRED_PRECOMPUTE_STEPS: tuple[PrecomputeStepName, ...] = ("geometry", "physics")

    def __init__(
        self,
        wall_temperature: Array | float | None = None,
        wall_model: WallModel | None = None,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        super().__init__(**precompute_steps)
        self.wall_temperature = wall_temperature

        if wall_model is None:
            self.ShearStressModel = ShearStress_CompressibleReacting
            self.HeatFluxModel = HeatFlux_Compressible
        # else:
        #     self.skin_friction_coefficient = skin_friction_coefficient

        # Provides momentum and energy source terms
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
        assert state is not None
        assert state.density is not None
        assert state.velocity is not None
        rhs = np.zeros((*state.shape, 2))
        x = self.geometry.xc[self.idx_input]
        characteristic_length = self.geometry.characteristic_length(time, x)
        hydraulic_diameter = self.geometry.hydraulic_diameter(time, x)

        # Compute gas properties
        T = state.temperature = self.physics.get_temperature(state)


        wall = get_wall_state(
            rho=state.density,
            U=state.velocity,
            mu=self.physics.get_mu(state),
            a=self.physics.get_sound_speed(state),
            cp=self.physics.get_cp(state),
            k=self.physics.get_thermal_conductivity(state),
            gamma=self.physics.get_gamma(state),
            Dh=hydraulic_diameter,
            T=T,
            wall_temperature=self.wall_temperature,
        )

        tau_wall = ShearStress_CompressibleReacting(wall)
        wall.Cf = tau_wall / wall.q
        q_wall = HeatFlux_Compressible(wall)


        rhs[:, 0] = -4.0 / hydraulic_diameter * tau_wall
        rhs[:, 1] = -4.0 / hydraulic_diameter * q_wall

        return np.ravel(rhs)
