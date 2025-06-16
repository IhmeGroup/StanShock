from __future__ import annotations

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array


class RightHandSide:
    def source(
        self, time: float, state_array: Array, physics: FluidPhysics, gamma_star: Array
    ) -> Array:
        """Compute the temporal gradient of the current state of the system."""
        state = physics.conservative_to_primitive(state_array, gamma_star)
        return self.source_from_primitives(time, state, physics)

    def source_from_primitives(
        self, time: float, state: FluidState, physics: FluidPhysics
    ) -> Array:
        assert state.gamma is not None
        state_array = physics.primitive_to_conservative(state)
        return self.source(time, state_array, physics, state.gamma)
