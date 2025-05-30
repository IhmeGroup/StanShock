from __future__ import annotations

import numpy as np

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array


class RightHandSide:
    idx_domain = np.s_[:]
    idx_source = np.s_[:]

    def source(
        self,
        time: float,
        state_array: Array,
        physics: FluidPhysics,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        """Compute the temporal gradient of the current state of the system."""
        state = physics.conservative_to_primitive(state_array, gamma_star, e0_star)
        state.gamma_star = gamma_star
        state.e0_star = e0_star
        return self.source_from_primitives(time, state, physics)

    def source_from_primitives(
        self, time: float, state: FluidState, physics: FluidPhysics
    ) -> Array:
        assert state.gamma_star is not None
        assert state.e0_star is not None
        state_array = physics.primitive_to_conservative(state)
        return self.source(time, state_array, physics, state.gamma_star, state.e0_star)


class CombinedSource(RightHandSide):
    def __init__(self, sources: list[RightHandSide]) -> None:
        self.sources = sources

    def source(
        self,
        time: float,
        state_array: Array,
        physics: FluidPhysics,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        rhs = np.zeros_like(state_array)

        for source in self.sources:
            rhs[source.idx_domain, source.idx_source] += source.source(
                time=time,
                state_array=state_array,
                physics=physics,
                gamma_star=gamma_star,
                e0_star=e0_star,
            )

        return rhs
