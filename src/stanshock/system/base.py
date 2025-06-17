from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Literal, TypedDict

import numpy as np

from stanshock.numerics.boundary_conditions import BoundaryConditions
from stanshock.numerics.face_average import FaceAverage
from stanshock.numerics.face_extrapolation import FaceExtrapolator
from stanshock.numerics.gradient import Gradient
from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array, Index, NotRequired, TypeAlias, Unpack
from stanshock.system.geometry import Geometry

# Define the optional precomputation steps to be called before a given source term
PrecomputeStepName: TypeAlias = Literal[
    "geometry",
    "physics",
    "boundary_conditions",
    "face_extrapolator",
    "face_average",
    "gradient",
]


class PrecomputeSteps(TypedDict):
    geometry: Geometry
    physics: NotRequired[FluidPhysics]
    boundary_conditions: NotRequired[BoundaryConditions]
    face_extrapolator: NotRequired[FaceExtrapolator]
    face_average: NotRequired[FaceAverage]
    gradient: NotRequired[Gradient]


class RightHandSide:
    REQUIRED_PRECOMPUTE_STEPS: ClassVar[tuple[PrecomputeStepName, ...]] = ("geometry",)

    def __init__(
        self,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        # Any precompute steps not provided will default to None
        self.geometry = precompute_steps.get("geometry")
        self.physics = precompute_steps.get("physics")
        self.boundary_conditions = precompute_steps.get("boundary_conditions")
        self.face_extrapolator = precompute_steps.get("face_extrapolator")
        self.face_average = precompute_steps.get("face_average")
        self.gradient = precompute_steps.get("gradient")

        self.idx_domain: Index = self.geometry.idx_cells
        self.idx_source: Index = np.s_[:]

        # Ensure required routines were provided
        for step in self.REQUIRED_PRECOMPUTE_STEPS:
            if getattr(self, step) is None:
                msg: str = f"Must provide {step} for {self.__class__.__name__}."
                raise ValueError(msg)

    def precompute(
        self,
        time: float,
        state_array: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> tuple[
        Array,
        FluidState | None,
        FluidState | None,
        FluidState | None,
        FluidState | None,
    ]:
        """Perform all calculations which must occur prior to source term evaluation."""
        if self.boundary_conditions is not None:
            state_array = self.boundary_conditions.update_ghost_layers(
                time, state_array
            )

        state: FluidState | None = None
        if self.physics is not None:
            state = self.physics.conservative_to_primitive(
                state_array, gamma_star, e0_star
            )
            if self.boundary_conditions is not None:
                state = self.boundary_conditions.update_ghost_states(time, state)

        face_states: FluidState | None = None
        avg_face_states: FluidState | None = None
        if self.face_extrapolator is not None:
            assert state is not None
            face_states = self.face_extrapolator(state)
            if self.boundary_conditions is not None:
                face_states = self.boundary_conditions.update_face_states(
                    time, face_states
                )

            if self.face_average is not None:
                avg_face_states = self.face_average(face_states)

        face_gradients: FluidState | None = None
        if self.gradient is not None:
            assert self.physics is not None
            assert state is not None
            state.temperature = self.physics.get_temperature(state)
            face_gradients = self.gradient.face_gradients(state, self.geometry)

        return state_array, state, face_states, avg_face_states, face_gradients

    def source(
        self,
        time: float,
        state_array: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        """Compute the temporal gradient of the current state of the system."""
        state_array, state, face_states, avg_face_states, face_gradients = (
            self.precompute(time, state_array, gamma_star, e0_star)
        )

        return self.source_implementation(
            time, state_array, state, face_states, avg_face_states, face_gradients
        )

    @abstractmethod
    def source_implementation(
        self,
        time: float,
        state_array: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Concrete implementation of the source term calculation."""


class CombinedSource(RightHandSide):
    def __init__(self, sources: list[RightHandSide]) -> None:
        self.sources = sources

    def source(
        self,
        time: float,
        state_array: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        rhs = np.zeros_like(state_array)

        for source in self.sources:
            rhs[source.idx_domain, source.idx_source] += source.source(
                time=time,
                state_array=state_array,
                gamma_star=gamma_star,
                e0_star=e0_star,
            )

        return rhs
