from __future__ import annotations

from abc import abstractmethod
from typing import Literal, TypedDict

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
    REQUIRED_PRECOMPUTE_STEPS: tuple[PrecomputeStepName, ...] = ("geometry",)

    def __init__(self, **precompute_steps: Unpack[PrecomputeSteps]) -> None:
        # Any precompute steps not provided will default to None
        self.geometry = precompute_steps.get("geometry")
        self.physics = precompute_steps.get("physics")
        self.boundary_conditions = precompute_steps.get("boundary_conditions")
        self.face_extrapolator = precompute_steps.get("face_extrapolator")
        self.face_average = precompute_steps.get("face_average")
        self.gradient = precompute_steps.get("gradient")

        # Index into (1D) global state array to be accessed by this source term
        self.idx_domain: Index = np.s_[:]
        # Index of cells to which the source term applies
        self.idx_update: Index = self.geometry.idx_cells
        # Index of transport equations to update
        self.idx_source: Index = np.s_[:]

        # Ensure required routines were provided
        for step in self.REQUIRED_PRECOMPUTE_STEPS:
            if getattr(self, step) is None:
                msg: str = f"Must provide {step} for {self.__class__.__name__}."
                raise ValueError(msg)

        if self.physics is not None:
            self.shape: tuple[int, int] = (
                self.geometry.n_cells,
                self.physics.n_scalars + 2,
            )
        else:
            self.shape = (self.geometry.n_cells, -1)

    def update_indices(self, time: float) -> None:
        """Hook to update time-varying idx_update indices."""
        _ = time

    def before_time_integration(
        self,
        time: float,
        state_array: Array,
        update_double_flux: bool = True,
    ) -> tuple[Array, Array | None, Array | None]:
        """Extract the state_array to be operated on.

        May include optional forward transforms.
        """
        self.update_indices(time)

        state_array_local: Array = state_array[self.idx_domain].reshape(self.shape)
        gamma_star = None
        e0_star = None
        if update_double_flux:
            assert self.physics is not None
            state = self.physics.conservative_to_primitive(state_array_local)
            gamma_star, e0_star = self.physics.get_double_flux_variables(state)

        return state_array_local, gamma_star, e0_star

    def after_time_integration(
        self, state_array: Array, state_array_local: Array
    ) -> Array:
        """Insert the result back into the original state_array.

        Can also apply any necessary inverse transforms.
        """
        state_array[self.idx_domain] = np.ravel(state_array_local)

        return state_array

    def add_source(self, y: Array, dy: Array) -> Array:
        """Add (2D) source term to the (1D) state array."""
        state_array_local = y.reshape(self.shape)
        state_array_local[self.idx_update, self.idx_source] += dy
        return np.ravel(state_array_local)

    def precompute_for_source(
        self,
        time: float,
        state_array_local: Array,
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
        assert self.physics is not None
        state_array_local = state_array_local.reshape(self.shape)

        if self.boundary_conditions is not None:
            state_array_local = self.boundary_conditions.update_ghost_layers(
                time, state_array_local
            )

        state: FluidState | None = None
        if self.physics is not None:
            state = self.physics.conservative_to_primitive(
                state_array_local, gamma_star, e0_star
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

        return state_array_local, state, face_states, avg_face_states, face_gradients

    def source(
        self,
        time: float,
        state_array_local: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        """Compute the temporal gradient of the current state of the system."""
        state_array_local, state, face_states, avg_face_states, face_gradients = (
            self.precompute_for_source(time, state_array_local, gamma_star, e0_star)
        )

        return self.source_implementation(
            time, state_array_local, state, face_states, avg_face_states, face_gradients
        )

    def source_full(
        self, time: float, state_array_local: Array, gamma_star: Array, e0_star: Array
    ) -> Array:
        """Reshape the source term to match the state array."""
        dydt = np.zeros_like(state_array_local)

        return self.add_source(
            dydt, self.source(time, state_array_local, gamma_star, e0_star)
        )

    @abstractmethod
    def source_implementation(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Concrete implementation of the source term calculation."""

    def postcompute(self, state_array_local: Array, state: FluidState) -> Array:
        """Undo any transforms to the state array during precompute steps."""
        _ = state
        return np.ravel(state_array_local)


FastSlowMode: TypeAlias = Literal["fast", "slow"]


class FastSlowSource(RightHandSide):
    _mode: FastSlowMode
    idx_implicit: Index
    idx_explicit: Index

    def __init__(self, **precompute_steps: Unpack[PrecomputeSteps]) -> None:
        """Split RHS into fast and slow source terms accessed by setting the mode."""
        super().__init__(**precompute_steps)
        self.idx_implicit = np.array([], dtype=np.int64)
        self.idx_explicit = self.geometry.idx_cells
        self._mode = "slow"

    @property
    def mode(self) -> FastSlowMode:
        return self._mode

    @mode.setter
    def mode(self, mode: FastSlowMode) -> None:
        self._mode = mode
        if mode == "fast":
            self.idx_update = self.idx_implicit
            self.source_implementation = self.source_fast
        elif mode == "slow":
            self.idx_update = self.idx_explicit
            self.source_implementation = self.source_slow

    @abstractmethod
    def source_slow(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Slow source terms to be integrated explicitly."""

    @abstractmethod
    def source_fast(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Fast source terms to be integrated implicitly."""


class CombinedSource(RightHandSide):
    def __init__(self, sources: list[RightHandSide]) -> None:
        self.sources = sources

        # Dynamically determine the set of unique precompute steps required
        required_steps: set[PrecomputeStepName] = {
            x for source in self.sources for x in source.REQUIRED_PRECOMPUTE_STEPS
        }
        self.REQUIRED_PRECOMPUTE_STEPS = tuple(required_steps)

    def source(
        self,
        time: float,
        state_array_local: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        rhs: Array = np.zeros_like(state_array_local)

        state_array_local, state, face_states, avg_face_states, face_gradients = (
            self.precompute_for_source(time, state_array_local, gamma_star, e0_star)
        )

        for source in self.sources:
            dydt = source.source_implementation(
                time=time,
                state_array_local=state_array_local,
                state=state,
                face_states=face_states,
                avg_face_states=avg_face_states,
                face_gradients=face_gradients,
            )
            rhs = source.add_source(rhs, dydt)

        return rhs
