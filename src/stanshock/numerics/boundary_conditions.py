from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Literal, TypedDict, TypeVar

from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Index, TypeAlias, np

_T = TypeVar("_T")


class BoundaryCondition(ABC, Generic[_T]):
    def __init__(self, location: Literal["left", "right"] = "left") -> None:
        self.location = location

    @abstractmethod
    def update(self, time: float, target: _T) -> _T:
        """Update the target of the specific boundary condition type."""


class GhostCell(BoundaryCondition[Array]):
    """Update the conservative variables in the ghost layers."""


class GhostCellPrimitive(BoundaryCondition[FluidState]):
    """Update the primitive variables in the ghost layers."""


class RiemannFlux(BoundaryCondition[FluidState]):
    """Update the primitive variables at the boundary face (input to Riemann solver)."""


class DirichletFlux(BoundaryCondition[Array]):
    """Directly set the flux through the boundary face."""


class FreezeCells(GhostCell):
    """Hold ghost cells constant (do nothing)."""

    def update(self, time: float, target: Array) -> Array:
        _: float = time
        return target


class ExtrapolateCells(GhostCell):
    """Extrapolates constant values from the last interior cell into the ghost layers."""

    def __init__(
        self, mt: int = 3, location: Literal["left", "right"] = "left"
    ) -> None:
        super().__init__(location)
        if self.location == "left":
            self.idx_interior: int = mt
            self.idx_exterior: Index = np.s_[:mt]
        else:
            self.idx_interior = -mt - 1
            self.idx_exterior = np.s_[-mt:]

    def update(self, time: float, target: Array) -> Array:
        _: float = time
        target[self.idx_exterior] = target[self.idx_interior]

        return target


class ExtrapolateCellsLinear(GhostCell):
    """Linearly extrapolates values from the interior cells into the ghost layers."""

    def __init__(
        self, mt: int = 3, location: Literal["left", "right"] = "left"
    ) -> None:
        super().__init__(location)
        if self.location == "left":
            self.idx_interior: Index = np.s_[mt : mt + 2]
            self.idx_exterior: Index = np.s_[:mt]
        else:
            self.idx_interior = np.s_[-mt - 2 : -mt]
            self.idx_exterior = np.s_[-mt:]
        self.mt = mt
        self.delta = np.arange(self.mt, 0, -1)[:, None]

    def update(self, time: float, target: Array) -> Array:
        _: float = time
        target[self.idx_exterior] = target[self.idx_interior][
            [0]
        ] - self.delta * np.diff(target[self.idx_interior], axis=0)

        return target


class PeriodicCells(GhostCell):
    """Replicates solution from opposite end of the domain into the ghost layers."""

    def __init__(
        self, mt: int = 3, location: Literal["left", "right"] = "left"
    ) -> None:
        super().__init__(location)
        if self.location == "left":
            self.idx_interior: Index = np.s_[-2 * mt : -mt]
            self.idx_exterior: Index = np.s_[:mt]
        else:
            self.idx_interior = np.s_[mt : 2 * mt]
            self.idx_exterior = np.s_[-mt:]

    def update(self, time: float, target: Array) -> Array:
        _: float = time
        target[self.idx_exterior] = target[self.idx_interior]

        return target


class SymmetryCells(GhostCell):
    """Mirrors interior solution into the ghost layers."""

    def __init__(
        self, mt: int = 3, location: Literal["left", "right"] = "left"
    ) -> None:
        super().__init__(location)
        if self.location == "left":
            self.idx_interior: Index = np.s_[mt : 2 * mt]
            self.idx_exterior: Index = np.s_[mt - 1 :: -1]
        else:
            self.idx_interior = np.s_[-2 * mt : -mt]
            self.idx_exterior = np.s_[: -mt - 1 : -1]

    def update(self, time: float, target: Array) -> Array:
        _: float = time
        target[self.idx_exterior] = target[self.idx_interior]
        target[self.idx_exterior][:, 0] = -target[self.idx_interior][:, 0]

        return target


class DeactivateWenoCells(GhostCellPrimitive):
    """Applies large values and variance to primitives in ghost layers.

    This is one approach to forcing the WENO stencil to only consider interior cells.
    """

    def __init__(
        self, mt: int = 3, location: Literal["left", "right"] = "left"
    ) -> None:
        super().__init__(location)
        if self.location == "left":
            self.values = (10.0 * np.arange(mt, 0, -1)) ** 10.0
            self.idx_exterior: Index = np.s_[:mt]
        else:
            self.values = (10.0 * np.arange(1, mt + 1)) ** 10.0
            self.idx_exterior = np.s_[-mt:]

    def update(self, time: float, target: FluidState) -> FluidState:
        _: float = time
        assert target.density is not None
        assert target.velocity is not None
        assert target.pressure is not None
        assert target.composition is not None

        values = self.values.copy()
        target.density[self.idx_exterior] = values
        target.velocity[self.idx_exterior] = values
        target.pressure[self.idx_exterior] = values
        target.composition[self.idx_exterior] = values[:, None]

        return target


class ExtrapolateFace(RiemannFlux):
    """Copy extrapolated fluid state from interior of the boundary face to the exterior side."""

    def __init__(self, location: Literal["left", "right"] = "left") -> None:
        super().__init__(location)
        if self.location == "left":
            self.idx_interior: tuple[int, int] = (1, 0)
            self.idx_exterior: tuple[int, int] = (0, 0)
        elif location == "right":
            self.idx_interior = (0, -1)
            self.idx_exterior = (1, -1)

    def update(self, time: float, target: FluidState) -> FluidState:
        _: float = time
        assert target.density is not None
        assert target.velocity is not None
        assert target.pressure is not None
        assert target.composition is not None

        target.density[self.idx_exterior] = target.density[self.idx_interior]
        target.velocity[self.idx_exterior] = target.velocity[self.idx_interior]
        target.pressure[self.idx_exterior] = target.pressure[self.idx_interior]
        target.composition[self.idx_exterior] = target.composition[self.idx_interior]

        return target


class AdiabaticWallFace(ExtrapolateFace):
    """Set the velocity at the wall face to zero."""

    def update(self, time: float, target: FluidState) -> FluidState:
        target = super().update(time, target)
        assert target.velocity is not None
        target.velocity[self.idx_exterior] = -target.velocity[self.idx_exterior]

        return target


ReferenceStateType: TypeAlias = tuple[
    float | None, float | None, float | None, Array | Sequence[float] | None
]


class SpecifiedFace(ExtrapolateFace):
    """Fully or partially specify the fluid state at the boundary face.

    Unspecified properties will be extrapolated from interior cells.
    """

    def __init__(
        self,
        reference_state: ReferenceStateType,
        location: Literal["left", "right"] = "left",
    ) -> None:
        super().__init__(location)

        self.reference_state = reference_state

    def update(self, time: float, target: FluidState) -> FluidState:
        target = super().update(time, target)

        if self.reference_state[0] is not None:
            assert target.density is not None
            target.density[self.idx_exterior] = self.reference_state[0]
        if self.reference_state[1] is not None:
            assert target.velocity is not None
            target.velocity[self.idx_exterior] = self.reference_state[1]
        if self.reference_state[2] is not None:
            assert target.pressure is not None
            target.pressure[self.idx_exterior] = self.reference_state[2]
        if self.reference_state[3] is not None:
            assert target.composition is not None
            target.composition[self.idx_exterior] = self.reference_state[3]

        return target


class SpecifiedFlux(DirichletFlux):
    """Directly set the flux through the boundary face."""

    def __init__(
        self, reference_flux: Array, location: Literal["left", "right"] = "left"
    ) -> None:
        super().__init__(location)
        if self.location == "left":
            self.idx_boundary_face = 0
        else:
            self.idx_boundary_face = -1

        self.reference_flux = reference_flux

    def update(self, time: float, target: Array) -> Array:
        _: float = time
        target[self.idx_boundary_face] = self.reference_flux

        return target


BCType: TypeAlias = BoundaryCondition[FluidState] | BoundaryCondition[Array]


class BoundaryConditions:
    def __init__(self, boundary_conditions: list[BCType]) -> None:
        self._boundary_conditions = boundary_conditions

    def append(self, boundary_condition: BCType) -> None:
        self._boundary_conditions += [boundary_condition]

    @property
    def ghost_cells(self) -> list[GhostCell]:
        return [
            boundary_condition
            for boundary_condition in self._boundary_conditions
            if isinstance(boundary_condition, GhostCell)
        ]

    @property
    def ghost_cells_primitive(self) -> list[GhostCellPrimitive]:
        return [
            boundary_condition
            for boundary_condition in self._boundary_conditions
            if isinstance(boundary_condition, GhostCellPrimitive)
        ]

    @property
    def riemann_fluxes(self) -> list[RiemannFlux]:
        return [
            boundary_condition
            for boundary_condition in self._boundary_conditions
            if isinstance(boundary_condition, RiemannFlux)
        ]

    @property
    def specified_fluxes(self) -> list[DirichletFlux]:
        return [
            boundary_condition
            for boundary_condition in self._boundary_conditions
            if isinstance(boundary_condition, DirichletFlux)
        ]

    def update_ghost_layers(self, time: float, state_array: Array) -> Array:
        for ghost_cell in self.ghost_cells:
            state_array = ghost_cell.update(time, target=state_array)

        return state_array

    def update_ghost_states(self, time: float, state: FluidState) -> FluidState:
        for ghost_cell_primitive in self.ghost_cells_primitive:
            state = ghost_cell_primitive.update(time, target=state)

        return state

    def update_face_states(self, time: float, face_states: FluidState) -> FluidState:
        for riemann_flux in self.riemann_fluxes:
            face_states = riemann_flux.update(time, target=face_states)

        return face_states

    def update_face_flux(self, time: float, face_flux: Array) -> Array:
        for specified_flux in self.specified_fluxes:
            face_flux = specified_flux.update(time, target=face_flux)

        return face_flux


BCNamesType: TypeAlias = Literal[
    "extrapolate", "outflow", "symmetry", "reflecting", "wall", "periodic"
]

BCLike: TypeAlias = BCType | BCNamesType | ReferenceStateType


class BCInput(TypedDict):
    left: Sequence[BCLike]
    right: Sequence[BCLike]


def set_boundary_conditions(
    boundary_conditions: BoundaryConditions | BCInput, mt: int = 3
) -> BoundaryConditions:
    """Convenience function to initialize different boundary conditions."""

    # If ghost layer method not specified, default to freezing (hold constant)
    default_ghost_layers: dict[str, GhostCell | GhostCellPrimitive] = {
        "left": ExtrapolateCells(mt=mt, location="left"),
        "right": ExtrapolateCells(mt=mt, location="right"),
    }

    # Convert lists into BoundaryConditions:
    if not isinstance(boundary_conditions, BoundaryConditions):
        bcs: list[BCType] = []
        bc_loc: Literal["left", "right"]
        for bc_loc, bc_specification in boundary_conditions.items():  # type: ignore[assignment]
            if isinstance(bc_specification, str):
                if bc_specification == "periodic":
                    bcs += [PeriodicCells(mt, location=bc_loc)]
                elif bc_specification == "extrapolate":
                    bcs += [ExtrapolateCells(mt, location=bc_loc)]
                elif bc_specification == "outflow":
                    bcs += [ExtrapolateFace(location=bc_loc)]
                elif bc_specification in ["symmetry", "reflecting"]:
                    bcs += [SymmetryCells(mt, location=bc_loc)]
                elif bc_specification == "wall":
                    bcs += [AdiabaticWallFace(location=bc_loc)]
            elif isinstance(bc_specification, BoundaryCondition):
                bcs += [bc_specification]
            elif isinstance(bc_specification, tuple):
                bcs += [
                    SpecifiedFace(reference_state=bc_specification, location=bc_loc)
                ]

        boundary_conditions = BoundaryConditions(boundary_conditions=bcs)

    # Don't use default GhostCell treatments if already specified
    for bc in boundary_conditions._boundary_conditions:
        if isinstance(bc, PeriodicCells):
            default_ghost_layers = {}
        elif isinstance(bc, GhostCell | GhostCellPrimitive):
            del default_ghost_layers[bc.location]

    for _, bc in default_ghost_layers.items():
        boundary_conditions.append(bc)

    return boundary_conditions
