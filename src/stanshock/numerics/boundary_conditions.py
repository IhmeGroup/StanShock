from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Literal, TypeVar, Union

from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Index, TypeAlias, np

T = TypeVar("T")


class BoundaryCondition(ABC, Generic[T]):
    def __init__(self, location: Literal["left", "right"] = "left") -> None:
        self.location = location

    @abstractmethod
    def update(self, time: float, target: T) -> T:
        """Update the target of the specific boundary condition type."""


class GhostCell(BoundaryCondition[Array]):
    """Update the conservative variables in the ghost layers."""


class RiemannFlux(BoundaryCondition[FluidState]):
    """Update the primitive variables at the boundary face (input to Riemann solver)."""


class SpecifiedFlux(BoundaryCondition[Array]):
    """Directly set the flux through the boundary face."""


class PadCells(GhostCell):
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
        target[self.idx_exterior, :] = target[self.idx_interior, :]

        return target


class Periodic(GhostCell):
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
        target[self.idx_exterior, :] = target[self.idx_interior, :]

        return target


class Symmetry(GhostCell):
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
        target[self.idx_exterior, :] = target[self.idx_interior, :]
        target[self.idx_exterior, 1] = -target[self.idx_interior, 1]

        return target


class Extrapolate(RiemannFlux):
    """Copy extrapolated fluid state from interior of the boundary face to the exterior side."""

    def __init__(self, location: Literal["left", "right"] = "left") -> None:
        super().__init__(location)
        if self.location == "left":
            self.idx_internal: tuple[int, int] = (1, 0)
            self.idx_external: tuple[int, int] = (0, 0)
        elif location == "right":
            self.idx_internal = (0, -1)
            self.idx_external = (1, -1)

    def update(self, time: float, target: FluidState) -> FluidState:
        _: float = time
        assert target.density is not None
        assert target.velocity is not None
        assert target.pressure is not None
        assert target.composition is not None

        target.density[self.idx_external] = target.density[self.idx_internal]
        target.velocity[self.idx_external] = target.velocity[self.idx_internal]
        target.pressure[self.idx_external] = target.pressure[self.idx_internal]
        target.composition[self.idx_external, :] = target.composition[
            self.idx_internal, :
        ]

        return target


class AdiabaticWall(Extrapolate):
    """Set the velocity at the wall face to zero."""

    def update(self, time: float, target: FluidState) -> FluidState:
        target = super().update(time, target)
        assert target.velocity is not None
        target.velocity[self.idx_external] = -target.velocity[self.idx_external]

        return target


class Inflow(Extrapolate):
    """Specify the fluid state at the wall face."""

    def __init__(
        self,
        reference_state: Sequence[float | Sequence[float] | None],
        location: Literal["left", "right"] = "left",
    ) -> None:
        super().__init__(location)

        self.reference_state = reference_state

    def update(self, time: float, target: FluidState) -> FluidState:
        target = super().update(time, target)

        if self.reference_state[0] is not None:
            assert target.density is not None
            target.density[self.idx_external] = self.reference_state[0]
        if self.reference_state[1] is not None:
            assert target.velocity is not None
            target.velocity[self.idx_external] = self.reference_state[1]
        if self.reference_state[2] is not None:
            assert target.pressure is not None
            target.pressure[self.idx_external] = self.reference_state[2]
        if self.reference_state[3] is not None:
            assert target.composition is not None
            target.composition[self.idx_external, :] = self.reference_state[3]

        return target


class DirichletInflow(SpecifiedFlux):
    """Directly set the flux through the wall face."""

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
        target[self.idx_boundary_face, :] = self.reference_flux

        return target


BCType: TypeAlias = Union[BoundaryCondition[FluidState], BoundaryCondition[Array]]


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
    def riemann_fluxes(self) -> list[RiemannFlux]:
        return [
            boundary_condition
            for boundary_condition in self._boundary_conditions
            if isinstance(boundary_condition, RiemannFlux)
        ]

    @property
    def specified_fluxes(self) -> list[SpecifiedFlux]:
        return [
            boundary_condition
            for boundary_condition in self._boundary_conditions
            if isinstance(boundary_condition, SpecifiedFlux)
        ]

    def update_ghost_layers(self, time: float, state_array: Array) -> Array:
        for ghost_cell in self.ghost_cells:
            state_array = ghost_cell.update(time, target=state_array)

        return state_array

    def update_face_states(self, time: float, face_states: FluidState) -> FluidState:
        for riemann_flux in self.riemann_fluxes:
            face_states = riemann_flux.update(time, target=face_states)

        return face_states

    def update_face_flux(self, time: float, face_flux: Array) -> Array:
        for specified_flux in self.specified_fluxes:
            face_flux = specified_flux.update(time, target=face_flux)

        return face_flux


BCNamesType: TypeAlias = Literal[
    "outflow", "symmetry", "reflecting", "wall", "periodic"
]


def set_boundary_conditions(
    boundary_conditions: BoundaryConditions | Sequence[BCNamesType | BCType],
    mt: int = 3,
) -> BoundaryConditions:
    """Convenience function to initialize different boundary conditions."""

    # Convert lists into BoundaryConditions:
    if not isinstance(boundary_conditions, BoundaryConditions):
        bc_locs: list[Literal["left", "right"]] = ["left", "right"]
        bcs: list[BCType] = []

        for bc_loc, bc_specification in zip(bc_locs, boundary_conditions):
            if isinstance(bc_specification, str):
                if bc_specification == "periodic":
                    bcs += [Periodic(mt, location=bc_loc)]
                elif bc_specification == "outflow":
                    bcs += [Extrapolate(location=bc_loc)]
                elif bc_specification in ["symmetry"]:
                    bcs += [Symmetry(mt, location=bc_loc)]
                elif bc_specification in ["reflecting", "wall"]:
                    bcs += [AdiabaticWall(location=bc_loc)]
            elif isinstance(bc_specification, BoundaryCondition):
                bcs += [bc_specification]

        boundary_conditions = BoundaryConditions(boundary_conditions=bcs)

    # If ghost layer method not specified, default to simple padding
    default_ghost_layers: dict[str, GhostCell] = {
        "left": PadCells(mt, location="left"),
        "right": PadCells(mt, location="right"),
    }
    for bc in boundary_conditions._boundary_conditions:
        if isinstance(bc, Periodic):
            default_ghost_layers = {}
        elif isinstance(bc, GhostCell):
            del default_ghost_layers[bc.location]

    for _, bc in default_ghost_layers.items():
        boundary_conditions.append(bc)

    return boundary_conditions
