from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cantera as ct
import numpy as np

from stanshock.system.backend import Array, Composition


@dataclass
class FluidState:
    shape: tuple[int, ...]

    density: Array | None = None
    temperature: Array | None = None
    pressure: Array | None = None
    internal_energy: Array | None = None
    mass_fractions: Array | None = None
    mole_fractions: Array | None = None
    mixture_fraction: Array | None = None
    progress_variable: Array | None = None
    normalized_progress_variable: Array | None = None
    composition: Array | None = None

    cp: Array | None = None
    gamma: Array | None = None
    viscosity: Array | None = None
    thermal_conductivity: Array | None = None
    sound_speed: Array | None = None

    velocity: Array | None = None

    gamma_star: Array | None = None
    e0_star: Array | None = None

    _cache_valid: bool = False


class FluidPhysics(ABC):
    def __init__(
        self,
        gas: ct.Solution,
        ox_def: Composition | None = None,
        fuel_def: Composition | None = None,
        prog_def: Composition | None = None,
    ) -> None:
        self.gas: ct.Solution = gas
        self.n_scalars: int = self.gas.n_species
        self.n_scalars_rho_sum: int = self.n_scalars
        self.scalar_names: list[str] = [
            species.lower() for species in self.gas.species_names
        ]

        self.is_flamelet: bool = False

        # For mixture fraction and progress variable definitions (optional):
        self.ox_def: Composition | None = ox_def  # Oxidizer molar composition
        self.fuel_def: Composition | None = fuel_def  # Fuel molar composition
        self.prog_def: Composition | None = (
            prog_def  # Progress variable molar composition
        )
        self.Z_weights: Array = np.zeros(self.gas.n_species)
        self.Z_offset: float = 0.0
        self.prog_weights: Array = np.zeros(self.gas.n_species)

        if self.ox_def is not None and self.fuel_def is not None:
            self.initialize_bilger_mixture_fraction()

        if self.prog_def is not None:
            self.initialize_progress_variable(self.prog_def)

    @abstractmethod
    def get_cp(self, state: FluidState) -> Array:
        """Compute specific heat capacity at constant pressure."""

    @abstractmethod
    def get_gamma(self, state: FluidState) -> Array:
        """Compute specific heat ratio, gamma."""

    def get_double_flux_variables(self, state: FluidState) -> tuple[Array, Array]:
        """Compute effective specific heat ratio, gamma*, and reference energy, e_0^*."""
        assert state.internal_energy is not None
        assert state.pressure is not None
        assert state.density is not None
        # Valid for any ideal gas, g* = rho*c^2/p = g
        state.gamma_star = self.get_gamma(state)
        state.e0_star = state.internal_energy - state.pressure / (
            state.density * (state.gamma_star - 1.0)
        )

        return state.gamma_star, state.e0_star

    @abstractmethod
    def get_mu(self, state: FluidState) -> Array:
        """Compute dynamic viscosity."""

    @abstractmethod
    def get_thermal_conductivity(self, state: FluidState) -> Array:
        """Compute thermal conductivity."""

    @abstractmethod
    def get_temperature(self, state: FluidState) -> Array:
        """Compute temperature of the gas."""

    @abstractmethod
    def get_pressure(self, state: FluidState) -> Array:
        """Compute pressure of the gas."""

    @abstractmethod
    def get_internal_energy(self, state: FluidState) -> Array:
        """Compute internal energy of the gas."""

    @abstractmethod
    def get_species_enthalpies(self, state: FluidState) -> Array:
        """Compute total enthalpies of each species."""

    @abstractmethod
    def get_sound_speed(self, state: FluidState) -> Array:
        """Compute speed of sound of the gas."""

    def get_thermal_diffusivity(self, state: FluidState) -> Array:
        """Compute thermal diffusivity, alpha = kappa / (rho * cp)."""
        kappa = self.get_thermal_conductivity(state)
        density = state.density
        assert density is not None
        cp = self.get_cp(state)
        return kappa / (density * cp)

    def get_mass_diffusivity(self, state: FluidState) -> Array:
        """Compute mass diffusivity (Unity Lewis Number assumption)."""
        return self.get_thermal_diffusivity(state)[:, None]

    def initialize_bilger_mixture_fraction(self) -> None:
        """Compute coefficients defining Bilger mixture fraction."""
        assert self.ox_def is not None
        assert self.fuel_def is not None

        self.Z_weights = np.zeros(self.gas.n_species)
        self.Z_offset = 0.0
        denom = 0.0

        # Set the values for C, H, and O:
        stoich: dict[str, float] = {
            "C": 2.0,
            "H": 0.5,
            "O": -1.0,
        }

        for element in self.gas.element_names:
            if element not in stoich:
                continue
            C = stoich[element]

            idx_element = self.gas.element_index(element)
            W = self.gas.atomic_weight(element)

            self.gas.X = self.ox_def
            Yo = self.gas.elemental_mass_fraction(element)

            self.gas.X = self.fuel_def
            Yf = self.gas.elemental_mass_fraction(element)

            denom += C * (Yf - Yo) / W

            for k in range(self.gas.n_species):
                self.Z_weights[k] += C * self.gas.n_atoms(k, idx_element)

            self.Z_offset -= C * Yo / W

        self.Z_weights /= denom * self.gas.molecular_weights
        self.Z_offset /= denom

    def get_bilger_mixture_fraction(self, Y: Array) -> Array:
        """Compute the Bilger mixture fraction from given mass fractions."""
        assert self.Z_weights is not None
        assert self.Z_offset is not None
        tmp = np.dot(Y, self.Z_weights)
        assert isinstance(tmp, np.ndarray)
        return np.clip(tmp + self.Z_offset, 0.0, 1.0)

    def initialize_progress_variable(self, prog_def: Composition) -> None:
        """Set coefficients defining progress variable."""
        self.prog_weights = np.zeros(self.gas.n_species)

        for sp, val in prog_def.items():
            self.prog_weights[self.gas.species_index(sp)] = val

        if np.sum(self.prog_weights) == 0.0:
            msg = "Progress Variable Weights Sum to Zero"
            raise Exception(msg)

        self.prog_weights /= np.sum(self.prog_weights)

    def get_progress_variable(self, Y: Array) -> Array:
        """Compute the progress variable from given mass fractions."""
        if self.prog_def is None:
            msg = "Progress Variable Not Defined"
            raise Exception(msg)

        tmp = np.dot(Y, self.prog_weights)
        assert isinstance(tmp, np.ndarray)
        return np.clip(tmp, 0.0, 1.0)

    def set_state(self, state: FluidState) -> FluidState:
        """Updates internal representation of the fluid state if needed."""
        state._cache_valid = True
        return state

    def get_composition(self, Y: Array) -> Array:
        """Converts mass fractions to set of transported scalars."""
        return Y

    def get_normalized_progress_variable(self, Z: Array, C: Array) -> Array:
        _ = Z, C
        msg = f"Normalized progress variable not implemented for {self.__class__.__name__}."
        raise NotImplementedError(msg)

    def lookup(self, var: str, state: FluidState) -> Array:
        _ = var, state
        msg = f"Looking up a variable by string is not implemented for {self.__class__.__name__}."
        raise NotImplementedError(msg)

    def primitive_to_conservative(self, state: FluidState) -> Array:
        """Transform primitive variables into vector of conservatives, accounting for chemical contributions."""
        # Compute total energy including chemical contributions
        state = self.set_state(state)
        assert state.velocity is not None
        assert state.density is not None
        assert state.composition is not None

        e_int = state.internal_energy
        if e_int is None:
            e_int = self.get_internal_energy(state)
        total_energy = e_int + 0.5 * state.velocity**2

        return np.concatenate(
            (
                (state.density * state.velocity)[..., None],
                (state.density * total_energy)[..., None],
                state.density[..., None] * state.composition,
            ),
            axis=-1,
            dtype=np.float64,
        )

    def conservative_to_primitive(
        self,
        state_array: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> FluidState:
        """Transform conservative variables into primitives, accounting for chemical contributions."""
        ru = state_array[..., 0]
        re_t = state_array[..., 1]
        rY = state_array[..., 2:]

        # Enforce non-negativity
        rY = np.clip(rY, 0, None, out=rY)

        r = np.sum(rY[..., : self.n_scalars_rho_sum], axis=-1, dtype=np.float64)
        assert isinstance(r, np.ndarray)
        u = ru / r
        e_int = (re_t / r) - 0.5 * u**2.0
        Y = rY / r[..., None]

        state = FluidState(
            shape=r.shape,
            density=r,
            velocity=u,
            internal_energy=e_int,
            composition=Y,
        )

        if gamma_star is not None:
            # Compute pressure using double-flux method
            assert e0_star is not None
            state.pressure = (gamma_star - 1.0) * r * (e_int - e0_star)

        return self.set_state(state)

    @abstractmethod
    def get_source_terms(self, state: FluidState) -> Array:
        """Compute reaction source terms corresponding to transported scalars."""
