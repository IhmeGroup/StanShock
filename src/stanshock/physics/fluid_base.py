from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cantera as ct
import numpy as np

from stanshock.system.backend import Array


@dataclass
class FluidState:
    shape: tuple[int, ...]

    density: np.ndarray | None = None
    temperature: np.ndarray | None = None
    pressure: np.ndarray | None = None
    internal_energy: np.ndarray | None = None
    mass_fractions: np.ndarray | None = None
    mole_fractions: np.ndarray | None = None
    mixture_fraction: np.ndarray | None = None
    progress_variable: np.ndarray | None = None
    normalized_progress_variable: np.ndarray | None = None
    composition: np.ndarray | None = None

    cp: np.ndarray | None = None
    gamma: np.ndarray | None = None
    viscosity: np.ndarray | None = None
    thermal_conductivity: np.ndarray | None = None
    sound_speed: np.ndarray | None = None

    velocity: np.ndarray | None = None

    _cache_valid: bool = False


class FluidPhysics(ABC):
    def __init__(self, gas: ct.Solution, ox_def=None, fuel_def=None, prog_def=None):
        self.gas = gas
        self.n_scalars = max(self.gas.n_species - 1, 1)
        self.scalar_names = [
            self.gas.species_name(i).lower() for i in range(self.n_scalars)
        ]

        self.is_flamelet = False

        # For mixture fraction and progress variable definitions (optional):
        self.ox_def = ox_def  # Oxidizer molar composition
        self.fuel_def = fuel_def  # Fuel molar composition
        self.prog_def = prog_def  # Progress variable molar composition
        self.Z_weights = None
        self.Z_offset = None
        self.prog_weights = None

        if self.ox_def is not None and self.fuel_def is not None:
            self.initialize_bilger_mixture_fraction()

        if self.prog_def is not None:
            self.initialize_progress_variable(self.prog_def)

    @abstractmethod
    def get_cp(self, state: FluidState):
        """Compute specific heat capacity at constant pressure."""

    @abstractmethod
    def get_gamma(self, state: FluidState):
        """Compute specific heat ratio, gamma."""

    @abstractmethod
    def get_mu(self, state: FluidState):
        """Compute dynamic viscosity."""

    @abstractmethod
    def get_thermal_conductivity(self, state: FluidState):
        """Compute thermal conductivity."""

    @abstractmethod
    def get_temperature(self, state: FluidState):
        """Compute temperature of the gas."""

    @abstractmethod
    def get_pressure(self, state: FluidState):
        """Compute pressure of the gas."""

    @abstractmethod
    def get_sound_speed(self, state: FluidState):
        """Compute speed of sound of the gas."""

    def get_thermal_diffusivity(self, state: FluidState):
        """Compute thermal diffusivity, alpha = kappa / (rho * cp)."""
        kappa = self.get_thermal_conductivity(state)
        density = state.density
        cp = self.get_cp(state)
        return kappa / (density * cp)

    def get_mass_diffusivity(self, state: FluidState):
        """Compute mass diffusivity (Unity Lewis Number assumption)."""
        return self.get_thermal_diffusivity(state)[:, None]

    def initialize_bilger_mixture_fraction(self):
        """Compute coefficients defining Bilger mixture fraction."""
        self.Z_weights = np.zeros(self.gas.n_species)
        self.Z_offset = 0.0
        denom = 0.0

        # Set the values for C, H, and O:
        stoich = {
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

    def get_bilger_mixture_fraction(self, Y):
        """Compute the Bilger mixture fraction from given mass fractions."""
        if self.gas.n_species > 1:
            Y_full = np.zeros(self.gas.n_species)
            Y_full[:-1] = Y
            Y_full[-1] = 1.0 - np.sum(Y_full[:-1])
        else:
            Y_full = Y
        return np.clip(np.dot(Y_full, self.Z_weights) + self.Z_offset, 0.0, 1.0)

    def initialize_progress_variable(self, prog_def: dict[str, float]):
        """Set coefficients defining progress variable."""
        self.prog_weights = np.zeros(self.gas.n_species)

        for sp, val in prog_def.items():
            self.prog_weights[self.gas.species_index(sp)] = val

        if np.sum(self.prog_weights) == 0.0:
            msg = "Progress Variable Weights Sum to Zero"
            raise Exception(msg)

        self.prog_weights /= np.sum(self.prog_weights)

    def get_progress_variable(self, Y):
        """Compute the progress variable from given mass fractions."""
        if self.prog_weights is None:
            msg = "Progress Variable Not Defined"
            raise Exception(msg)

        if self.gas.n_species > 1:
            Y_full = np.zeros(self.gas.n_species)
            Y_full[:-1] = Y
            Y_full[-1] = 1.0 - np.sum(Y_full[:-1])
        else:
            Y_full = Y
        return np.clip(np.dot(Y_full, self.prog_weights), 0.0, 1.0)

    def set_state(self, state: FluidState) -> FluidState:
        """Updates internal representation of the fluid state if needed."""
        self._cache_valid = True
        return state

    def get_composition(self, Y):
        """Converts mass fractions to set of transported scalars."""
        return Y

    def get_normalized_progress_variable(self, Z, C):
        msg = f"Normalized progress variable not implemented for {self.__class__}."
        raise NotImplementedError(msg)

    def lookup(self, var: str, state: FluidState):
        msg = (
            f"Looking up a variable by string is not implemented for {self.__class__}."
        )
        raise NotImplementedError(msg)

    @abstractmethod
    def primitive_to_conservative(self, state: FluidState):
        """Transform primitive variables into vector of conservatives."""

    @abstractmethod
    def conservative_to_primitive(self, state_array: Array, gamma: Array) -> FluidState:
        """Transform conservative variables into primitives."""

    @abstractmethod
    def get_source_terms(self, state: FluidState):
        """Compute reaction source terms corresponding to transported scalars."""

    @abstractmethod
    def get_viscous_flux(
        self,
        face_states: FluidState,
        dudx: Array,
        dTdx: Array,
        dYdx: Array,
    ) -> Array:
        """Compute viscous fluxes."""
