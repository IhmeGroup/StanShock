from __future__ import annotations

import numpy as np
from cantera import gas_constant

from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Unpack
from stanshock.system.base import PrecomputeSteps, RightHandSide


class ChemistrySource(RightHandSide):
    PRECOMPUTE_STEPS = ("geometry", "physics")

    def __init__(self, **precompute_steps: Unpack[PrecomputeSteps]) -> None:
        super().__init__(**precompute_steps)
        self.idx_source = np.s_[2:]

    def source_implementation(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Compute the temporal gradient of the current state of the system."""
        _ = time, state_array_local, face_states, avg_face_states, face_gradients
        assert self.physics is not None
        assert state is not None
        return self.physics.get_source_terms(state)


class ConstantVolumeChemistry(RightHandSide):
    def __init__(self, **precompute_steps: Unpack[PrecomputeSteps]) -> None:
        super().__init__(**precompute_steps)

        # Restrict the domain to the cells
        n_species = self.physics.n_scalars
        self.idx_domain = self.geometry.idx_cells
        self.idx_update = np.s_[:]
        self.shape_subset: tuple[int, ...] = (self.shape_update[0], n_species + 1)

        # Define density and velocity as constants to be set before time integration
        self.density_initial: Array = np.zeros((0,))
        self.velocity_initial: Array = np.zeros((0,))

    def before_time_integration(
        self,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[Array, Array | None, Array | None]:
        """Store temperature and mass fractions in the state array."""
        state_array_local, gamma_star_local, e0_star_local = (
            super().before_time_integration(time, state_array, gamma_star, e0_star)
        )

        state = self.physics.conservative_to_primitive(
            state_array_local.reshape(self.shape_update),
            gamma_star_local,
            e0_star_local,
        )

        assert state.density is not None
        self.density_initial = state.density
        assert state.velocity is not None
        self.velocity_initial = state.velocity

        assert state.composition is not None
        state_array_new: Array = np.zeros(self.shape_subset)
        state_array_new[:, 0] = self.physics.get_temperature(state)
        state_array_new[:, 1:] = state.composition

        return np.ravel(state_array_new), gamma_star_local, e0_star_local

    def after_time_integration(
        self,
        time: float,
        state_array_local: Array,
        gamma_star_local: Array | None,
        e0_star_local: Array | None,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[Array, Array | None, Array | None]:
        # Compute conservative state from initial density and updated mass fractions + temperature
        state_array_local = np.reshape(state_array_local, self.shape_subset)
        state: FluidState = FluidState(
            shape=(self.shape_subset[0],),
            density=self.density_initial,
            velocity=self.velocity_initial,
            temperature=state_array_local[:, 0],
            composition=state_array_local[:, 1:],
            gamma_star=gamma_star_local,
            e0_star=e0_star_local,
        )
        state.pressure = self.physics.get_pressure(state)
        state_array_local = self.physics.primitive_to_conservative(state)

        # Update double-flux variables
        if gamma_star is not None:
            gamma_star_local, e0_star_local = self.physics.get_double_flux_variables(
                state
            )
            gamma_star[self.idx_domain] = gamma_star_local
            assert e0_star is not None
            e0_star[self.idx_domain] = e0_star_local

        # Update the state array
        state_array = np.reshape(state_array, self.shape_full)
        state_array[self.idx_domain] = state_array_local

        # Update the domain indices for next time step
        self.update_indices(time, state)

        return np.ravel(state_array), gamma_star, e0_star

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
        """Store temperature and mass fractions in the state array."""
        _ = time, gamma_star, e0_star
        state_array_local = state_array_local.reshape(self.shape_subset)

        state: FluidState = FluidState(
            shape=(self.shape_subset[0],),
            density=self.density_initial,
            velocity=self.velocity_initial,
            temperature=state_array_local[:, 0],
            composition=state_array_local[:, 1:],
        )

        return np.ravel(state_array_local), state, None, None, None

    def source_implementation(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        _ = time, state_array_local, face_states, avg_face_states, face_gradients
        assert isinstance(self.physics, CanteraInterface)
        assert state is not None
        assert state.temperature is not None
        temperature = state.temperature
        wdot: Array = self.physics.get_source_terms(state)

        eRT = self.physics.sol.standard_int_energies_RT
        cv = self.physics.get_cp(state) / self.physics.get_gamma(state)

        dydt: Array = np.zeros(self.shape_subset)
        dydt[:, 0] = -np.sum(eRT * wdot, axis=-1) * (
            gas_constant * temperature / (self.density_initial * cv)
        )
        dydt[:, 1:] = (
            wdot
            * self.physics.gas.molecular_weights[None, :]
            / self.density_initial[:, None]
        )

        return np.ravel(dydt)
