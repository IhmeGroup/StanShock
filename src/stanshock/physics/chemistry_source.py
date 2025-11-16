from __future__ import annotations

import numpy as np
from cantera import gas_constant

from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Index, Unpack
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
    PRECOMPUTE_STEPS = ("geometry", "physics")

    def __init__(self, **precompute_steps: Unpack[PrecomputeSteps]) -> None:
        super().__init__(**precompute_steps)
        assert self.physics is not None
        self.idx_source: Index = np.arange(1, self.physics.n_scalars + 2)

        # Define some parameters to be set during precompute step
        self.density0: Array = np.zeros((0,))
        self.shape_subset: tuple[int, ...] = (0, 0)

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
        _ = time
        assert self.physics is not None
        n_species = self.physics.n_scalars
        state_array_local = state_array_local.reshape(self.shape)
        self.shape = state_array_local.shape

        state_array_local = state_array_local[self.idx_update]
        self.shape_subset = state_array_local.shape

        state: FluidState = self.physics.conservative_to_primitive(
            state_array_local, gamma_star, e0_star
        )

        assert state.density is not None
        self.density0 = state.density

        state_array_new: Array = np.zeros((self.shape_subset[0], n_species + 1))
        state_array_new[:, :-1] = state.composition
        state_array_new[:, -1] = self.physics.get_temperature(state)

        return state_array_new, state, None, None, None

    def source(
        self,
        time: float,
        state_array_local: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        _ = time, gamma_star, e0_star
        assert self.physics is not None

        temperature = state_array_local[:, -1]

        state: FluidState = FluidState(
            shape=(self.shape_subset[0],),
            density=self.density0,
            temperature=temperature,
            composition=state_array_local[:, :-1],
        )
        wdot: Array = self.physics.get_source_terms(state)

        eRT = self.physics.sol.standard_int_energies_RT
        cv = self.physics.get_cp(state) / self.physics.get_gamma(state)

        dydt: Array = np.zeros(self.shape_subset)
        dydt[:, :-1] = (
            wdot * self.physics.gas.molecular_weights[None, :] / self.density0[:, None]
        )
        dydt[:, -1] = (
            -np.sum(eRT * wdot, axis=-1, keepdims=True)
            * (gas_constant * temperature / (self.density0 * cv))[:, None]
        )

        return np.ravel(dydt)

    def postcompute(self, state_array_local: Array, state: FluidState) -> Array:
        # Compute state from initial density and updated mass fractions + temperature
        assert self.physics is not None
        state: FluidState = FluidState(
            shape=(self.shape_subset[0],),
            density=self.density0,
            temperature=state_array_local[:, -1],
            composition=state_array_local[:, :-1],
        )
        state.pressure = self.physics.get_pressure(state)

        state_array_local = self.physics.primitive_to_conservative(state)

        # Update the state array
        state_array_full = np.zeros(self.shape)
        state_array_full[self.idx_update, self.idx_source] = state_array_local[
            :, self.idx_source
        ]

        return np.ravel(state_array_local)
