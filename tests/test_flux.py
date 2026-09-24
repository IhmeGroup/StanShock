from __future__ import annotations

import numpy as np
import pytest

from stanshock.components.combustor import Combustor
from stanshock.numerics.inviscid_flux import hllc_flux, lax_friedrichs_flux

lax_friedrichs_flux = lax_friedrichs_flux.__wrapped__  # unwrap for coverage testing
hllc_flux = hllc_flux.__wrapped__


def test_lax_friedrich_predicts_constant_flux():
    r = 5.0
    u = -2.0
    p = 3.0
    Y = 1.0
    gamma = 1.2
    e0 = 10.0
    flux = lax_friedrichs_flux(
        rLR=np.array([[r], [r]]),
        uLR=np.array([[u], [u]]),
        pLR=np.array([[p], [p]]),
        YLR=np.array([[[Y]], [[Y]]]),
        gamma=np.array([[gamma], [gamma]]),
        e0=np.array([[e0], [e0]]),
    )[0]
    H = gamma * p / (gamma - 1.0) + r * (e0 + 0.5 * u**2.0)
    expected_flux = np.array([r * u**2 + p, H * u, r * Y * u])
    assert np.allclose(flux, expected_flux)


def test_hllc_predicts_constant_flux():
    r = 2.0
    u = 0.5
    p = 3.0
    Y = 1.0
    gamma = 1.2
    e0 = 10.0
    num_faces = 10
    num_sides = 2
    num_species = 1
    shape = (num_sides, num_faces)
    flux = hllc_flux(
        rLR=np.full(shape, r),
        uLR=np.full(shape, u),
        pLR=np.full(shape, p),
        YLR=np.full((num_sides, num_faces, num_species), Y),
        gamma=np.full(shape, gamma),
        e0=np.full(shape, e0),
    )
    H = gamma * p / (gamma - 1.0) + r * (e0 + 0.5 * u**2.0)
    expected_flux = np.array([r * u**2 + p, H * u, r * Y * u])[np.newaxis, ...]
    expected_flux = np.repeat(expected_flux, num_faces, axis=0)
    assert np.allclose(flux, expected_flux)


def test_isentropic_flow_relations(isentropic_flow: Combustor) -> None:
    # Get initial state from given solution
    physics = isentropic_flow.physics
    t = isentropic_flow.t
    state = isentropic_flow.initialization()
    state_array = np.ravel(physics.primitive_to_conservative(state))
    gamma_star, e0_star = physics.get_double_flux_variables(state)

    # Get source terms from inviscid flux
    y, gamma_star_local, e0_star_local = (
        isentropic_flow.inviscid_flux.before_time_integration(
            t, state_array, gamma_star, e0_star
        )
    )
    source_flux = isentropic_flow.inviscid_flux.source(
        t, y, gamma_star_local, e0_star_local
    )

    # Get source terms from area change
    area_change = isentropic_flow.area_change
    assert area_change is not None
    area_change.update_indices(t, state)
    area_change.mode = "slow"
    y, gamma_star_local, e0_star_local = area_change.before_time_integration(
        t, state_array, gamma_star, e0_star
    )
    source_area = area_change.source_full(t, y, gamma_star_local, e0_star_local)

    # Project mass flux onto species fluxes
    source_area = np.reshape(source_area, area_change.shape_output)
    source_area = np.pad(source_area, ((0, 0), (0, physics.n_scalars - 1)), mode="edge")
    source_area[:, 2:] *= area_change.composition_frozen[area_change.idx_output]
    source_area = np.ravel(source_area)

    assert source_flux == pytest.approx(-source_area, rel=1e-3)
