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
        gamma=np.array([gamma]),
        e0=np.array([e0]),
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
    flux = hllc_flux(
        rLR=r * np.ones((num_sides, num_faces)),
        uLR=u * np.ones((num_sides, num_faces)),
        pLR=p * np.ones((num_sides, num_faces)),
        YLR=Y * np.ones((num_sides, num_faces, num_species)),
        gamma=gamma * np.ones(num_faces),
        e0=e0 * np.ones(num_faces),
    )
    H = gamma * p / (gamma - 1.0) + r * (e0 + 0.5 * u**2.0)
    expected_flux = np.array([r * u**2 + p, H * u, r * Y * u])[np.newaxis, ...]
    expected_flux = np.repeat(expected_flux, num_faces, axis=0)
    assert np.allclose(flux, expected_flux)


def test_isentropic_flow_relations(isentropic_flow: Combustor) -> None:
    # Get initial state from given solution
    t = isentropic_flow.t
    state = isentropic_flow.initialization()
    state_array = np.ravel(isentropic_flow.physics.primitive_to_conservative(state))
    gamma_star, e0_star = isentropic_flow.physics.get_double_flux_variables(state)

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
    assert isentropic_flow.area_change is not None
    y, gamma_star_local, e0_star_local = (
        isentropic_flow.area_change.before_time_integration(
            t, state_array, gamma_star, e0_star
        )
    )
    source_area = isentropic_flow.area_change.source_full(
        t,
        y,
        gamma_star_local,
        e0_star_local,
    )
    assert source_flux == pytest.approx(-source_area, rel=1e-3)
