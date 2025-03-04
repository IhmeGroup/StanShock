from __future__ import annotations

import numpy as np

from stanshock.numerics.inviscid_flux import hllc_flux, lax_friedrichs_flux

lax_friedrichs_flux = lax_friedrichs_flux.__wrapped__  # unwrap for coverage testing
hllc_flux = hllc_flux.__wrapped__


def test_lax_friedrich_predicts_constant_flux():
    r = 5.0
    u = -2.0
    p = 3.0
    Y = 1.0
    gamma = 1.2
    flux = lax_friedrichs_flux(
        rLR=np.array([[r], [r]]),
        uLR=np.array([[u], [u]]),
        pLR=np.array([[p], [p]]),
        YLR=np.array([[[Y]], [[Y]]]),
        gamma=np.array([gamma]),
    )[0]
    H = gamma * p / (gamma - 1.0) + 0.5 * r * u**2.0
    expected_flux = np.array([r * u, r * u**2 + p, H * u, r * Y * u])
    assert np.allclose(flux, expected_flux)


def test_hllc_predicts_constant_flux():
    r = 2.0
    u = 0.5
    p = 3.0
    Y = 1.0
    gamma = 1.2
    num_faces = 10
    num_sides = 2
    num_species = 1
    flux = hllc_flux(
        rLR=r * np.ones((num_sides, num_faces)),
        uLR=u * np.ones((num_sides, num_faces)),
        pLR=p * np.ones((num_sides, num_faces)),
        YLR=Y * np.ones((num_sides, num_faces, num_species)),
        gamma=gamma * np.ones(num_faces),
    )
    H = gamma * p / (gamma - 1.0) + 0.5 * r * u**2.0
    expected_flux = np.array([r * u, r * u**2 + p, H * u, r * Y * u])[np.newaxis, ...]
    expected_flux = np.repeat(expected_flux, num_faces, axis=0)
    assert np.allclose(flux, expected_flux)
