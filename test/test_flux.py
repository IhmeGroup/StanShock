import unittest

import numpy as np

from StanShock.stanShock import LF, HLLC

LF = LF.__wrapped__  # unwrap for coverage testing
HLLC = HLLC.__wrapped__


class TestFlux(unittest.TestCase):
    def test_lax_friedrich_predicts_constant_flux(self):
        r = 5.0
        u = -2.0
        p = 3.0
        Y = 1.0
        gamma = 1.2
        flux = LF(
            rLR=np.array([[r], [r]]),
            uLR=np.array([[u], [u]]),
            pLR=np.array([[p], [p]]),
            YLR=np.array([[[Y]], [[Y]]]),
            gamma=np.array([gamma])
        )[0]
        H = gamma*p/(gamma-1.0) + 0.5*r*u**2.0
        expected_flux = np.array([r*u, r*u**2+p, H*u, r*Y*u])
        self.assertTrue(np.allclose(flux, expected_flux))

    def test_hllc_predicts_constant_flux(self):
        r = 2.0
        u = 0.5
        p = 3.0
        Y = 1.0
        gamma = 1.2
        num_faces = 10
        num_sides = 2
        num_species = 1
        flux = HLLC(
            rLR=r * np.ones((num_sides, num_faces)),
            uLR=u * np.ones((num_sides, num_faces)),
            pLR=p * np.ones((num_sides, num_faces)),
            YLR=Y * np.ones((num_sides, num_faces, num_species)),
            gamma=gamma * np.ones(num_faces)
        )
        H = gamma*p/(gamma-1.0) + 0.5*r*u**2.0
        expected_flux = np.array([r*u, r*u**2+p, H*u, r*Y*u])[np.newaxis, ...]
        expected_flux = np.repeat(expected_flux, num_faces, axis=0)
        self.assertTrue(np.allclose(flux, expected_flux))


if __name__ == '__main__':
    unittest.main()
