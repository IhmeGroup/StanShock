import unittest

import numpy as np

from StanShock.stanShock import WENO5, mt

WENO5 = WENO5.__wrapped__  # unwrap for test coverage


class TestWENO(unittest.TestCase):
    def test_weno_interpolates_constant(self):
        num_interior_cells = 10
        num_ghost_cells = 2 * mt
        num_cells = num_interior_cells + num_ghost_cells
        num_species = 2
        r = 1.0
        u = 2.0
        p = 3.0
        Y = 1.0 / num_species
        gamma = 5.0 / 3.0
        face_values = WENO5(
            r=r * np.ones(num_cells),
            u=u * np.ones(num_cells),
            p=p * np.ones(num_cells),
            Y=Y * np.ones((num_cells, num_species)),
            gamma=gamma * np.ones(num_cells),
        )
        num_sides = 2
        num_faces = num_interior_cells + 1
        num_primitives = 3 + num_species
        expected_face_values = np.ones((num_sides, num_faces, num_primitives))
        constant_state = np.array([r, u, p] + num_species*[Y])
        expected_face_values *= constant_state[np.newaxis, np.newaxis, :]
        self.assertTrue(np.allclose(expected_face_values, face_values))

    def test_weno_interpolates_discontinuity(self):
        num_interior_cells = 2 * 5
        num_ghost_cells = 2 * mt
        num_cells = num_interior_cells + num_ghost_cells
        num_species = 2
        r = (0.5, 2.0)
        u = (-1.0, 1.0)
        p = (1e6, 1.)
        Y = (
            (1.0, 0.0),
            (0.0, 1.0)
        )
        gamma = 1.4
        face_values = WENO5(
            r=np.concatenate([r[0] * np.ones(num_cells // 2), r[1] * np.ones(num_cells // 2)]),
            u=np.concatenate([u[0] * np.ones(num_cells // 2), u[1] * np.ones(num_cells // 2)]),
            p=np.concatenate([p[0] * np.ones(num_cells // 2), p[1] * np.ones(num_cells // 2)]),
            Y=np.concatenate([
                np.ones((num_cells // 2, num_species)) * np.array(Y[0])[np.newaxis, :],
                np.ones((num_cells // 2, num_species)) * np.array(Y[1])[np.newaxis, :],
            ]),
            gamma=gamma * np.ones(num_cells),
        )
        num_sides = 2
        num_faces = num_interior_cells + 1
        num_primitives = 3 + num_species
        expected_face_values = np.ones((num_sides, num_faces, num_primitives))
        left_state = np.array([r[0], u[0], p[0], Y[0][0], Y[0][1]])
        right_state = np.array([r[1], u[1], p[1], Y[1][0], Y[1][1]])
        expected_face_values[:, :num_faces//2, :] = left_state[np.newaxis, np.newaxis, :]
        expected_face_values[:, num_faces//2:, :] = right_state[np.newaxis, np.newaxis, :]
        expected_face_values[0, num_faces//2, :] = left_state[np.newaxis, np.newaxis, :]
        self.assertTrue(np.allclose(expected_face_values, face_values))


if __name__ == '__main__':
    unittest.main()
