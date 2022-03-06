import unittest

import cantera as ct
import numpy as np

from StanShock.stanShock import thermoTable, getR, getCp

getR = getR.__wrapped__  # unwrap for coverage
getCp = getCp.__wrapped__


class TestThermo(unittest.TestCase):
    _mech = "test/resources/HeliumArgon.xml"

    def test_table_computes_correct_temperatures(self):
        gas = ct.Solution(self._mech)
        argon_mass_fractions = np.linspace(0, 1)[:, np.newaxis]
        helium_mass_fractions = 1.0 - argon_mass_fractions
        mass_fractions = np.hstack([argon_mass_fractions, helium_mass_fractions])
        densities = np.logspace(-1, 1)
        pressures = np.logspace(6, 4)
        actual_temperatures = []
        for state in zip(densities, pressures, mass_fractions):
            gas.DPY = state
            actual_temperatures.append(gas.T)
        actual_temperatures = np.array(actual_temperatures)

        table = thermoTable(gas)
        predicted_temperatures = table.getTemperature(densities, pressures, mass_fractions)
        self.assertTrue(np.allclose(actual_temperatures, predicted_temperatures))

    def test_monatomic_gas_has_constant_gamma(self):
        gas = ct.Solution(self._mech)
        temperatures = np.linspace(50.0, 5000.0)[:, np.newaxis]
        mass_fractions = np.hstack([np.ones_like(temperatures), np.zeros_like(temperatures)])
        table = thermoTable(gas)
        gammas = table.getGamma(temperatures[:, 0], mass_fractions)
        gammas_are_constant = np.allclose(gammas, gammas[0])
        self.assertTrue(gammas_are_constant)

    def test_single_species_gas_has_correct_constant(self):
        molecular_weight = np.array([7.0, 3.0])
        mass_fraction = np.array([1, 0])[np.newaxis, :]
        actual_gas_constant = ct.gas_constant / molecular_weight[0]
        predicted_gas_constant = getR(mass_fraction, molecular_weight)[0]
        self.assertEqual(actual_gas_constant, predicted_gas_constant)

    def test_cp_increases_with_larger_coefficients(self):
        temperatures = np.linspace(300, 3000)
        temperature_table = temperatures
        mass_fractions = np.ones_like(temperatures)[:, np.newaxis]
        a = np.ones_like(mass_fractions)
        b = np.ones_like(mass_fractions)
        specific_heats_with_small_a = getCp(temperatures, mass_fractions, temperature_table, a, b)
        specific_heats_with_large_a = getCp(temperatures, mass_fractions, temperature_table, 10 * a, b)
        self.assertTrue(np.all(specific_heats_with_small_a <= specific_heats_with_large_a))
        specific_heats_with_small_b = getCp(temperatures, mass_fractions, temperature_table, a, b)
        specific_heats_with_large_b = getCp(temperatures, mass_fractions, temperature_table, a, 10 * b)
        self.assertTrue(np.all(specific_heats_with_small_b <= specific_heats_with_large_b))

    def test_formation_enthalpy_is_invariant_to_temperature(self):
        temperatures = np.array([1000.0, 5000.0])
        gas = ct.Solution(self._mech)
        table = thermoTable(gas)
        mass_fractions = np.hstack([
            np.ones_like(temperatures)[:, np.newaxis],
            np.zeros_like(temperatures)[:, np.newaxis]
        ])
        enthalpies = table.getH0(temperatures, mass_fractions)
        self.assertAlmostEqual(enthalpies[0], enthalpies[1], 3)

    def test_out_of_bounds_temperature_raises_exception(self):
        temperatures = np.array([-100])
        temperature_table = temperatures + 100
        mass_fractions = np.ones_like(temperatures)[:, np.newaxis]
        a = np.ones_like(mass_fractions)
        b = np.ones_like(mass_fractions)
        with self.assertRaises(Exception):
            getCp(temperatures, mass_fractions, temperature_table, a, b)

        gas = ct.Solution(self._mech)
        table = thermoTable(gas)
        with self.assertRaises(Exception):
            table.getH0(temperatures, mass_fractions)


if __name__ == '__main__':
    unittest.main()
