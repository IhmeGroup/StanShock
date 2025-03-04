from __future__ import annotations

import cantera as ct
import numpy as np
import pytest

from stanshock.physics.thermo.table import (
    ThermoTable,
    get_cp_compiled,
    get_specific_gas_constants_compiled,
)

get_specific_gas_constants_compiled = (
    get_specific_gas_constants_compiled.__wrapped__
)  # unwrap for coverage
# get_cp_compiled = get_cp_compiled.__wrapped__


mech = "tests/resources/HeliumArgon.yaml"


def test_table_computes_correct_temperatures():
    gas = ct.Solution(mech)
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

    table = ThermoTable(gas)
    predicted_temperatures = table.get_temperature(densities, pressures, mass_fractions)
    assert np.allclose(actual_temperatures, predicted_temperatures)


def test_monatomic_gas_has_constant_gamma():
    gas = ct.Solution(mech)
    temperatures = np.linspace(50.0, 5000.0)[:, np.newaxis]
    mass_fractions = np.hstack(
        [np.ones_like(temperatures), np.zeros_like(temperatures)]
    )
    table = ThermoTable(gas)
    gammas = table.get_gamma(temperatures[:, 0], mass_fractions)
    gammas_are_constant = np.allclose(gammas, gammas[0])
    assert gammas_are_constant


def test_single_species_gas_has_correct_constant():
    molecular_weight = np.array([7.0, 3.0])
    mass_fraction = np.array([1, 0])[np.newaxis, :]
    actual_gas_constant = ct.gas_constant / molecular_weight[0]
    predicted_gas_constant = get_specific_gas_constants_compiled(
        mass_fraction, molecular_weight
    )[0]
    assert actual_gas_constant == predicted_gas_constant


def test_cp_increases_with_larger_coefficients():
    temperatures = np.linspace(300, 3000)
    temperature_table = temperatures
    mass_fractions = np.ones_like(temperatures)[:, np.newaxis]
    a = np.ones_like(mass_fractions)
    b = np.ones_like(mass_fractions)
    specific_heats_with_small_a = get_cp_compiled(
        temperatures, mass_fractions, temperature_table, a, b
    )
    specific_heats_with_large_a = get_cp_compiled(
        temperatures, mass_fractions, temperature_table, 10 * a, b
    )
    assert np.all(specific_heats_with_small_a <= specific_heats_with_large_a)
    specific_heats_with_small_b = get_cp_compiled(
        temperatures, mass_fractions, temperature_table, a, b
    )
    specific_heats_with_large_b = get_cp_compiled(
        temperatures, mass_fractions, temperature_table, a, 10 * b
    )
    assert np.all(specific_heats_with_small_b <= specific_heats_with_large_b)


def test_formation_enthalpy_is_invariant_to_temperature():
    temperatures = np.array([1000.0, 5000.0])
    gas = ct.Solution(mech)
    table = ThermoTable(gas)
    mass_fractions = np.hstack(
        [
            np.ones_like(temperatures)[:, np.newaxis],
            np.zeros_like(temperatures)[:, np.newaxis],
        ]
    )
    enthalpies = table.get_frozen_enthalpy(temperatures, mass_fractions)
    assert enthalpies[0] == pytest.approx(enthalpies[1], rel=1e-3)


def test_out_of_bounds_temperature_raises_exception():
    temperatures = np.array([-100, -90])
    temperature_table = temperatures + 100
    mass_fractions = np.ones_like(temperatures)[:, np.newaxis]
    a = np.ones_like(mass_fractions)
    b = np.ones_like(mass_fractions)
    with pytest.raises(ValueError, match="Temperature out of bounds"):
        get_cp_compiled(temperatures, mass_fractions, temperature_table, a, b)

    gas = ct.Solution(mech)
    table = ThermoTable(gas)
    with pytest.raises(ValueError, match="Temperature not within table"):
        table.get_frozen_enthalpy(temperatures, mass_fractions)
