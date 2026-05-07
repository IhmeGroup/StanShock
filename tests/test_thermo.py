from __future__ import annotations

import cantera as ct
import numpy as np
import pytest

from stanshock.physics.fluid_base import FluidState
from stanshock.physics.thermotable import (
    ThermoTable,
    get_cp_compiled,
    get_specific_gas_constant_compiled,
)

get_specific_gas_constant_compiled = (
    get_specific_gas_constant_compiled.__wrapped__
)  # unwrap for coverage
get_cp_compiled = get_cp_compiled.__wrapped__


mech = "data/mechanisms/HeliumArgon.yaml"


def test_table_computes_correct_temperatures() -> None:
    N = 1001
    gas = ct.Solution(mech)
    sol = ct.SolutionArray(gas, (N,))
    argon_mass_fractions = np.linspace(0, 1, N)[:, None]
    helium_mass_fractions = 1.0 - argon_mass_fractions
    mass_fractions = np.hstack([argon_mass_fractions, helium_mass_fractions])
    temperatures = np.logspace(np.log10(gas.max_temp), np.log10(gas.min_temp), N)
    pressures = np.logspace(6, 4, N)

    sol.TPY = temperatures, pressures, mass_fractions
    densities = sol.density_mass

    table = ThermoTable(gas)

    # Ideal gas law
    predicted_temperatures = table.get_temperature(
        FluidState(
            shape=(N,),
            density=densities,
            pressure=pressures,
            composition=mass_fractions,
        )
    )
    assert temperatures == pytest.approx(predicted_temperatures)

    # Energy to temperature
    predicted_temperatures = table.get_temperature(
        FluidState(
            shape=(N,),
            internal_energy=sol.int_energy_mass,
            composition=mass_fractions,
        )
    )
    assert temperatures == pytest.approx(predicted_temperatures, rel=0.013)


def test_monatomic_gas_has_constant_gamma() -> None:
    gas = ct.Solution(mech)
    temperatures = np.linspace(gas.min_temp, gas.max_temp)[:, None]
    mass_fractions = np.hstack(
        [np.ones_like(temperatures), np.zeros_like(temperatures)]
    )
    table = ThermoTable(gas)
    gammas = table.get_gamma(
        FluidState(
            shape=temperatures.shape[0],
            temperature=temperatures[:, 0],
            composition=mass_fractions,
        )
    )
    assert gammas == pytest.approx(gammas[0])


def test_single_species_gas_has_correct_constant() -> None:
    molecular_weight = np.array([7.0, 3.0])
    mass_fraction = np.array([1, 0])[None, :]
    actual_gas_constant = ct.gas_constant / molecular_weight[0]
    predicted_gas_constant = get_specific_gas_constant_compiled(
        mass_fraction, molecular_weight
    )[0]
    assert actual_gas_constant == predicted_gas_constant


def test_cp_increases_with_larger_coefficients() -> None:
    temperatures = np.linspace(300, 3000)
    temperature_table = temperatures
    mass_fractions = np.ones_like(temperatures)[:, None]
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


def test_out_of_bounds_temperature_raises_exception():
    temperatures = np.array([-100, -90])
    temperature_table = temperatures + 100
    mass_fractions = np.ones_like(temperatures)[:, None]
    a = np.ones_like(mass_fractions)
    b = np.ones_like(mass_fractions)
    with pytest.raises(ValueError, match="Temperature out of bounds"):
        get_cp_compiled(temperatures, mass_fractions, temperature_table, a, b)

    gas = ct.Solution(mech)
    table = ThermoTable(gas)
    with pytest.raises(ValueError, match="Temperature not within table"):
        table.get_species_enthalpies(FluidState(shape=(2,), temperature=temperatures))
