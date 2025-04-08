from __future__ import annotations

import numpy as np

from stanshock.physics.fluid_base import FluidState


def smoothing_function(x, xShock, Delta, phiLeft, phiRight):
    """
    This helper function returns the function of the variable smoothed
    over the interface
        inputs:
            x = numpy array of cell centers
            phiLeft = the value of the variable on the left side
            phiRight = the value of the variable on the right side
            xShock = the mean of the shock location
    """
    dphidx = (phiRight - phiLeft) / Delta
    phi = (phiLeft + phiRight) / 2.0 + dphidx * (x - xShock)
    phi[x < (xShock - Delta / 2.0)] = phiLeft
    phi[x > (xShock + Delta / 2.0)] = phiRight
    return phi


def smoothing_function_gradient(x, xShock, Delta, phiLeft, phiRight):
    """
    This helper function returns the derivative of the smoothing function
        inputs:
            x = numpy array of cell centers
            phiLeft = the value of the variable on the left side
            phiRight = the value of the variable on the right side
            xShock = the mean of the shock location
    """
    dphidx = (phiRight - phiLeft) / Delta
    dphidx = np.ones(len(x)) * dphidx
    dphidx[x < (xShock - Delta / 2.0)] = 0.0
    dphidx[x > (xShock + Delta / 2.0)] = 0.0
    return dphidx


def initialize_constant(domain, gas, u) -> FluidState:
    """
    This helper function initializes a constant state
        inputs:
            gas = Cantera solution object at the desired thermodynamic state
            u = velocity
    """
    n = domain.n
    ones = np.ones(n)

    # Initialize state
    composition = domain.physics.get_composition(gas.Y)

    return FluidState(
        shape=n,
        density=ones * gas.density,
        pressure=ones * gas.P,
        gamma=ones * (gas.cp / gas.cv),
        velocity=ones * u,
        composition=np.tile(composition, (n, 1)),
    )


def initialize_riemann_problem(
    domain, left_state, right_state, shock_location
) -> FluidState:
    """
    This helper function initializes a Riemann Problem
        inputs:
            left_state = a tuple containing the Cantera solution object at the
                         the desired thermodynamic state and the velocity:
                         (canteraSolution,u)
            right_state = a tuple containing the Cantera solution object at the
                          the desired thermodynamic state and the velocity:
                          (canteraSolution,u)
            shock_location = x-location of the shock within the domain
    """
    gas = domain.physics.gas
    left_gas, u_left = left_state
    right_gas, u_right = right_state
    if (
        left_gas.species_names != gas.species_names
        or right_gas.species_names != gas.species_names
    ):
        msg = "Input gasses must be the same as the initialized gas."
        raise Exception(msg)

    # Initialize with left state
    state = initialize_constant(domain, left_gas, u_left)

    # Override with right state
    state_right = initialize_constant(domain, right_gas, u_right)

    index = np.where(domain.x >= shock_location)[0]
    state.density[index] = state_right.density[index]
    state.velocity[index] = state_right.velocity[index]
    state.pressure[index] = state_right.pressure[index]
    state.composition[index, :] = state_right.composition[index, :]
    state.gamma[index] = state_right.gamma[index]

    return state


def initialize_diffuse_interface(
    domain, left_state, right_state, shock_location, delta_smoothing
) -> FluidState:
    """
    This helper function initializes an interface smoothed over a distance
        inputs:
            left_state = a tuple containing the Cantera solution object at the
                         the desired thermodynamic state and the velocity:
                         (canteraSolution,u)
            right_state = a tuple containing the Cantera solution object at the
                          the desired thermodynamic state and the velocity:
                          (canteraSolution,u)
            shock_location = x-location of the shock within the domain
            delta_smoothing = distance over which the interface is smoothed linearly
    """
    gas = domain.physics.gas
    left_gas, u_left = left_state
    right_gas, u_right = right_state
    if (
        left_gas.species_names != gas.species_names
        or right_gas.species_names != gas.species_names
    ):
        msg = "Input gasses must be the same as the initialized gas."
        raise Exception(msg)

    gamma_left = left_gas.cp / left_gas.cv
    gamma_right = right_gas.cp / right_gas.cv
    composition_left = domain.physics.get_composition(left_gas.Y)
    composition_right = domain.physics.get_composition(right_gas.Y)

    # Smooth transition between left and right states
    r = smoothing_function(
        domain.x, shock_location, delta_smoothing, left_gas.density, right_gas.density
    )
    domain.u = smoothing_function(
        domain.x, shock_location, delta_smoothing, u_left, u_right
    )
    p = smoothing_function(
        domain.x, shock_location, delta_smoothing, left_gas.P, right_gas.P
    )
    gamma = smoothing_function(
        domain.x, shock_location, delta_smoothing, gamma_left, gamma_right
    )

    composition = np.zeros((domain.n, domain.n_scalars))
    for kSp in range(domain.n_scalars):
        composition[:, kSp] = smoothing_function(
            domain.x,
            shock_location,
            delta_smoothing,
            composition_left[:, kSp],
            composition_right[:, kSp],
        )

    return FluidState(
        shape=domain.n,
        density=r,
        pressure=p,
        gamma=gamma,
        composition=composition,
    )
