from __future__ import annotations

import numpy as np


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


def initialize_constant(domain, state, x):
    """
    This helper function initializes a constant state
        inputs:
            state = a tuple containing the Cantera solution object at the
                    the desired thermodynamic state and the velocity:
                    (canteraSolution,u)
            x = the grid for the problem
    """
    # Initialize grid
    domain.n = len(x)
    domain.x = x
    domain.dx = domain.x[1] - domain.x[0]
    # Initialize state
    domain.r = np.ones(domain.n) * state[0].density
    domain.u = np.ones(domain.n) * state[1]
    domain.p = np.ones(domain.n) * state[0].P
    domain.Y = np.zeros((domain.n, domain.n_scalars))
    if domain.physics == "FPV":
        domain.Y[:, 0] = domain.get_bilger_mixture_fraction(state[0].Y)
        domain.Y[:, 1] = domain.get_progress_variable(state[0].Y)
    elif domain.physics == "FRC":
        for k in range(domain.n_scalars):
            domain.Y[:, k] = state[0].Y[k]
    domain.gamma = np.ones(domain.n) * (state[0].cp / state[0].cv)
    # No flame thickening
    domain.F = np.ones_like(domain.r)


def initialize_riemann_problem(domain, leftState, rightState, geometry):
    """
    This helper function initializes a Riemann Problem
        inputs:
            leftState = a tuple containing the Cantera solution object at the
                        the desired thermodynamic state and the velocity:
                        (canteraSolution,u)
            rightState = a tuple containing the Cantera solution object at the
                        the desired thermodynamic state and the velocity:
                        (canteraSolution,u)
            geometry = a tuple containing the relevant geometry for the
                        problem: (numberCells,xMinimum,xMaximum,shockLocation)
    """
    gas = domain.gas
    if (
        leftState[0].species_names != gas.species_names
        or rightState[0].species_names != gas.species_names
    ):
        msg = "Input gasses must be the same as the initialized gas."
        raise Exception(msg)
    domain.n = geometry[0]
    domain.x = np.linspace(geometry[1], geometry[2], domain.n)
    domain.dx = domain.x[1] - domain.x[0]
    # initialization for left state
    domain.r = np.ones(domain.n) * leftState[0].density
    domain.u = np.ones(domain.n) * leftState[1]
    domain.p = np.ones(domain.n) * leftState[0].P
    domain.Y = np.zeros((domain.n, domain.n_scalars))
    if domain.physics == "FPV":
        domain.Y[:, 0] = domain.get_bilger_mixture_fraction(leftState[0].Y)
        domain.Y[:, 1] = domain.get_progress_variable(leftState[0].Y)
    elif domain.physics == "FRC":
        for kSp in range(domain.n_scalars):
            domain.Y[:, kSp] = leftState[0].Y[kSp]
    domain.gamma = np.ones(domain.n) * (leftState[0].cp / leftState[0].cv)
    # right state
    index = domain.x >= geometry[3]
    domain.r[index] = rightState[0].density
    domain.u[index] = rightState[1]
    domain.p[index] = rightState[0].P
    if domain.physics == "FPV":
        domain.Y[index, 0] = domain.get_bilger_mixture_fraction(rightState[0].Y)
        domain.Y[index, 1] = domain.get_progress_variable(rightState[0].Y)
    elif domain.physics == "FRC":
        for kSp in range(domain.n_scalars):
            domain.Y[index, kSp] = rightState[0].Y[kSp]
    domain.gamma[index] = rightState[0].cp / rightState[0].cv
    domain.F = np.ones_like(domain.r)


def initialize_diffuse_interface(domain, leftState, rightState, geometry, Delta):
    """
    This helper function initializes an interface smoothed over a distance
        inputs:
            leftState = a tuple containing the Cantera solution object at the
                        the desired thermodynamic state and the velocity:
                        (canteraSolution,u)
            rightState =  a tuple containing the Cantera solution object at the
                        the desired thermodynamic state and the velocity:
                        (canteraSolution,u)
            geometry = a tuple containing the relevant geometry for the
                        problem: (numberCells,xMinimum,xMaximum,shockLocation)
            Delta =    distance over which the interface is smoothed linearly
    """
    gas = domain.gas
    if (
        leftState[0].species_names != gas.species_names
        or rightState[0].species_names != gas.species_names
    ):
        msg = "Input gasses must be the same as the initialized gas."
        raise Exception(msg)
    domain.n = geometry[0]
    domain.x = np.linspace(geometry[1], geometry[2], domain.n)
    domain.dx = domain.x[1] - domain.x[0]
    xShock = geometry[3]
    leftGas = leftState[0]
    uLeft = leftState[1]
    gammaLeft = leftGas.cp / leftGas.cv
    rightGas = rightState[0]
    uRight = rightState[1]
    gammaRight = rightGas.cp / rightGas.cv
    # initialization for left state
    domain.r = smoothing_function(
        domain.x, xShock, Delta, leftGas.density, rightGas.density
    )
    domain.u = smoothing_function(domain.x, xShock, Delta, uLeft, uRight)
    domain.p = smoothing_function(domain.x, xShock, Delta, leftGas.P, rightGas.P)
    domain.Y = np.zeros((domain.n, domain.n_scalars))
    if domain.physics == "FPV":
        domain.Y[:, 0] = smoothing_function(
            domain.x,
            xShock,
            Delta,
            domain.get_bilger_mixture_fraction(leftGas.Y),
            domain.get_bilger_mixture_fraction(rightGas.Y),
        )
        domain.Y[:, 1] = smoothing_function(
            domain.x, xShock, Delta, domain.get_progress_variable(leftGas.Y), domain.get_get_progress_variableress_variable(rightGas.Y)
        )
    elif domain.physics == "FRC":
        for kSp in range(domain.n_scalars):
            domain.Y[:, kSp] = smoothing_function(
                domain.x, xShock, Delta, leftGas.Y[kSp], rightGas.Y[kSp]
            )
    domain.gamma = smoothing_function(domain.x, xShock, Delta, gammaLeft, gammaRight)
    domain.F = np.ones_like(domain.r)
