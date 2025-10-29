from __future__ import annotations

import cantera as ct
import numpy as np

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.geometry import Geometry
from stanshock.utils.isentropic import mach_from_area_ratio, property_ratios


def smoothing_function(
    x: Array,
    xShock: float,
    Delta: float,
    phiLeft: float,
    phiRight: float,
) -> Array:
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


def smoothing_function_gradient(
    x: Array,
    xShock: float,
    Delta: float,
    phiLeft: float,
    phiRight: float,
) -> Array:
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


def initialize_constant(
    geometry: Geometry,
    physics: FluidPhysics,
    gas: ct.Solution,
    u: float,
) -> FluidState:
    """
    This helper function initializes a constant state
        inputs:
            gas = Cantera solution object at the desired thermodynamic state
            u = velocity
    """
    n = geometry.n_cells
    ones = np.ones(n)

    # Initialize state
    composition = physics.get_composition(gas.Y[None, :])

    return FluidState(
        shape=(n,),
        density=ones * gas.density,
        velocity=ones * u,
        pressure=ones * gas.P,
        gamma=ones * (gas.cp / gas.cv),
        composition=np.tile(composition, (n, 1)),
    )


def initialize_riemann_problem(
    geometry: Geometry,
    physics: FluidPhysics,
    left_state: tuple[ct.Solution, float],
    right_state: tuple[ct.Solution, float],
    shock_location: float,
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
    gas = physics.gas
    left_gas, u_left = left_state
    right_gas, u_right = right_state
    if (
        left_gas.species_names != gas.species_names
        or right_gas.species_names != gas.species_names
    ):
        msg = "Input gasses must be the same as the initialized gas."
        raise Exception(msg)

    # Initialize with left state
    state = initialize_constant(geometry, physics, left_gas, u_left)

    # Override with right state
    state_right = initialize_constant(geometry, physics, right_gas, u_right)

    index = np.where(geometry.xc >= shock_location)[0]
    state.density[index] = state_right.density[index]
    state.velocity[index] = state_right.velocity[index]
    state.pressure[index] = state_right.pressure[index]
    state.composition[index, :] = state_right.composition[index, :]
    state.gamma[index] = state_right.gamma[index]

    return state


def initialize_diffuse_interface(
    geometry: Geometry,
    physics: FluidPhysics,
    left_state: tuple[ct.Solution, float],
    right_state: tuple[ct.Solution, float],
    shock_location: float,
    delta_smoothing: float,
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
    gas = physics.gas
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
    composition_left = physics.get_composition(left_gas.Y)
    composition_right = physics.get_composition(right_gas.Y)

    # Smooth transition between left and right states
    r = smoothing_function(
        geometry.xc,
        shock_location,
        delta_smoothing,
        left_gas.density,
        right_gas.density,
    )
    geometry.u = smoothing_function(
        geometry.xc, shock_location, delta_smoothing, u_left, u_right
    )
    p = smoothing_function(
        geometry.xc, shock_location, delta_smoothing, left_gas.P, right_gas.P
    )
    gamma = smoothing_function(
        geometry.xc, shock_location, delta_smoothing, gamma_left, gamma_right
    )

    composition = np.zeros((geometry.n_cells, geometry.n_scalars))
    for kSp in range(geometry.n_scalars):
        composition[:, kSp] = smoothing_function(
            geometry.xc,
            shock_location,
            delta_smoothing,
            composition_left[:, kSp],
            composition_right[:, kSp],
        )

    return FluidState(
        shape=(geometry.n_cells,),
        density=r,
        pressure=p,
        gamma=gamma,
        composition=composition,
    )


def initialize_isentropic(
    geometry: Geometry,
    physics: FluidPhysics,
    inflow_state: ct.Solution,
    throat_area: float | None = None,
    subsonic_inflow: bool = True,
    subsonic_outflow: bool = False,
) -> FluidState:
    """
    Applies isentropic flow relations to set initial conditions.

    Note that this formulation is only valid for a flow with constant specific
    heat ratio. It could be extended for non-ideal gas equations of state
    """
    # Get cross-sectional area from the geometry
    x = geometry.x
    area = geometry.area(0.0, x)
    assert isinstance(area, np.ndarray)

    # If throat area is not given, assume choked flow
    min_area: float = area.min()
    throat_area = min_area if throat_area is None else min(min_area, throat_area)

    # Get inflow properties for the gas
    g = inflow_state.cp / inflow_state.cv
    P_in = inflow_state.P
    rho_in = inflow_state.density_mass
    composition = physics.get_composition(inflow_state.Y)

    # Get the permissible subsonic and supersonic Mach numbers throughout
    # the domain from the area ratio
    area_ratio = area / throat_area
    area_ratio_min = area_ratio.min()
    choked_flow = area_ratio_min <= 1.0
    if choked_flow:
        # Adjust throat area based on choked flow - will affect requested boundary conditions
        area_ratio /= area_ratio_min
        throat_area /= area_ratio_min
    elif subsonic_inflow != subsonic_outflow:
        print(
            "Warning: Subsonic/supersonic transition requested, but flow may not be choked.\n"
            + f"Minimum area ratio = {area_ratio_min}."
        )

    # Solve for allowable Mach numbers corresponding to given area ratio
    subsonic_mach: Array = mach_from_area_ratio(area_ratio, g, subsonic=True)
    supersonic_mach: Array = mach_from_area_ratio(area_ratio, g, subsonic=False)

    # Combine into one mach profile
    mach = subsonic_mach
    if subsonic_inflow != subsonic_outflow:
        idx = np.argmin(area_ratio)

        if subsonic_inflow:
            mach[idx + 1 :] = supersonic_mach[idx + 1 :]
        else:
            mach[:idx] = supersonic_mach[:idx]
    elif not subsonic_inflow and not subsonic_outflow:
        mach = supersonic_mach

    # Get stagnation properties based on inflow
    _, inflow_Pratio, inflow_rhoratio = property_ratios(mach[0], g)

    # Get properties throughout
    _, Pratio_profile, rhoratio_profile = property_ratios(mach, g)

    n = geometry.n
    state = FluidState(
        shape=(n,),
        pressure=P_in / inflow_Pratio * Pratio_profile,
        density=rho_in / inflow_rhoratio * rhoratio_profile,
        gamma=g * np.ones((n,)),
        composition=np.broadcast_to(composition, (n, composition.shape[0])).copy(),
    )
    state.velocity = mach * physics.get_sound_speed(state)

    return state


def initialize_isentropic_total(
    geometry: Geometry,
    physics: FluidPhysics,
    total_state: ct.Solution,
    inflow_mach: float | None = None,
    inflow_pressure: float | None = None,
    outflow_pressure: float | None = None,
    throat_area: float | None = None,
    subsonic_inflow: bool = True,
    subsonic_outflow: bool = True,
) -> FluidState:
    """
    Applies isentropic flow relations to set initial conditions.

    Note that this formulation is only valid for a flow with constant specific
    heat ratio. It could be extended for non-ideal gas equations of state
    """
    # Get cross-sectional area from the geometry
    x = geometry.x
    area = geometry.area(0.0, x)
    assert isinstance(area, np.ndarray)

    # Get stagnation properties for the gas
    g = total_state.cp / total_state.cv
    Pt = total_state.P
    Tt = total_state.T
    rhot = total_state.density_mass
    composition = physics.get_composition(total_state.Y)

    # Determine throat area from given inputs
    inputs_done = False
    too_many_inputs_prefix = "Too many inputs specified. "
    error_msg = "Must specify one of inflow_velocity, inflow_pressure, outflow_pressure, or throat area."

    if throat_area is not None:
        inputs_done = True

    if inflow_mach is not None:
        Tratio, _, _ = property_ratios(inflow_mach, g)
        inflow_area = area[0]
        throat_area = (
            inflow_area
            * inflow_mach
            * (2.0 * Tratio / (g + 1)) ** (0.5 * (g + 1) / (g - 1))
        )
        if inputs_done:
            raise ValueError(too_many_inputs_prefix + error_msg)
        inputs_done = True

    if inflow_pressure is not None:
        Pratio = inflow_pressure / Pt
        Tratio = Pratio ** ((g - 1.0) / g)
        inflow_mach = np.sqrt(2.0 * (1.0 / Tratio - 1.0) / (g - 1.0))
        inflow_area = area[0]
        throat_area = (
            inflow_area
            * inflow_mach
            * (2.0 * Tratio / (g + 1)) ** (0.5 * (g + 1) / (g - 1))
        )
        if inputs_done:
            raise ValueError(too_many_inputs_prefix + error_msg)
        inputs_done = True

    if outflow_pressure is not None:
        Pratio = outflow_pressure / Pt
        Tratio = Pratio ** ((g - 1.0) / g)
        outflow_mach = np.sqrt(2.0 * (1.0 / Tratio - 1.0) / (g - 1.0))
        outflow_area = area[-1]
        throat_area = (
            outflow_area
            * outflow_mach
            * (2.0 * Tratio / (g + 1)) ** (0.5 * (g + 1) / (g - 1))
        )
        if inputs_done:
            raise ValueError(too_many_inputs_prefix + error_msg)
        inputs_done = True

    if not inputs_done:
        raise ValueError(error_msg)

    # Get the permissible subsonic and supersonic Mach numbers throughout
    # the domain from the area ratio
    assert isinstance(throat_area, float)
    area_ratio = area / throat_area

    # Check if flow is choked
    area_ratio_min = area_ratio.min()
    choked_flow = area_ratio_min <= 1.0
    if choked_flow:
        # Adjust throat area based on choked flow - will affect requested boundary conditions
        area_ratio /= area_ratio_min
        throat_area /= area_ratio_min

    # Solve for allowable Mach numbers corresponding to given area ratio
    subsonic_mach: Array = mach_from_area_ratio(area_ratio, g, subsonic=True)
    supersonic_mach: Array = mach_from_area_ratio(area_ratio, g, subsonic=False)

    # Combine into one mach profile
    mach = subsonic_mach
    if subsonic_inflow != subsonic_outflow:
        if not choked_flow:
            msg = f"Subsonic-supersonic transition requested, but flow is not choked. {area_ratio_min = }"
            raise ValueError(msg)
        idx = np.argmin(area_ratio)

        if subsonic_inflow:
            mach[idx:] = supersonic_mach[idx:]
        else:
            mach[:idx] = supersonic_mach[:idx]
    elif not subsonic_inflow and not subsonic_outflow:
        mach = supersonic_mach

    # Get properties throughout
    Tratio_profile, Pratio_profile, rhoratio_profile = property_ratios(mach, g)

    n = geometry.n
    state = FluidState(
        shape=(n,),
        temperature=Tt * Tratio_profile,
        pressure=Pt * Pratio_profile,
        density=rhot * rhoratio_profile,
        gamma=g * np.ones((n,)),
        composition=np.broadcast_to(composition, (n, composition.shape[0])).copy(),
    )
    state.velocity = mach * physics.get_sound_speed(state)

    return state
