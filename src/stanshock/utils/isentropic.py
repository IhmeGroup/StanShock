from __future__ import annotations

from typing import overload

import numpy as np
from scipy.optimize import fsolve, minimize_scalar, root_scalar
from scipy.optimize.elementwise import find_root

from stanshock.system.backend import Array


@overload
def area_ratio_from_mach(mach: float, g: float) -> float: ...
@overload
def area_ratio_from_mach(mach: Array, g: float) -> Array: ...
def area_ratio_from_mach(mach: Array | float, g: float) -> Array | float:
    exponent = (g + 1) / (2 * (g - 1))
    return ((1 + 0.5 * (g - 1) * mach**2) / (0.5 * (g + 1))) ** exponent / np.maximum(
        mach, 1.0e-16
    )


def mach_from_area_ratio(area_ratio: Array, g: float, subsonic: bool = True) -> Array:
    """Find Mach number corresponding to given area ratio, element-wise."""

    def residual(mach: Array, g: float, area_ratio_target: Array | float) -> Array:
        return area_ratio_from_mach(mach, g) - area_ratio_target

    bracket = ([1e-5], [1.0]) if subsonic else ([1.0], [10.0])
    mach: Array = find_root(residual, bracket, args=(g, area_ratio)).x

    # Because the element-wise root solve tends to fail near Ma = 1,
    # perform a minimization to fix any NaN values:
    def residual2(mach: float, g: float, area_ratio_target: float) -> float:
        return np.abs(area_ratio_from_mach(mach, g) - area_ratio_target)

    bracket2 = (0.8, 1.0) if subsonic else (1.0, 1.2)
    for i in np.where(np.isnan(mach))[0]:
        mach[i] = minimize_scalar(
            residual2, args=(g, area_ratio[i]), bracket=bracket2
        ).x

    return mach


@overload
def property_ratios(mach: float, g: float) -> tuple[float, float, float]: ...
@overload
def property_ratios(mach: Array, g: Array | float) -> tuple[Array, Array, Array]: ...
def property_ratios(
    mach: Array | float, g: Array | float
) -> tuple[Array | float, Array | float, Array | float]:
    """Ratios of static temperature, pressure, and density to their total (stagnation) values."""
    temperature_ratio = 1.0 / (1.0 + 0.5 * (g - 1) * mach**2)
    pressure_ratio = temperature_ratio ** (g / (g - 1))
    density_ratio = temperature_ratio ** (1 / (g - 1))

    return temperature_ratio, pressure_ratio, density_ratio


def compute_ratios_across_oblique_shock(
    mach: float, gamma: float, theta: float
) -> tuple[float, float, float, float]:
    def f_beta(beta: Array) -> Array:
        return (
            (2 / np.tan(beta))
            * (
                (mach**2 * np.sin(beta) ** 2 - 1)
                / (mach**2 * (gamma + np.cos(2 * beta)) + 2)
            )
        ) - np.tan(theta)

    beta_sol = fsolve(f_beta, x0=np.pi / 6.0)[0]

    Mn0 = mach * np.sin(beta_sol)

    Mn1 = np.sqrt(
        (1 + ((gamma - 1) / 2) * Mn0**2) / ((gamma * Mn0**2) - ((gamma - 1) / 2))
    )
    M1 = Mn1 / (np.sin(beta_sol - theta))

    density_ratio = ((gamma + 1) * Mn0**2) / (2 + (gamma - 1) * Mn0**2)

    pressure_ratio = 1 + ((2 * gamma) / (gamma + 1)) * (Mn0**2 - 1)

    temperature_ratio = pressure_ratio * (1 / density_ratio)

    return M1, density_ratio, pressure_ratio, temperature_ratio


# Functions for compressible isentropic fuel injectors
def mach_from_pressure_ratio(pr: float, g: float) -> float:
    """Compute Mach number from pressure ratio, p/p0."""
    choked: bool = pr < (2.0 / (g + 1)) ** (g / (g - 1))
    if choked:
        # Exit pressure is no longer equal to the ambient pressure
        # We know based on geometry that the Mach number at the orifice is 1
        return 1.0

    # Exit pressure is equal to the ambient pressure
    return float(np.sqrt(2 / (g - 1) * (pr ** ((1 - g) / g) - 1)))


def mdot_from_pressure_ratio(
    p0: float, T0: float, R0: float, g: float, pa: float, Ae: float
) -> float:
    pr = pa / p0
    choked: bool = pr < (2.0 / (g + 1)) ** (g / (g - 1))

    tmp: float = Ae * p0 / np.sqrt(R0 * T0)
    if choked:
        return float(tmp * np.sqrt(g) * (2.0 / (g + 1)) ** ((g + 1) / (2 * (g - 1))))
    return float(
        tmp
        * pr ** (1.0 / g)
        * np.sqrt((2.0 * g / (g - 1.0)) * (1.0 - pr ** ((g - 1.0) / g)))
    )


def p0_from_mdot(
    mdot: float, T0: float, R0: float, g: float, pa: float, Ae: float
) -> tuple[float, float]:
    """Compute the manifold pressure which produces a given mass flow rate."""

    def residual(p0: float) -> float:
        return mdot_from_pressure_ratio(p0, T0, R0, g, pa, Ae) - mdot

    result = root_scalar(residual, x0=pa)
    p0 = result.root
    mach = mach_from_pressure_ratio(pa / p0, g)
    return p0, mach
