from __future__ import annotations

from typing import overload

import numpy as np
from scipy.optimize import fsolve, minimize_scalar
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
