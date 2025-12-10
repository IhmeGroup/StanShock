from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from cantera import Solution

from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.processing.initialize import (
    InitializeConstant,
    InitializeDiffuseInterface,
    InitializeIsentropic,
    InitializeRiemannProblem,
)
from stanshock.system.geometry import Geometry, initialize_geometry
from stanshock.utils.isentropic import area_ratio_from_mach


@pytest.fixture
def const_geometry() -> Geometry:
    xf = np.linspace(0.0, 10.0, 51, dtype=np.float64)

    return initialize_geometry(xf=xf, area=5.0)


@pytest.fixture
def left_state(mechanism: Path) -> tuple[Solution, float]:
    gas = Solution(mechanism)
    gas.TP = 300.0, 10132.50
    return (gas, 10.0)


@pytest.fixture
def right_state(mechanism: Path) -> tuple[Solution, float]:
    gas = Solution(mechanism)
    gas.TP = 1000.0, 101325.0
    return (gas, 1.0)


@pytest.fixture
def physics(left_state: tuple[Solution, float]) -> CanteraInterface:
    return CanteraInterface(left_state[0])


def test_constant_init(
    gas: Solution, const_geometry: Geometry, physics: CanteraInterface
) -> None:
    state = InitializeConstant(const_geometry, physics, gas, 10.0)()

    assert state.density is not None
    assert np.all(state.density == gas.density_mass)


def test_riemann_init(
    left_state: tuple[Solution, float],
    right_state: tuple[Solution, float],
    const_geometry: Geometry,
    physics: CanteraInterface,
) -> None:
    x_midpoint = 0.5 * (const_geometry.xf[0] + const_geometry.xf[-1])
    state = InitializeRiemannProblem(
        const_geometry, physics, left_state, right_state, x_midpoint
    )()

    assert state.density is not None
    idx_left = np.where(const_geometry.xc < x_midpoint)[0]
    assert np.all(state.density[idx_left] == left_state[0].density_mass)
    idx_right = np.where(const_geometry.xc > x_midpoint)[0]
    assert np.all(state.density[idx_right] == right_state[0].density_mass)


def test_diffuse_interface_init(
    left_state: tuple[Solution, float],
    right_state: tuple[Solution, float],
    const_geometry: Geometry,
    physics: CanteraInterface,
) -> None:
    x_midpoint = 0.5 * (const_geometry.xf[0] + const_geometry.xf[-1])
    dx_smooth = 0.05 * (const_geometry.xf[-1] - const_geometry.xf[0])
    state = InitializeDiffuseInterface(
        const_geometry, physics, left_state, right_state, x_midpoint, dx_smooth
    )()

    assert state.density is not None

    x_left = x_midpoint - 0.5 * dx_smooth
    rho_left = left_state[0].density_mass
    idx_left = np.where(const_geometry.xc < x_left)[0]
    assert np.all(state.density[idx_left] == rho_left)

    x_right = x_midpoint + 0.5 * dx_smooth
    rho_right = right_state[0].density_mass
    idx_right = np.where(const_geometry.xc > x_right)[0]
    assert np.all(state.density[idx_right] == rho_right)

    idx_interpolate = np.where(
        np.logical_and(const_geometry.xc > x_left, const_geometry.xc < x_right)
    )[0]
    dphidx = (rho_right - rho_left) / dx_smooth
    dx = const_geometry.dx
    if isinstance(dx, np.ndarray):
        dx = np.diff(const_geometry.xc[idx_interpolate])
    assert np.diff(state.density[idx_interpolate]) / dx == pytest.approx(dphidx)


def test_isentropic_init(
    left_state: tuple[Solution, float], physics: CanteraInterface, geometry: Geometry
) -> None:
    choking_area_ratio = 10.0
    gas = left_state[0]
    area = geometry.area(0.0, geometry.xf)
    throat_area: float = area[0] / choking_area_ratio
    subsonic_outflow = area.min() - throat_area > 1e-3
    state = InitializeIsentropic(
        geometry=geometry,
        physics=physics,
        inflow_state=gas,
        throat_area=throat_area,
        subsonic_inflow=True,
        subsonic_outflow=subsonic_outflow,
    )()

    # Verify that the Mach numbers match the area ratio
    assert state.velocity is not None
    g = gas.cp / gas.cv
    area_ratio = geometry.area(0.0, geometry.xc) / throat_area
    mach = state.velocity / physics.get_sound_speed(state)
    area_ratio_target = area_ratio_from_mach(mach, g)

    assert area_ratio_target == pytest.approx(area_ratio)

    # Verify the inflow and outflow Mach numbers are correct
    assert mach[-1] < 1.0 if subsonic_outflow else mach[-1] > 1.0
