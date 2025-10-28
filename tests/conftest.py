from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import pytest
from cantera import Solution

from stanshock.components.combustor import Combustor
from stanshock.numerics.boundary_conditions import Inflow
from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.physics.fluid_base import FluidPhysics
from stanshock.physics.thermotable import ThermoTable
from stanshock.system.backend import Array
from stanshock.system.geometry import Geometry, initialize_geometry
from stanshock.utils.isentropic import mach_from_area_ratio

T = TypeVar("T")


class FixtureRequest(pytest.FixtureRequest, Generic[T], ABC):
    param: T


# Command line option for enabling slow tests
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(
    config: pytest.Config, items: Iterable[pytest.Item]
) -> None:
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow: pytest.MarkDecorator = pytest.mark.skip(
        reason="need --runslow option to run"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# Reusable fixtures for each piece of a full case setup
# @pytest.fixture(params=[21, 201], ids=["coarse", "fine"], scope="session")
# @pytest.fixture(params=[21], ids=["coarse"], scope="session")
@pytest.fixture(params=[201], ids=["fine"], scope="session")
def num_points(request: FixtureRequest[int]) -> int:
    return request.param


@pytest.fixture(params=[2.0, 10.0], scope="session")
def area_range(request: FixtureRequest[float], scope="session") -> float:
    # Ratio of the largest to smallest flow area
    return request.param


@pytest.fixture(
    params=["constant", "converging", "diverging", "converging-diverging"],
    scope="session",
)
def area(request: FixtureRequest[str], area_range: float, num_points: int) -> Array:
    area: Array = np.ones((num_points,), dtype=np.float64)
    if request.param == "converging":
        area = np.linspace(area_range, 1.0, num_points, dtype=np.float64)
    elif request.param == "diverging":
        area = np.linspace(1.0, area_range, num_points, dtype=np.float64)
    elif request.param == "converging-diverging":
        scale = 5.0
        shift = 2.0
        x = np.linspace(-scale, scale, num_points, dtype=np.float64)
        area = np.tanh(x - shift) - np.tanh(x + shift) + 2.0
        area_min = np.tanh(-shift) - np.tanh(shift) + 2.0
        area_max = np.tanh(scale - shift) - np.tanh(scale + shift) + 2.0
        area = (area - area_min) / (area_max - area_min) * (
            1.0 - 1.0 / area_range
        ) + 1.0 / area_range
    return area


@pytest.fixture(
    params=["box", "cylinder"],
    scope="session",
)
def geometry(request: FixtureRequest[str], area: Array) -> Geometry:
    num_points = area.shape[0]
    x = np.linspace(0.0, 10.0, num_points, dtype=np.float64)

    if request.param == "box":
        geometry: Geometry = initialize_geometry(x=x, h=area)
    elif request.param == "cylinder":
        geometry = initialize_geometry(x=x, d_outer=2.0 * np.sqrt(area / np.pi))
    else:
        geometry = initialize_geometry(x=x, area=area)
    return geometry


@pytest.fixture(
    params=["HeliumArgon.yaml"],
    ids=["HeAr"],
    scope="session",
)
def mechanism(request: FixtureRequest[str]) -> Path:
    return (
        Path(__file__).resolve().parent / ".." / "data" / "mechanisms" / request.param
    )


@pytest.fixture(scope="session")
def gas(mechanism: Path) -> Solution:
    gas = Solution(mechanism)
    nsp = gas.n_species
    gas.TPY = 3000.0, 30e6, np.ones((nsp,)) / nsp
    return gas


@pytest.fixture(
    params=[ThermoTable, CanteraInterface],
    scope="session",
)
def fluid_physics(
    request: FixtureRequest[type[FluidPhysics]], gas: Solution
) -> FluidPhysics:
    return request.param(gas)


# Set inflow boundary condition for a choked flow with area ratio 10.0
@pytest.fixture(scope="session")
def inflow_bc(gas: Solution) -> Inflow:
    g: float = gas.cp / gas.cv
    area_ratio = np.array([10.0])

    # Get subsonic result
    inflow_mach: float = mach_from_area_ratio(area_ratio, g, subsonic=True)[0]

    # Get corresponding inflow velocity and return inflow definition
    inflow_velocity = inflow_mach * gas.sound_speed

    return Inflow(reference_state=(gas.density, inflow_velocity, gas.P, gas.Y))


@pytest.fixture(scope="session")
def isentropic_flow(
    gas: Solution, fluid_physics: FluidPhysics, geometry: Geometry, inflow_bc: Inflow
) -> Combustor:
    # Get throat area at which flow will choke
    throat_area: float = geometry.area(0.0, geometry.x[0]) / 10.0

    # Set up simulation object
    return Combustor(
        boundary_conditions=[inflow_bc, "outflow"],
        geometry=geometry,
        initialization=("isentropic", gas, throat_area),
        physics=fluid_physics,
    )
