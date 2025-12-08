from __future__ import annotations

from abc import ABC
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import pytest
from cantera import Solution

from stanshock.components.combustor import Combustor
from stanshock.numerics.boundary_conditions import BCInput, FreezeCells, SpecifiedFace
from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.physics.fluid_base import FluidPhysics
from stanshock.physics.thermotable import ThermoTable
from stanshock.processing.initialize import InitializeIsentropic
from stanshock.system.backend import Array
from stanshock.system.geometry import (
    Geometry,
    SpatioTemporalFunction,
    initialize_geometry,
)
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
@pytest.fixture(params=[401, 1001], ids=["coarse", "fine"], scope="session")
def num_points(request: FixtureRequest[int]) -> int:
    return request.param


choking_area_ratio = 10.0


@pytest.fixture(
    params=["constant", "converging", "diverging", "converging-diverging"],
    scope="session",
)
def area(request: FixtureRequest[str]) -> SpatioTemporalFunction:
    if request.param == "converging":
        area_range = 4.0

        def area_function(_t: float, x: Array) -> Array:
            return 1.0 + (1.0 - 0.1 * x) * (area_range - 1.0)

    elif request.param == "diverging":
        area_range = 2.0

        def area_function(_t: float, x: Array) -> Array:
            return 1.0 + 0.1 * x * (area_range - 1.0)

    elif request.param == "converging-diverging":
        area_range = choking_area_ratio

        def area_function(_t: float, x: Array) -> Array:
            scale = 5.0
            shift = 2.0

            x = x - scale

            area = np.tanh(x - shift) - np.tanh(x + shift) + 2.0
            area_min = np.tanh(-shift) - np.tanh(shift) + 2.0
            area_max = np.tanh(scale - shift) - np.tanh(scale + shift) + 2.0
            return (area - area_min) / (area_max - area_min) * (
                1.0 - 1.0 / area_range
            ) + 1.0 / area_range
    else:

        def area_function(_t: float, x: Array) -> Array:
            return np.ones_like(x)

    return area_function


@pytest.fixture(
    params=["box", "cylinder"],
    scope="session",
)
def geometry(
    request: FixtureRequest[str], area: SpatioTemporalFunction, num_points: int
) -> Geometry:
    xf = np.linspace(0.0, 10.0, num_points, dtype=np.float64)

    if request.param == "box":
        geometry: Geometry = initialize_geometry(xf=xf, h=area)

    elif request.param == "cylinder":

        def d_outer(t: float, x: Array) -> Array:
            return 2.0 * np.sqrt(area(t, x) / np.pi)

        geometry = initialize_geometry(xf=xf, d_outer=d_outer)
    else:
        geometry = initialize_geometry(xf=xf, area=area)
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
    return Solution(mechanism)


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
def inflow_bc(gas: Solution) -> SpecifiedFace:
    nsp = gas.n_species
    gas.TPY = 3000.0, 30e6, np.ones((nsp,)) / nsp
    g: float = gas.cp / gas.cv
    area_ratio = np.array([choking_area_ratio])

    # Get subsonic result
    inflow_mach: float = mach_from_area_ratio(area_ratio, g, subsonic=True)[0]

    # Get corresponding inflow velocity and return inflow definition
    inflow_velocity = inflow_mach * gas.sound_speed

    return SpecifiedFace(reference_state=(gas.density, inflow_velocity, gas.P, gas.Y))


@pytest.fixture(scope="session")
def isentropic_flow(
    gas: Solution,
    fluid_physics: FluidPhysics,
    geometry: Geometry,
) -> Combustor:
    # Reinitialize Solution object to inflow conditions
    nsp = gas.n_species
    gas.TPY = 3000.0, 30e6, np.ones((nsp,)) / nsp
    bcs: BCInput = {
        "left": FreezeCells(location="left"),
        "right": FreezeCells(location="right"),
    }

    # Get throat area at which flow will choke
    area = geometry.area(0.0, geometry.xf)
    throat_area: float = area[0] / choking_area_ratio
    subsonic_outflow = area.min() - throat_area > 1e-3
    initialization = InitializeIsentropic(
        geometry=geometry,
        physics=fluid_physics,
        inflow_state=gas,
        throat_area=throat_area,
        subsonic_inflow=True,
        subsonic_outflow=subsonic_outflow,
    )

    # Set up simulation object
    return Combustor(
        boundary_conditions=bcs,
        geometry=geometry,
        initialization=initialization,
        physics=fluid_physics,
    )
