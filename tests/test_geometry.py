from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Generic, TypeAlias, TypedDict, TypeVar

import numpy as np
import pytest
from scipy.integrate import quad

from stanshock.system.backend import Array
from stanshock.system.geometry import (
    AsymmetricBox,
    Box,
    Cylinder,
    Geometry,
    SpatioTemporalLike,
    initialize_geometry,
)

T = TypeVar("T")


class FixtureRequest(pytest.FixtureRequest, Generic[T], ABC):
    param: T


_FuncDerivIntegral: TypeAlias = tuple[
    str, SpatioTemporalLike, Callable[[float], float], Callable[[float], float]
]


class PointData(TypedDict):
    x: Array
    area: float
    perimeter: float
    hydraulic_diameter: float
    characteristic_length: float


class GeometryTestCase(TypedDict):
    geometry: type[Geometry]
    definition: dict[str, SpatioTemporalLike | None]
    surface_area: float
    volume: float
    point_data: PointData


nfaces = 201
xf: Array = np.linspace(-1.0, 1.0, nfaces, dtype=np.float64)
x0: Array = np.array([0.0])

cases: dict[str, GeometryTestCase] = {
    "generic-quadratic": {
        "geometry": Geometry,
        "definition": {
            "xf": xf,
            "area": lambda _t, x: x**2 + 1.0,
            "perimeter": lambda _t, x: x**2 + 1.0,
            "dlnA_dx": lambda _t, x: 2.0 * x / (x**2 + 1.0),
        },
        "surface_area": np.nan,
        "volume": 8.0 / 3.0,
        "point_data": {
            "x": x0,
            "area": 1.0,
            "perimeter": 1.0,
            "hydraulic_diameter": 4.0,
            "characteristic_length": 4.0,
        },
    },
    "cylinder-const": {
        "geometry": Cylinder,
        "definition": {
            "xf": xf,
            "d_outer": 0.5,
            "d_inner": 0.1,
            "n_ghost_layers": 3,
        },
        "surface_area": np.pi,
        "volume": 0.5 * np.pi * (0.5**2 - 0.1**2),
        "point_data": {
            "x": x0,
            "area": 0.25 * np.pi * (0.5**2 - 0.1**2),
            "perimeter": np.pi * (0.5 + 0.1),
            "hydraulic_diameter": 0.5 - 0.1,
            "characteristic_length": 0.5 * (0.5 - 0.1),
        },
    },
    "box-contract": {
        "geometry": Box,
        "definition": {
            "xf": xf,
            "h": np.linspace(2.0, 1.0, nfaces, dtype=np.float64),
            "w": 1.0,
            "n_ghost_layers": 1,
        },
        "surface_area": 2.0 * (np.sqrt(4.25) + 3.0),
        "volume": 3.0,
        "point_data": {
            "x": x0,
            "area": 1.5,
            "perimeter": 5.0,
            "hydraulic_diameter": 3.0 / 2.5,
            "characteristic_length": 3.0 / 2.5,
        },
    },
    "asymm-expand": {
        "geometry": AsymmetricBox,
        "definition": {
            "xf": xf,
            "upper_wall": (np.array([-1.0, 1.0]), np.array([1.0, 2.0])),
            "lower_wall": None,
        },
        "surface_area": np.sqrt(5.0) + 8.0,
        "volume": 3.0,
        "point_data": {
            "x": x0,
            "area": 1.5,
            "perimeter": 5.0,
            "hydraulic_diameter": 3.0 / 2.5,
            "characteristic_length": 3.0 / 2.5,
        },
    },
}

case_ids: tuple[str, ...] = tuple(cases.keys())
case_data: tuple[GeometryTestCase, ...] = tuple(cases[id] for id in case_ids)


@pytest.fixture(params=case_data, ids=case_ids, scope="module")
def test_case(request: FixtureRequest[GeometryTestCase]) -> GeometryTestCase:
    return request.param


@pytest.fixture(scope="module")
def geometry(test_case: GeometryTestCase) -> Geometry:
    geometry_type = test_case["geometry"]
    kwargs = test_case["definition"]
    return geometry_type(**kwargs)


def test_geometry_initializer(test_case: GeometryTestCase) -> None:
    geometry_type = test_case["geometry"]
    kwargs = test_case["definition"]
    geometry_test = initialize_geometry(**kwargs)

    assert type(geometry_test) is geometry_type


def test_adding_ghost_layers(geometry: Geometry) -> None:
    n_ghost_layers = 3
    n_ghost_layers_added = n_ghost_layers - geometry.n_ghost_layers

    n_cells_before = len(geometry.xc)
    x_cells_before = geometry.xc[geometry.idx_cells]

    geometry.setup_ghost_layers(n_ghost_layers=3)

    n_cells_after = len(geometry.xc)
    x_cells_after = geometry.xc[geometry.idx_cells]

    assert n_cells_after == n_cells_before + 2 * n_ghost_layers_added
    assert np.array_equal(x_cells_after, x_cells_before)
    assert np.all(np.diff(geometry.xc) > 0.0)


def test_volume_against_quad(geometry: Geometry) -> None:
    volume_total: float = geometry.volume().sum()
    volume_quad: float = quad(
        lambda x: geometry.area(0.0, x), a=geometry.xf[0], b=geometry.xf[-1]
    )[0]

    assert volume_total == pytest.approx(volume_quad)


def test_surface_area_against_analytical(
    test_case: GeometryTestCase, geometry: Geometry
) -> None:
    surf_area_analytical = test_case["surface_area"]

    if test_case["geometry"] is Geometry:
        pytest.xfail(
            "Geometry objects don't have enough information to compute surface area."
        )

    surf_area_total: float = geometry.surface_area().sum()

    assert surf_area_total == pytest.approx(surf_area_analytical, rel=1e-5)


def test_volume_against_analytical(
    test_case: GeometryTestCase, geometry: Geometry
) -> None:
    volume_analytical = test_case["volume"]

    volume_total: float = geometry.volume().sum()

    assert volume_total == pytest.approx(volume_analytical, rel=1e-5)


def test_cross_section_against_analytical(
    test_case: GeometryTestCase, geometry: Geometry
) -> None:
    point_data = test_case["point_data"]
    x = point_data["x"]

    assert geometry.area(0.0, x) == pytest.approx(point_data["area"])
    assert geometry.perimeter(0.0, x) == pytest.approx(point_data["perimeter"])
    assert geometry.hydraulic_diameter(0.0, x) == pytest.approx(
        point_data["hydraulic_diameter"]
    )
    assert geometry.characteristic_length(0.0, x) == pytest.approx(
        point_data["characteristic_length"]
    )
