from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Generic, TypeAlias, TypeVar

import numpy as np
import pytest
from scipy.integrate import quad

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


# Function, f(x), and its derivative + integral w.r.t. x
@pytest.fixture(
    params=[
        ("constant", 1.0, lambda _x: 0.0, lambda x: x),
        (
            "expand",
            (np.array([-1.0, 1.0]), np.array([1.0, 2.0])),  # y(x) = 0.5*x + 1.5
            lambda _x: 0.5,
            lambda x: 0.25 * x**2 + 1.5 * x,
        ),
        (
            "contract",
            (np.array([-1.0, 1.0]), np.array([2.0, 1.0])),  # y(x) = -0.5*x + 1.5
            lambda _x: -0.5,
            lambda x: -0.25 * x**2 + 1.5 * x,
        ),
        (
            "quadratic",
            lambda _t, x: x**2 + 1.0,
            lambda x: 2.0 * x,
            lambda x: x**3 / 3.0 + x,
        ),
    ],
    ids=["constant", "expand", "contract", "quadratic"],
    scope="module",
)
def key_function(request: FixtureRequest[_FuncDerivIntegral]) -> _FuncDerivIntegral:
    return request.param


# Geometry object, function used to define it, surface integral, and volume integral
@pytest.fixture(
    params=[
        (Geometry, "area"),
        (Cylinder, "d_outer"),
        (Box, "h"),
        (AsymmetricBox, "upper_wall"),
    ],
    ids=["generic", "cylinder", "box", "asymmbox"],
    scope="module",
)
def geometry_def(
    request: FixtureRequest[tuple[type[Geometry], str]],
) -> tuple[type[Geometry], str]:
    return request.param


@pytest.fixture(scope="module")
def geometry(
    geometry_def: tuple[type[Geometry], str], key_function: _FuncDerivIntegral
) -> Geometry:
    geometry_type, func_name = geometry_def
    xf = np.linspace(-1.0, 1.0, 201)
    kwargs = {"xf": xf, func_name: key_function[1]}
    return geometry_type(**kwargs)


def test_geometry_initializer(
    geometry_def: tuple[type[Geometry], str],
    key_function: _FuncDerivIntegral,
    geometry: Geometry,
) -> None:
    geometry_type, func_name = geometry_def
    xf = np.linspace(-1.0, 1.0, 201)
    kwargs = {"xf": xf, func_name: key_function[1]}
    geometry_test = initialize_geometry(**kwargs)

    assert type(geometry_test) is geometry_type
    assert geometry_test.area(0.0, xf) == pytest.approx(geometry.area(0.0, xf))


def test_surface_area_against_quad(
    geometry: Geometry, key_function: _FuncDerivIntegral
) -> None:
    _, _, dfdx, int_f = key_function

    if isinstance(geometry, Cylinder):
        # Surface of revolution
        f = geometry.d_outer
        surf_area_quad: float = (
            np.pi
            * quad(
                lambda x: f(0.0, x) * np.sqrt(1.0 + (0.5 * dfdx(x)) ** 2),
                a=geometry.xf[0],
                b=geometry.xf[-1],
            )[0]
        )
    elif isinstance(geometry, AsymmetricBox):
        w = geometry.w(0.0, geometry.xf)
        if isinstance(w, float):
            arc_length = quad(
                lambda x: np.sqrt(1.0 + dfdx(x) ** 2),
                a=geometry.xf[0],
                b=geometry.xf[-1],
            )[0]
            surf_area_quad = 2.0 * (
                int_f(geometry.xf[-1]) - int_f(geometry.xf[0])
            ) + w * (geometry.xf[-1] - geometry.xf[0] + arc_length)
        else:
            pytest.xfail("Not set up to handle variable width AsymmetricBox.")
    elif isinstance(geometry, Box):
        w = geometry.w(0.0, geometry.xf)
        if isinstance(w, float):
            arc_length = quad(
                lambda x: np.sqrt(1.0 + (0.5 * dfdx(x)) ** 2),
                a=geometry.xf[0],
                b=geometry.xf[-1],
            )[0]
            surf_area_quad = 2.0 * (
                int_f(geometry.xf[-1]) - int_f(geometry.xf[0]) + w * arc_length
            )
        else:
            pytest.xfail("Not set up to handle variable width Box.")
    else:
        pytest.xfail(
            "Geometry objects don't have enough information to compute surface area."
        )

    surf_area_total: float = geometry.surface_area().sum()

    assert surf_area_total == pytest.approx(surf_area_quad, rel=1e-5)


def test_volume_against_quad(geometry: Geometry) -> None:
    volume_total: float = geometry.volume().sum()
    volume_quad: float = quad(
        lambda x: geometry.area(0.0, x), a=geometry.xf[0], b=geometry.xf[-1]
    )[0]

    assert volume_total == pytest.approx(volume_quad)


def test_surface_area_against_analytical(
    geometry: Geometry, key_function: _FuncDerivIntegral
) -> None:
    function_name, _, _, _ = key_function

    if isinstance(geometry, Cylinder):
        if function_name == "constant":
            surf_area_analytical = 2.0 * np.pi
        elif function_name == "quadratic":
            surf_area_analytical = (
                0.25
                * np.pi
                * (3.0 * np.arctanh(1.0 / np.sqrt(2.0)) + 7.0 * np.sqrt(2.0))
            )
        else:
            surf_area_analytical = np.pi * (np.sqrt(17.0) - 0.5 * np.sqrt(4.25))
    elif isinstance(geometry, AsymmetricBox):
        if function_name == "constant":
            surf_area_analytical = 8.0
        elif function_name == "quadratic":
            surf_area_analytical = 22.0 / 3.0 + np.sqrt(5.0) + 0.5 * np.asinh(2.0)
        else:
            surf_area_analytical = np.sqrt(5.0) + 8.0
    elif isinstance(geometry, Box):
        if function_name == "constant":
            surf_area_analytical = 8.0
        elif function_name == "quadratic":
            surf_area_analytical = (
                16.0 / 3.0 + np.sqrt(8.0) + 2.0 * np.arctanh(1.0 / np.sqrt(2.0))
            )
        else:
            surf_area_analytical = 2.0 * (np.sqrt(4.25) + 3.0)
    else:
        pytest.xfail(
            "Geometry objects don't have enough information to compute surface area."
        )

    surf_area_total: float = geometry.surface_area().sum()

    assert surf_area_total == pytest.approx(surf_area_analytical, rel=1e-5)


def test_volume_against_analytical(
    geometry: Geometry, key_function: _FuncDerivIntegral
) -> None:
    function_name, _, _, int_f = key_function

    if isinstance(geometry, Cylinder):
        if function_name == "constant":
            volume_analytical = 0.5 * np.pi
        elif function_name == "quadratic":
            volume_analytical = 14.0 / 15.0 * np.pi
        else:
            volume_analytical = np.pi * (4.0 - 0.5) / 3.0
    elif isinstance(geometry, Box | AsymmetricBox):
        if function_name == "constant":
            volume_analytical = 2.0
        elif function_name == "quadratic":
            volume_analytical = 8.0 / 3.0
        else:
            volume_analytical = 3.0
    else:
        # For Geometry f(x) is the area, so int_f(x) is the volume
        volume_analytical = int_f(geometry.xf[-1]) - int_f(geometry.xf[0])

    volume_total: float = geometry.volume().sum()

    assert volume_total == pytest.approx(volume_analytical)
