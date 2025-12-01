from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypedDict

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from stanshock.numerics.time_integration import (
    RK4,
    SSPRK3,
    FastSlowIntegrator,
    ForwardEuler,
    HeunsMethod,
    LieSplitting,
    MidpointMethod,
    OperatorSplitting,
    StrangSplitting,
    SymmetricallyWeightedSequentialSplitting,
    ThirdOrderSplitting,
    TimeIntegrator,
)
from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Unpack
from stanshock.system.base import (
    CombinedSource,
    FastSlowSource,
    PrecomputeSteps,
    RightHandSide,
)

from .conftest import FixtureRequest


# Set up some simple test problems
class Brusselator(RightHandSide):
    REQUIRED_PRECOMPUTE_STEPS = ()

    def __init__(self) -> None:
        super().__init__()
        self.shape_full = (-1, 2)
        self.shape_input = (-1, 2)
        self.shape_output = (-1, 2)

        self.abcd: tuple[float, float, float, float] = (1.0, 3.0, 1.0, 1.0)

    def source(
        self,
        time: float,
        state_array_local: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        _ = time, gamma_star, e0_star
        a, b, c, _ = self.abcd
        state_array_local = np.reshape(state_array_local, self.shape_input)
        rhs = np.zeros_like(state_array_local)

        x = state_array_local[:, 0]
        y = state_array_local[:, 1]

        rhs[:, 0] = a - (b + 1.0) * x + c * x**2 * y
        rhs[:, 1] = b * x - c * x**2 * y

        return np.ravel(rhs)


class Circle(RightHandSide):
    REQUIRED_PRECOMPUTE_STEPS = ()

    def __init__(self) -> None:
        super().__init__()
        self.shape_full = (-1, 2)
        self.shape_input = (-1, 2)
        self.shape_output = (-1, 2)

    def source(
        self,
        time: float,
        state_array_local: Array,
        gamma_star: Array | None = None,
        e0_star: Array | None = None,
    ) -> Array:
        _ = time, gamma_star, e0_star
        state_array_local = np.reshape(state_array_local, self.shape_input)
        rhs = np.zeros_like(state_array_local)

        x = state_array_local[:, 0]
        y = state_array_local[:, 1]

        theta = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)

        rhs[:, 0] = -r * np.sin(theta)
        rhs[:, 1] = r * np.cos(theta)

        return np.ravel(rhs)


class LotkaVolterra(FastSlowSource):
    REQUIRED_PRECOMPUTE_STEPS = ()
    idx_output_explicit = np.s_[:]
    idx_output_implicit = np.s_[:]
    idx_source_explicit = np.array([0])
    idx_source_implicit = np.array([1])

    def __init__(
        self,
        mode: Literal["slow", "fast"] = "slow",
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        """Split RHS into fast and slow source terms accessed by setting the mode."""
        super().__init__(mode, **precompute_steps)
        self.shape_full = self.shape_input = (-1, 2)
        self.shape_output = (-1, 1)

    def source_slow(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Slow source term is constant."""
        _ = time, state, face_states, avg_face_states, face_gradients
        assert state_array_local is not None
        # print(f"Slow source: {state_array_local.shape = }")
        u = state_array_local[:, 0]
        v = state_array_local[:, 1]
        return u * (v - 1.0)

    def source_fast(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Fast source term is exponential."""
        _ = time, state, face_states, avg_face_states, face_gradients
        assert state_array_local is not None
        # print(f"Fast source: {state_array_local.shape = }")
        u = state_array_local[:, 0]
        v = state_array_local[:, 1]
        return v * (1.0 - u)


# Set up test cases for typical source terms and integrators
@dataclass
class TimeIntegrationCase:
    rhs: type[RightHandSide] | list[type[RightHandSide]]
    t_span: tuple[float, float]
    y_init: Array
    analytical_function: Callable[[Array], Array] | None = None
    reference_solution: Array = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.rhs, list):
            rhs: RightHandSide = CombinedSource([rhs() for rhs in self.rhs])
        elif issubclass(self.rhs, FastSlowSource):
            rhs = CombinedSource([self.rhs("slow"), self.rhs("fast")])
        else:
            rhs = self.rhs()

        n = 5 * 7 * 8 * 9 * 11 + 1
        t_eval = np.linspace(*self.t_span, n)

        if self.analytical_function is None:
            ode_result = solve_ivp(
                fun=rhs.source,
                t_span=self.t_span,
                y0=self.y_init,
                t_eval=t_eval,
                method="DOP853",
                atol=1e-15,
                rtol=3e-14,
                # dense_output=True,
            )
            self.reference_solution = ode_result.y
        else:
            self.reference_solution = self.analytical_function(t_eval)


# Setup for standard time integration tests
class IntegratorInfo(TypedDict):
    integrator: type[TimeIntegrator]
    order: int


test_problems: dict[str, TimeIntegrationCase] = {
    "brusselator": TimeIntegrationCase(
        rhs=Brusselator,
        t_span=(0.0, 20.0),
        y_init=np.array((1.0, 0.0)),
    ),
    "circle": TimeIntegrationCase(
        rhs=Circle,
        t_span=(0.0, 6.0 * np.pi),
        y_init=np.array((1.0, 0.0)),
        analytical_function=lambda t: np.array((np.cos(t), np.sin(t))),
    ),
    "lotkavolterra": TimeIntegrationCase(
        rhs=LotkaVolterra,
        t_span=(0.0, 2.5),
        y_init=np.array((10.0, 6.0)),
    ),
}

test_integrators: dict[str, IntegratorInfo] = {
    # Standard integrators
    "FE": {"integrator": ForwardEuler, "order": 1},
    "heun": {"integrator": HeunsMethod, "order": 2},
    "midpnt": {"integrator": MidpointMethod, "order": 2},
    "ssprk3": {"integrator": SSPRK3, "order": 3},
    "rk4": {"integrator": RK4, "order": 4},
    # Operator-split integrators
    "lie": {"integrator": LieSplitting, "order": 1},
    "fastslow": {"integrator": FastSlowIntegrator, "order": 1},
    "strang": {"integrator": StrangSplitting, "order": 2},
    "swss": {"integrator": SymmetricallyWeightedSequentialSplitting, "order": 2},
    # Haven't observed third-order convergence with this:
    "o3split": {"integrator": ThirdOrderSplitting, "order": 2},
}


def get_convergence_order(x: Array, y: Array, n_discard: int = 4) -> float:
    # Remove points very close to machine epsilon
    idx = np.where(y >= 1e-15)[0]
    x = np.log(x[idx])
    y = np.log(y[idx])

    resid = 1.0
    slope = -5.0
    for i in range(n_discard + 1):
        # Estimate the order of convergence
        poly, (resid, _, _, _) = np.polynomial.Polynomial.fit(
            x[i:], y[i:], 1, full=True
        )
        slope = poly.convert().coef[1]

        if resid < 0.01:
            # If the fit is poor, optionally discard up to n_discard of the
            # smallest values, in case truncation error is dominating.
            break

    return slope


@pytest.fixture(
    params=list(test_problems.values()), ids=list(test_problems.keys()), scope="module"
)
def test_case(request: FixtureRequest[TimeIntegrationCase]) -> TimeIntegrationCase:
    return request.param


@pytest.fixture(
    params=list(test_integrators.values()),
    ids=list(test_integrators.keys()),
    scope="module",
)
def integrator_info(request: FixtureRequest[IntegratorInfo]) -> IntegratorInfo:
    return request.param


@pytest.fixture
def integrator(
    integrator_info: IntegratorInfo, test_case: TimeIntegrationCase
) -> TimeIntegrator:
    integrator = integrator_info["integrator"]

    if integrator is FastSlowIntegrator:
        # Requires FastSlowSource
        if isinstance(test_case.rhs, list) or not issubclass(
            test_case.rhs, FastSlowSource
        ):
            pytest.skip("FastSlowIntegrator requires a FastSlowSource.")
        return integrator(test_case.rhs())

    rhs: RightHandSide | None = None
    rhs_list: list[RightHandSide] | None = None

    if isinstance(test_case.rhs, list):
        rhs_list = [rhs() for rhs in test_case.rhs]
    elif issubclass(test_case.rhs, FastSlowSource):
        # Split a fast/slow source into two RHS objects
        rhs_list = [test_case.rhs("slow"), test_case.rhs("fast")]
    else:
        rhs = test_case.rhs()

    if issubclass(integrator, OperatorSplitting):
        if rhs_list is None:
            pytest.skip("Can't apply operator splitting to single source term.")
        else:
            operators: tuple[TimeIntegrator, ...] = tuple(
                [MidpointMethod(rhs) for rhs in rhs_list]
            )

        return integrator(operators=operators)

    if rhs_list is not None:
        rhs = CombinedSource(rhs_list)

    assert rhs is not None
    return integrator(rhs)


def test_local_truncation_error_convergence(
    test_case: TimeIntegrationCase,
    integrator_info: IntegratorInfo,
    integrator: TimeIntegrator,
) -> None:
    """Verify that error from a single step converges with the expected order."""
    # Preallocate different time step sizes
    t0, tf = test_case.t_span
    n = np.shape(test_case.reference_solution)[1] - 1
    dt0 = (tf - t0) / n
    t_ref = np.linspace(t0, tf, n + 1)

    N = 12
    dt = np.arange(1, N + 1, dtype=np.float64) * dt0

    # Get reference solution
    y_ref = test_case.reference_solution

    # Take single time step with different step sizes
    y_err_local = np.zeros((N,))
    for i in range(N):
        assert dt[i] == pytest.approx(t_ref[i + 1] - t_ref[0])
        _, y, _, _ = integrator.advance(dt[i], t0, test_case.y_init.copy(), None, None)
        y_err_local[i] = np.linalg.norm(y_ref[:, i + 1] - y)

    err_slope = get_convergence_order(dt, y_err_local)

    assert err_slope == pytest.approx(integrator_info["order"] + 1, abs=0.03)


def test_global_truncation_error_convergence(
    test_case: TimeIntegrationCase,
    integrator_info: IntegratorInfo,
    integrator: TimeIntegrator,
) -> None:
    """Verify that error over a fixed interval converges with time step size."""
    # Preallocate different time step sizes
    t0, tf = test_case.t_span
    n = np.shape(test_case.reference_solution)[1] - 1
    dt0 = (tf - t0) / n

    N = 12
    dt = np.arange(1, N + 1, dtype=np.float64) * dt0

    # Get reference solution
    y_init = test_case.y_init
    y_ref = test_case.reference_solution[:, -1]

    # Integrate over t_span with different step sizes
    y_err_global = np.zeros((N,))
    for i in range(N):
        t = t0 + 0
        y = y_init.copy()
        for _j in range(n // (i + 1)):
            t, y, _, _ = integrator.advance(dt[i], t, y, None, None)
        assert t == pytest.approx(tf)
        y_err_global[i] = np.linalg.norm(y_ref - y)

    err_slope = get_convergence_order(dt, y_err_global)

    assert err_slope == pytest.approx(integrator_info["order"], abs=0.05)
