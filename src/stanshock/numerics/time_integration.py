from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from stanshock.physics.fluid_base import FluidPhysics
from stanshock.system.backend import Array
from stanshock.system.base import FastSlowSource, RightHandSide


class TimeIntegrator(ABC):
    def __init__(self, rhs: RightHandSide) -> None:
        self.rhs = rhs

    @abstractmethod
    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array,
        e0_star: Array,
    ) -> tuple[float, Array]:
        """Integrate the state from `time` to `time+dt` given the rhs."""


class ScipyIVP(TimeIntegrator):
    def __init__(self, rhs: RightHandSide, method: str = "LSODA", **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Interface to SciPy's IVP solvers.

        The specific solver can be selected with the `method` input, and additional
        solver options can be passed in as keyword arguments. See documentation for
        `scipy.integrate.solve_ivp`.
        """
        # from scipy.integrate import ode
        from scipy.integrate import solve_ivp

        # self.integrator = ode
        self.integrator = solve_ivp

        self.rhs = rhs
        self.method = method
        self.solver_options = kwargs

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array,
        e0_star: Array,
    ) -> tuple[float, Array]:
        y0, state0, _, _, _ = self.rhs.precompute(
            time, state_array, gamma_star, e0_star
        )
        assert state0 is not None

        results = self.integrator(
            fun=self.rhs.source,
            t_span=(time, time + dt),
            y0=y0,
            method=self.method,
            args=(gamma_star, e0_star),
            **self.solver_options,
        )

        state_array = self.rhs.postcompute(results.y[:, -1], state0)

        return results.t[-1], state_array


class RungeKuttaBase(TimeIntegrator):
    n_rk_stage: int = 0
    rk_coeff: Array = np.zeros((1, 4))

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array,
        e0_star: Array,
    ) -> tuple[float, Array]:
        t = time
        y: Array = np.ravel(state_array)

        t0 = t + 0
        y0 = y.copy()

        for i_rk_stage in range(self.n_rk_stage):
            a, b, c, d = self.rk_coeff[i_rk_stage]

            dydt = self.rhs.source(
                time=t,
                state_array=y,
                gamma_star=gamma_star,
                e0_star=e0_star,
            )

            if b != 1.0:
                y *= b

            if a != 0.0:
                y += a * y0

            y = self.rhs.add_source(y, c * dt * dydt)

            t = t0 + d * dt

        return t, y


class ForwardEuler(RungeKuttaBase):
    n_rk_stage: int = 1
    rk_coeff: Array = np.array([[0.0, 1.0, 1.0, 1.0]])


class MidpointMethod(RungeKuttaBase):
    n_rk_stage: int = 2
    rk_coeff: Array = np.array([[0.0, 0.5, 0.5, 0.5], [0.0, 1.0, 1.0, 1.0]])


class HeunsMethod(RungeKuttaBase):
    n_rk_stage: int = 2
    rk_coeff: Array = np.array([[0.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 1.0]])


class SSPRK3(RungeKuttaBase):
    """
    3rd-order strong stability preserving Runge Kutta (SSPRK3). This scheme is
    stable for CFL <= 1.

    References
    ----------
    Dale E. Durran, “Numerical Methods for Fluid Dynamics”, Springer.
    Second Edition.
    """

    n_rk_stage: int = 3
    rk_coeff: Array = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [0.75, 0.25, 0.25, 0.5],
            [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0],
        ]
    )


class RK4(RungeKuttaBase):
    n_rk_stage: int = 4
    rk_coeff: Array = np.array(
        [
            [0.0, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0 / 6.0, 0.5],
            [0.5, 0.5, 1.0 / 3.0, 1.0],
            [0.0, 1.0, 1.0 / 6.0, 1.0],
        ]
    )


class StrangSplitting(TimeIntegrator):
    def __init__(
        self, transport_operator: TimeIntegrator, reaction_operator: TimeIntegrator
    ) -> None:
        self.transport_operator = transport_operator
        self.reaction_operator = reaction_operator

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array,
        e0_star: Array,
    ) -> tuple[float, Array]:
        y = state_array.flatten()

        # Take half-step with transport terms
        _, y = self.transport_operator.advance(
            dt=0.5 * dt,
            time=time,
            state_array=y,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

        # Take full-step with reaction terms
        t, y = self.reaction_operator.advance(
            dt=dt,
            time=time,
            state_array=y,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

        # Take half-step with transport terms
        _, y = self.transport_operator.advance(
            dt=0.5 * dt,
            time=time + 0.5 * dt,
            state_array=y,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

        return t, y


class LieSplitting(TimeIntegrator):
    def __init__(
        self, operators: list[TimeIntegrator], update_double_flux: bool = False
    ) -> None:
        self.operators = operators
        self.update_double_flux = update_double_flux

        # Get fluid physics from any of the operators
        self.physics: FluidPhysics | None = None
        if self.update_double_flux:
            for operator in operators:
                if operator.rhs.physics is not None:
                    self.physics = operator.rhs.physics
                    break

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array,
        e0_star: Array,
    ) -> tuple[float, Array]:
        y = state_array.flatten()

        for operator in self.operators:
            _, y = operator.advance(
                dt=dt,
                time=time,
                state_array=y,
                gamma_star=gamma_star,
                e0_star=e0_star,
            )

            # Optionally: Update the double flux variables
            if self.update_double_flux:
                assert self.physics is not None
                state = self.physics.conservative_to_primitive(y, gamma_star, e0_star)
                state.temperature = self.physics.get_temperature(state)
                gamma_star, e0_star = self.physics.get_double_flux_variables(state)

        return time + dt, y


class FastSlowIntegrator(TimeIntegrator):
    def __init__(
        self,
        rhs: RightHandSide,
        fast_integrator: type[TimeIntegrator] = ScipyIVP,
        slow_integrator: type[TimeIntegrator] = ForwardEuler,
    ) -> None:
        """Apply different integrators to stiff and non-stiff terms."""
        self.rhs = rhs
        self.fast_integrator = fast_integrator(self.rhs)
        self.slow_integrator = slow_integrator(self.rhs)

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array,
        e0_star: Array,
    ) -> tuple[float, Array]:
        assert isinstance(self.rhs, FastSlowSource)
        y_full = np.zeros_like(state_array)
        self.rhs.update_indices(time)

        # Integrate fast terms
        self.rhs.mode = "fast"
        _, y_fast = self.fast_integrator.advance(
            dt, time, state_array, gamma_star, e0_star
        )
        y_full = self.rhs.add_source(y_full, y_fast)

        # Integrate slow terms
        self.rhs.mode = "slow"
        _, y_slow = self.slow_integrator.advance(
            dt, time, state_array, gamma_star, e0_star
        )
        y_full = self.rhs.add_source(y_full, y_slow)

        return time + dt, y_full
