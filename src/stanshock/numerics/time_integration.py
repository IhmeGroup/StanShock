from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from stanshock.system.backend import Array, Unpack
from stanshock.system.base import FastSlowSource, RightHandSide

if TYPE_CHECKING:
    from scipy.integrate._ivp.ivp import _IVPMethod, _SolverOptions


class TimeIntegrator(ABC):
    def __init__(self, rhs: RightHandSide) -> None:
        self.rhs = rhs

    @abstractmethod
    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[float, Array, Array | None, Array | None]:
        """Integrate the state from `time` to `time+dt` given the rhs."""


class ScipyIVP(TimeIntegrator):
    def __init__(
        self,
        rhs: RightHandSide,
        method: _IVPMethod = "LSODA",
        **options: Unpack[_SolverOptions],
    ) -> None:
        """Interface to SciPy's IVP solvers.

        The specific solver can be selected with the `method` input, and additional
        solver options can be passed in as keyword arguments. See documentation for
        `scipy.integrate.solve_ivp`.
        """
        from scipy.integrate import solve_ivp

        self.integrator = solve_ivp

        self.rhs = rhs
        self.solver_options = options

        if self.rhs.jac is not None:
            self.solver_options["jac"] = self.rhs.jac
            self.solver_options["lband"] = 0
            self.solver_options["uband"] = 0
        self.method = method

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[float, Array, Array | None, Array | None]:
        y0, gamma_star_local, e0_star_local = self.rhs.before_time_integration(
            time, state_array, gamma_star, e0_star
        )

        results = self.integrator(
            fun=self.rhs.source,
            t_span=(time, time + dt),
            y0=y0,
            method=self.method,
            args=(gamma_star_local, e0_star_local),
            **self.solver_options,
        )

        time = results.t[-1]
        state_array, gamma_star, e0_star = self.rhs.after_time_integration(
            time=time,
            state_array_local=results.y[:, -1],
            gamma_star_local=gamma_star_local,
            e0_star_local=e0_star_local,
            state_array=state_array,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

        return time, state_array, gamma_star, e0_star


class RungeKuttaBase(TimeIntegrator):
    n_rk_stage: int = 0
    rk_coeff: Array = np.zeros((1, 4))

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[float, Array, Array | None, Array | None]:
        t = time
        y0, gamma_star_local, e0_star_local = self.rhs.before_time_integration(
            t, state_array, gamma_star, e0_star
        )

        t0 = t + 0
        y = y0.copy()

        for i_rk_stage in range(self.n_rk_stage):
            a, b, c, d = self.rk_coeff[i_rk_stage]

            dydt = self.rhs.source(
                time=t,
                state_array_local=y,
                gamma_star=gamma_star_local,
                e0_star=e0_star_local,
            )

            if b != 1.0:
                y = b * y

            if a != 0.0:
                y = y + a * y0

            y = self.rhs.add_source(y, c * dt * dydt)

            t = t0 + d * dt

        state_array, gamma_star, e0_star = self.rhs.after_time_integration(
            time=time,
            state_array_local=y,
            gamma_star_local=gamma_star_local,
            e0_star_local=e0_star_local,
            state_array=state_array,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )
        return t, state_array, gamma_star, e0_star


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
        self.transport_operator: TimeIntegrator = transport_operator
        self.reaction_operator: TimeIntegrator = reaction_operator

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[float, Array, Array | None, Array | None]:
        # Take half-step with transport terms
        _, state_array, _, _ = self.transport_operator.advance(
            dt=0.5 * dt,
            time=time,
            state_array=state_array,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

        # Take full-step with reaction terms
        t, state_array, _, _ = self.reaction_operator.advance(
            dt=dt,
            time=time,
            state_array=state_array,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

        # Take half-step with transport terms
        _, state_array, gamma_star, e0_star = self.transport_operator.advance(
            dt=0.5 * dt,
            time=time + 0.5 * dt,
            state_array=state_array,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

        return t, state_array, gamma_star, e0_star


class LieSplitting(TimeIntegrator):
    def __init__(
        self, operators: list[TimeIntegrator], update_double_flux: bool = False
    ) -> None:
        self.operators: list[TimeIntegrator] = operators
        self.update_double_flux: bool = update_double_flux

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[float, Array, Array | None, Array | None]:
        gamma_star_temp: Array | None = None
        e0_star_temp: Array | None = None
        for operator in self.operators:
            _, state_array, gamma_star_temp, e0_star_temp = operator.advance(
                dt=dt,
                time=time,
                state_array=state_array,
                gamma_star=gamma_star,
                e0_star=e0_star,
            )

            # Optionally: Update the double flux variables after each step
            if self.update_double_flux:
                gamma_star, e0_star = gamma_star_temp, e0_star_temp

        # Otherwise, only update the double flux at the end
        if not self.update_double_flux:
            gamma_star, e0_star = gamma_star_temp, e0_star_temp

        return time + dt, state_array, gamma_star, e0_star


class FastSlowIntegrator(TimeIntegrator):
    def __init__(
        self,
        rhs: RightHandSide,
        fast_integrator: type[TimeIntegrator] = ScipyIVP,
        slow_integrator: type[TimeIntegrator] = ForwardEuler,
    ) -> None:
        """Apply different integrators to stiff and non-stiff terms."""
        self.rhs: RightHandSide = rhs
        self.fast_integrator: TimeIntegrator = fast_integrator(self.rhs)
        self.slow_integrator: TimeIntegrator = slow_integrator(self.rhs)

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[float, Array, Array | None, Array | None]:
        assert isinstance(self.rhs, FastSlowSource)

        # Integrate fast terms
        self.rhs.mode = "fast"
        if len(state_array[self.rhs.idx_implicit]) > 0:
            _, state_array, gamma_star, e0_star = self.fast_integrator.advance(
                dt, time, state_array, gamma_star, e0_star
            )

        # Integrate slow terms
        self.rhs.mode = "slow"
        _, state_array, gamma_star, e0_star = self.slow_integrator.advance(
            dt, time, state_array, gamma_star, e0_star
        )

        return time + dt, state_array, gamma_star, e0_star
