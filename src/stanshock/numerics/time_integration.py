from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from stanshock.physics.fluid_base import FluidPhysics
from stanshock.system.backend import Array
from stanshock.system.base import RightHandSide


class TimeIntegrator(ABC):
    def __init__(self, rhs: RightHandSide) -> None:
        self.rhs = rhs

    @abstractmethod
    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        physics: FluidPhysics,
        gamma_star: Array,
    ) -> tuple[float, Array]:
        """Integrate the state from `time` to `time+dt` given the rhs."""


class ScipyIVP(TimeIntegrator):
    def __init__(self, rhs: RightHandSide, method: str="LSODA", **kwargs) -> None:
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
        physics: FluidPhysics,
        gamma_star: Array,
    ) -> tuple[float, Array]:
        # integrator = self.integrator(self.rhs.source).set_integrator("lsoda")
        # integrator.set_initial_value(y=state_array, t=time)
        # integrator.set_f_params(args=(physics, gamma_star))
        # integrator.integrate(t=time + dt)

        results = self.integrator(
            fun=self.rhs.source,
            t_span=(time, time+dt),
            y0=state_array.flatten(),
            method=self.method,
            args=(physics, gamma_star),
            **self.solver_options,
        )

        return results.t[-1], results.y[:, -1]


class RungeKuttaBase(TimeIntegrator):
    n_rk_stage: int = 0
    rk_coeff: Array = np.zeros((1, 4))

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        physics: FluidPhysics,
        gamma_star: Array,
    ) -> tuple[float, Array]:
        t = time
        y = state_array.flatten()

        t0 = t + 0
        y0 = y.copy()

        for i_rk_stage in range(self.n_rk_stage):
            a, b, c, d = self.rk_coeff[i_rk_stage]

            dydt = self.rhs.source(
                time=t,
                state_array=state_array,
                physics=physics,
                gamma_star=gamma_star,
            )

            if b != 1.0:
                y *= b

            if a != 0.0:
                y += a * y0

            y += c * dt * dydt

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
        physics: FluidPhysics,
        gamma_star: Array,
    ) -> tuple[float, Array]:
        y = state_array.flatten()

        # Take half-step with transport terms
        _, y = self.transport_operator.advance(
            dt=0.5 * dt,
            time=time,
            state_array=y,
            physics=physics,
            gamma_star=gamma_star,
        )

        # Take full-step with reaction terms
        t, y = self.reaction_operator.advance(
            dt=dt,
            time=time,
            state_array=y,
            physics=physics,
            gamma_star=gamma_star,
        )

        # Take half-step with transport terms
        _, y = self.transport_operator.advance(
            dt=0.5 * dt,
            time=time + 0.5 * dt,
            state_array=y,
            physics=physics,
            gamma_star=gamma_star,
        )

        return t, y


class LieSplitting(TimeIntegrator):
    def __init__(self, operators: list[TimeIntegrator], update_gamma_star: bool=False) -> None:
        self.operators = operators
        self.update_gamma_star = update_gamma_star

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        physics: FluidPhysics,
        gamma_star: Array,
    ) -> tuple[float, Array]:
        y = state_array.flatten()

        for operator in self.operators:
            _, y = operator.advance(
                dt=dt, time=time, state_array=y, physics=physics, gamma_star=gamma_star
            )

            # Optionally: Update the gamma_star value
            if self.update_gamma_star:
                state = physics.conservative_to_primitive(y, gamma_star)
                state.temperature = physics.get_temperature(state)
                gamma_star = physics.get_gamma(state)

        return time+dt, y
