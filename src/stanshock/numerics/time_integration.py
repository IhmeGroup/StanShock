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
            fun=self.rhs.source_full,
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


class SimpleRungeKuttaBase(TimeIntegrator):
    """Simplified form for explicit Runge-Kutta approaches.

    This base class implements a simplified form of the Runge-Kutta equations,
    where each stage requires only the initial value, `y0`, the current `y`
    value computed by the previous stage, and the corresponding gradient. This
    is in contrast to the standard approach in which gradients from all prior
    steps must be retained.

    Coefficients `(a, b, c, d)` determine he update from stage `i` to `i+1` as:

    ```
    y[i+1] = a*y0 + b*y[i] + c*dt*rhs(t[i], y[i])
    t[i+1] = t0 + d*dt
    ```

    Note that not all explicit Runge-Kutta methods can be expressed in this
    format. Each approach is implemented via the following class attributes:

    `n_rk_stage`: Number of stages.
    `rk_coeff`: Array of shape `(n_rk_stage, 4)`, where each row provides
                the coefficients `(a, b, c, d)` for its corresponding stage.
    """

    n_rk_stage: int = 0
    rk_coeff: Array = np.zeros((0, 4))

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[float, Array, Array | None, Array | None]:
        t0 = time
        y0, gamma_star_local, e0_star_local = self.rhs.before_time_integration(
            t0, state_array, gamma_star, e0_star
        )

        t = t0 + 0
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


class ForwardEuler(SimpleRungeKuttaBase):
    n_rk_stage: int = 1
    rk_coeff: Array = np.array([[0.0, 1.0, 1.0, 1.0]])


class MidpointMethod(SimpleRungeKuttaBase):
    n_rk_stage: int = 2
    rk_coeff: Array = np.array([[0.0, 1.0, 0.5, 0.5], [1.0, 0.0, 1.0, 1.0]])


class HeunsMethod(SimpleRungeKuttaBase):
    n_rk_stage: int = 2
    rk_coeff: Array = np.array([[0.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 1.0]])


class SSPRK3(SimpleRungeKuttaBase):
    """
    3rd-order strong stability preserving Runge Kutta (SSPRK3). This scheme is
    stable for CFL <= 1.

    References
    ----------
    Dale E. Durran, "Numerical Methods for Fluid Dynamics", Springer.
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


class RungeKuttaBase(TimeIntegrator):
    """Explicit Runge-Kutta methods using Butcher tableau.

    Simply set the following class attributes on a derived class:

    `n_rk_stage`: Number of stages.
    `butcher`: Butcher tableau for the method with `n_rk_stage + 1` rows and
               `n_rk_stage` columns. The extra row stores the `b` coefficients.
    `c`: The `c` coefficients, which determine the intermediate time steps.
    """

    n_rk_stage: int = 0
    butcher: Array = np.zeros((n_rk_stage + 1, n_rk_stage))
    c: Array = np.zeros((n_rk_stage + 1,))

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[float, Array, Array | None, Array | None]:
        t0 = time
        y0, gamma_star_local, e0_star_local = self.rhs.before_time_integration(
            t0, state_array, gamma_star, e0_star
        )

        t = t0 + 0
        y = y0.copy()
        k = []

        for i_rk_stage in range(1, self.n_rk_stage + 1):
            # Compute gradient from previous stage
            k_stage = self.rhs.source(
                time=t,
                state_array_local=y,
                gamma_star=gamma_star_local,
                e0_star=e0_star_local,
            )
            k += [k_stage]

            # Update state and time for current stage
            t = t0 + self.c[i_rk_stage] * dt
            y = y0.copy()
            for j in range(i_rk_stage):
                y = self.rhs.add_source(y, self.butcher[i_rk_stage, j] * k[j] * dt)

        state_array, gamma_star, e0_star = self.rhs.after_time_integration(
            time=t,
            state_array_local=y,
            gamma_star_local=gamma_star_local,
            e0_star_local=e0_star_local,
            state_array=state_array,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )
        return t, state_array, gamma_star, e0_star


class RK4(RungeKuttaBase):
    """ "Classic" 4th-order Runge-Kutta scheme."""

    n_rk_stage: int = 4
    butcher: Array = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
        ]
    )
    c: Array = np.array([0.0, 0.5, 0.5, 1.0, 1.0])


class OperatorSplitting(TimeIntegrator):
    """Base class for operator-splitting approaches.

    Takes in a tuple of two or more operators (`TimeIntegrator` objects) to be
    integrated together using an operator-splitting approach.

    In each stage of the approach multiple operators are applied sequentially,
    and in some approaches the results across multiple stages are averaged
    together for improved accuracy or stability.
    """

    stages: tuple[tuple[tuple[int, float], ...], ...] = ()
    stage_coeffs: tuple[float, ...] = ()

    def __init__(
        self,
        operators: tuple[TimeIntegrator, ...],
        update_double_flux: bool = False,
    ) -> None:
        self.operators = operators
        self.update_double_flux: bool = update_double_flux

    def advance(
        self,
        dt: float,
        time: float,
        state_array: Array,
        gamma_star: Array | None,
        e0_star: Array | None,
    ) -> tuple[float, Array, Array | None, Array | None]:
        # Initialize final values
        state_array_final = np.zeros_like(state_array)
        gamma_star_final = np.zeros_like(gamma_star) if gamma_star is not None else None
        e0_star_final = np.zeros_like(e0_star) if e0_star is not None else None

        # Initialize intermediate double-flux variables
        gamma_star_stage: Array | None = None
        e0_star_stage: Array | None = None
        gamma_star_temp: Array | None = None
        e0_star_temp: Array | None = None

        for istage, stage in enumerate(self.stages):
            state_array_stage = state_array.copy()
            if gamma_star is not None and e0_star is not None:
                gamma_star_stage = gamma_star.copy()
                e0_star_stage = e0_star.copy()

            for ioperator, cdt in stage:
                advance = self.operators[ioperator].advance

                _, state_array_stage, gamma_star_temp, e0_star_temp = advance(
                    dt=cdt * dt,
                    time=time,
                    state_array=state_array_stage,
                    gamma_star=gamma_star_stage,
                    e0_star=e0_star_stage,
                )

                # Optionally: Update the double flux variables after each step
                if self.update_double_flux:
                    gamma_star_stage, e0_star_stage = gamma_star_temp, e0_star_temp

            # Otherwise, only update the double flux at the end
            if not self.update_double_flux:
                gamma_star_stage, e0_star_stage = gamma_star_temp, e0_star_temp

            # Accumulate data from each stage
            coeff = self.stage_coeffs[istage]
            state_array_final = state_array_final + coeff * state_array_stage

            if gamma_star_stage is not None and gamma_star_final is not None:
                gamma_star_final = gamma_star_final + coeff * gamma_star_stage

            if e0_star_stage is not None and e0_star_final is not None:
                e0_star_final = e0_star_final + coeff * e0_star_stage

        return time + dt, state_array_final, gamma_star_final, e0_star_final


class StrangSplitting(OperatorSplitting):
    """Classical second-order operator-splitting approach by Strang.

    In this approach, the first operator is integrated to the midpoint, then
    the second operator is integrated for the full time step, and finally the
    first operator is integrated from the midpoint to the end.
    """

    stages = (((0, 0.5), (1, 1.0), (0, 0.5)),)
    stage_coeffs = (1.0,)


class SymmetricallyWeightedSequentialSplitting(OperatorSplitting):
    """Second-order operator splitting approach from Csomós et. al [1].

    This approach applies Strang splitting twice, swapping the operators and
    then averaging the results together to make the approach symmetric.

    References
    ----------
    [1] Csomós et al. 2005. "Weighted sequential splittings and their analysis."
        Comput. Math. with Appl. 50, 7 (2005), 1017-1031.
    """

    stages = (
        ((0, 0.5), (1, 1.0), (0, 0.5)),
        ((1, 0.5), (0, 1.0), (1, 0.5)),
    )
    stage_coeffs = (0.5, 0.5)


class ThirdOrderSplitting(OperatorSplitting):
    """Third-order operator splitting approach from Jia and Li [1].

    Essentially a weighted combination of a symmetric Lie-splitting and symmetric
    Strang-splitting approach. Note that third-order accuracy has not been
    observed in practice with this scheme, so there may be some error in its
    implementation or constraints on achieving third-order convergence.

    References
    ----------
    [1] Jia et al. 2011. "A third accurate operator splitting method."
        Math. Comput. Model. 53, 1-2 (2011), 387-396.
    """

    stages: tuple[tuple[tuple[int, float], ...], ...] = (
        ((0, 0.5), (1, 1.0), (0, 0.5)),
        ((1, 0.5), (0, 1.0), (1, 0.5)),
        ((0, 1.0), (1, 1.0)),
        ((1, 1.0), (0, 1.0)),
    )
    stage_coeffs: tuple[float, ...] = (-1.0 / 6.0, -1.0 / 6.0, 2.0 / 3.0, 2.0 / 3.0)


class LieSplitting(OperatorSplitting):
    """First-order operator splitting approach by Lie.

    Simplest splitting approach, in which each term is integrated in sequence.
    """

    def __init__(
        self,
        operators: tuple[TimeIntegrator, ...],
        update_double_flux: bool = False,
    ) -> None:
        self.operators = operators
        self.update_double_flux: bool = update_double_flux

        n_operators = len(self.operators)
        self.stages = (tuple((i, 1.0) for i in range(n_operators)),)
        # self.stage_coeffs = tuple(i for i in range(n_operators))
        self.stage_coeffs = (1.0,)


class StrangSplittingOld(TimeIntegrator):
    """Deprecated implementation of Strang splitting.

    This specifies a transport and reaction operator, where the Strang approach
    is more general than this. In fact, the order of the operators can be safely
    swapped; however, there have been publications [1,2] which indicate lower
    error is achieved when the stiff terms (the reactions) take the half steps.

    References
    ----------
    [1] Ren et al. 2014. "Dynamic adaptive chemistry with operator splitting
        schemes for reactive flow simulations." J. Comput. Phys. 263,
        (April 2014), 19-36.
    [2] Sportisse. 2000. "An Analysis of Operator Splitting Techniques in the
        Stiff Case." J. Comput. Phys. 161, 1 (2000), 140-168.
    """

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
        # Take half-step with reaction terms
        _, state_array, _, _ = self.reaction_operator.advance(
            dt=0.5 * dt,
            time=time,
            state_array=state_array,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

        # Take full-step with transport terms
        t, state_array, _, _ = self.transport_operator.advance(
            dt=dt,
            time=time,
            state_array=state_array,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

        # Take half-step with reaction terms
        _, state_array, gamma_star, e0_star = self.reaction_operator.advance(
            dt=0.5 * dt,
            time=time + 0.5 * dt,
            state_array=state_array,
            gamma_star=gamma_star,
            e0_star=e0_star,
        )

        return t, state_array, gamma_star, e0_star


class LieSplittingOld(TimeIntegrator):
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
        rhs: FastSlowSource,
        fast_integrator: type[TimeIntegrator] = ScipyIVP,
        slow_integrator: type[TimeIntegrator] = MidpointMethod,
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

        # Integrate slow terms
        self.rhs.mode = "slow"
        _, state_array, gamma_star, e0_star = self.slow_integrator.advance(
            dt, time, state_array, gamma_star, e0_star
        )

        # Integrate fast terms
        self.rhs.mode = "fast"
        if len(state_array[self.rhs.idx_output_implicit]) > 0:
            _, state_array, gamma_star, e0_star = self.fast_integrator.advance(
                dt, time, state_array, gamma_star, e0_star
            )

        return time + dt, state_array, gamma_star, e0_star
