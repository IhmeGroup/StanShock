from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from scipy import integrate

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array, Index
from stanshock.system.base import RightHandSide

SpatioTemporalFunction: TypeAlias = Callable[[float, Array], Array | float]


# Classes which turn scalars and arrays into spatiotemporal functions
class ConstantValue:
    def __init__(self, constant: float | None) -> None:
        if constant is None:
            constant = 0.0
        self.constant: float = constant

    def __call__(self, _time: float = 0.0, _x: Array | None = None) -> float:
        return self.constant


class LinearInterpolator:
    def __init__(self, xp: Array, fp: Array, x_default: Array | None = None) -> None:
        self.xp = xp
        self.fp = fp

        if x_default is None:
            self.x = xp
        else:
            self.x = x_default

    def __call__(self, _time: float = 0.0, x: Array | None = None) -> Array:
        if x is None:
            x = self.x
        return np.interp(x=x, xp=self.xp, fp=self.fp)


class Geometry:
    def __init__(
        self,
        x: Array,
        area: SpatioTemporalFunction | Array | float,
        perimeter: SpatioTemporalFunction | Array | float,
        dlnA_dt: SpatioTemporalFunction | None = None,
        dlnA_dx: SpatioTemporalFunction | None = None,
    ) -> None:
        self.x: Array = x
        self.n: int = len(self.x)
        self.dx: float = self.x[1] - self.x[0]

        # Turn constant-value areas+perimeters into functions of t and x
        self.area: SpatioTemporalFunction
        self.perimeter: SpatioTemporalFunction

        if isinstance(area, float | int):
            self.area = ConstantValue(constant=area)
        elif isinstance(area, np.ndarray):
            self.area = LinearInterpolator(xp=self.x, fp=area)
        else:
            self.area = area

        if isinstance(perimeter, float | int):
            self.perimeter = ConstantValue(constant=perimeter)
        elif isinstance(perimeter, np.ndarray):
            self.perimeter = LinearInterpolator(xp=self.x, fp=perimeter)
        else:
            self.perimeter = perimeter

        # Set up area derivatives
        self.dlnA_dx: SpatioTemporalFunction | None
        self.dlnA_dt: SpatioTemporalFunction | None = dlnA_dt

        if dlnA_dx is None:
            if isinstance(area, float | int):
                self.dlnA_dx = None
            elif isinstance(area, np.ndarray):
                x_midpoint: Array = 0.5 * (self.x[1:] + self.x[:-1])
                area_midpoint: Array = 0.5 * (area[1:] + area[:-1])
                dlnA_dx_fd: Array = np.diff(area) / (np.diff(self.x) * area_midpoint)
                self.dlnA_dx = LinearInterpolator(
                    xp=x_midpoint, fp=dlnA_dx_fd, x_default=self.x
                )
            else:
                msg = "Cannot (yet) automatically determine dlnA_dx from callable area."
                raise NotImplementedError(msg)
        else:
            self.dlnA_dx = dlnA_dx

    def hydraulic_diameter(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.x

        return 4.0 * self.area(time, x) / self.perimeter(time, x)

    def characteristic_length(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.x

        return self.hydraulic_diameter(time, x)


class Cylinder(Geometry):
    def __init__(
        self,
        x: Array,
        d_outer: SpatioTemporalFunction | Array | float,
        d_inner: SpatioTemporalFunction | Array | float | None = None,
        dlnA_dt: SpatioTemporalFunction | None = None,
        dlnA_dx: SpatioTemporalFunction | None = None,
    ) -> None:
        # Set up functional form of inner and outer diameters
        self.d_outer: SpatioTemporalFunction
        self.d_inner: SpatioTemporalFunction

        if isinstance(d_outer, float | int):
            self.d_outer = ConstantValue(constant=d_outer)
        elif isinstance(d_outer, np.ndarray):
            self.d_outer = LinearInterpolator(xp=self.x, fp=d_outer)
        else:
            self.d_outer = d_outer

        if d_inner is None:
            d_inner = 0.0

        if isinstance(d_inner, float | int):
            self.d_inner = ConstantValue(constant=d_inner)
        elif isinstance(d_inner, np.ndarray):
            self.d_inner = LinearInterpolator(xp=self.x, fp=d_inner)
        else:
            self.d_inner = d_inner

        # Set up functional forms of area and perimeter
        area: SpatioTemporalFunction | Array | float = self._area
        perimeter: SpatioTemporalFunction | Array | float = self._perimeter

        if isinstance(d_outer, np.ndarray | float | int) and isinstance(
            d_inner, np.ndarray | float | int
        ):
            area = 0.25 * np.pi * (d_outer**2 - d_inner**2)
            perimeter = 0.5 * np.pi * (d_outer + d_inner)

        super().__init__(x, area, perimeter, dlnA_dt, dlnA_dx)

    def _area(self, t: float, x: Array) -> Array | float:
        return 0.25 * np.pi * (self.d_outer(t, x) ** 2 - self.d_inner(t, x) ** 2)

    def _perimeter(self, t: float, x: Array) -> Array | float:
        return 0.5 * np.pi * (self.d_outer(t, x) + self.d_inner(t, x))

    def hydraulic_diameter(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.x

        return self.d_outer(time, x) - self.d_inner(time, x)

    def characteristic_length(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.x

        d_outer: Array | float = self.d_outer(time, x)
        d_inner: Array | float = self.d_inner(time, x)
        characteristic_length: Array | float = d_outer - d_inner

        if isinstance(d_inner, float | int) and d_inner > 0:
            characteristic_length *= 0.5
        else:
            assert isinstance(characteristic_length, np.ndarray)
            characteristic_length[d_inner > 0] *= 0.5

        return characteristic_length


class Box(Geometry):
    def __init__(
        self,
        x: Array,
        h: SpatioTemporalFunction | Array | float,
        w: SpatioTemporalFunction | Array | float,
        dlnA_dt: SpatioTemporalFunction | None = None,
        dlnA_dx: SpatioTemporalFunction | None = None,
    ) -> None:
        # Set up functional forms of height and width
        self.h: SpatioTemporalFunction
        self.w: SpatioTemporalFunction

        if isinstance(h, float | int):
            self.h = ConstantValue(constant=h)
        elif isinstance(h, np.ndarray):
            self.h = LinearInterpolator(xp=self.x, fp=h)
        else:
            self.h = h

        if isinstance(w, float | int):
            self.w = ConstantValue(constant=w)
        elif isinstance(w, np.ndarray):
            self.w = LinearInterpolator(xp=self.x, fp=w)
        else:
            self.w = w

        # Set up functional forms of area and perimeter
        area: SpatioTemporalFunction | Array | float = self._area
        perimeter: SpatioTemporalFunction | Array | float = self._perimeter

        if isinstance(h, np.ndarray | float | int) and isinstance(
            w, np.ndarray | float | int
        ):
            area = h * w
            perimeter = 2 * (h + w)

        super().__init__(x, area, perimeter, dlnA_dt, dlnA_dx)

    def _area(self, t: float, x: Array) -> Array | float:
        return self.h(t, x) * self.w(t, x)

    def _perimeter(self, t: float, x: Array) -> Array | float:
        return 2.0 * (self.h(t, x) + self.w(t, x))

    def hydraulic_diameter(
        self, time: float = 0.0, x: Array | None = None
    ) -> Array | float:
        if x is None:
            x = self.x

        h: Array | float = self.h(time, x)
        w: Array | float = self.w(time, x)
        return 2 * h * w / (h + w)


class AreaChange(RightHandSide):
    def __init__(self, geometry: Geometry) -> None:
        self.geometry: Geometry = geometry
        self.integrator = integrate.ode(f=self.source_fast).set_integrator(name="lsoda")

        # Define global indices
        self.idx_locations = np.s_[:]
        self.idx_source_terms = np.s_[:]

        # For geometries with no area change, replace the source term with a no-op
        self.no_area_change: bool = (
            self.geometry.dlnA_dt is None and self.geometry.dlnA_dx is None
        )

    def source(
        self,
        time: float,
        state_array: Array,
        physics: FluidPhysics,
        gamma_star: Array,
        dt: float,
    ) -> Array:
        if self.no_area_change:
            return np.zeros((1,))

        # Compute the density from the state array
        rY0 = state_array[:, 2:]
        r0 = rY0[..., : physics.n_scalars_rho_sum].sum(axis=-1)
        Y0 = rY0 / r0[..., None]

        state0_compact = np.zeros((state_array.shape[0], 3))
        state0_compact[:, 0] = r0
        state0_compact[:, 1:] = state_array[:, :2]  # ru and rE

        # Divide domain between explicit and implicit source terms
        idx_explicit: Index = np.arange(self.geometry.x.shape[0], dtype=np.int64)
        assert isinstance(idx_explicit, np.ndarray)
        idx_implicit: Index = np.array([], dtype=np.int64)
        assert isinstance(idx_implicit, np.ndarray)
        if self.geometry.dlnA_dt is not None:
            dlnA_dt: Array | float = self.geometry.dlnA_dt(time, self.geometry.x)
            assert isinstance(dlnA_dt, np.ndarray)
            idx_implicit = np.where(dlnA_dt != 0.0)[0]
            idx_explicit = np.where(dlnA_dt == 0.0)[0]

        # Integrate fast terms implicitly
        rhs: Array = np.zeros_like(
            state_array[self.idx_locations, self.idx_source_terms]
        )
        for i in idx_implicit:
            # Initialize
            y0: Array = state0_compact[i, :].copy()
            args: tuple[float, float] = self.geometry.x[i], gamma_star[i]
            self.integrator.set_initial_value(y=y0, t=time)
            self.integrator.set_f_params(args)

            # Solve
            self.integrator.integrate(t=time + dt)

            # Store RHS source term
            rhs_compact: Array = (self.integrator.y - state0_compact[i, :]) / dt
            rhs[i, 0:2] += rhs_compact[1:]  # ru and rE
            rhs[i, 2:] += rhs_compact[0] * Y0[i, :]  # rY sources

        # Add slow source terms
        state: FluidState = physics.conservative_to_primitive(
            state_array, gamma=gamma_star
        )
        rhs_compact = self.source_slow(
            time=time, state0_compact=state0_compact, state=state, idx=idx_explicit
        )
        rhs[idx_explicit, 0:2] += rhs_compact[idx_explicit, 1:]  # ru and rE
        rhs[idx_explicit, 2:] += (
            rhs_compact[idx_explicit, 0:1] * Y0[idx_explicit, :]
        )  # rY sources

        return rhs

    def source_slow(
        self, time: float, state0_compact: Array, state: FluidState, idx: Index
    ) -> Array:
        """Area change contributions to RHS."""
        rhs_compact: Array = np.zeros_like(state0_compact[idx])
        x: Array = self.geometry.x[idx]

        if self.geometry.dlnA_dt is not None:
            dlnA_dt: Array = self.geometry.dlnA_dt(time, x)
            rhs_compact -= state0_compact[idx, :] * dlnA_dt

        if self.geometry.dlnA_dx is not None:
            assert state.pressure is not None
            assert state.velocity is not None

            dlnA_dx: Array = self.geometry.dlnA_dx(time, x)
            rhs_compact[:, 0] -= state0_compact[idx, 1] * dlnA_dx
            rhs_compact[:, 1] -= (
                state0_compact[idx, 1] ** 2.0 / state0_compact[idx, 0]
            ) * dlnA_dx
            rhs_compact[:, 2] -= (
                state.velocity[idx] * (state0_compact[idx, 2] + state.pressure[idx])
            ) * dlnA_dx

        return rhs_compact

    def source_fast(self, time: float, y: Array, args: tuple[float, float]) -> Array:
        """Fast source terms for quasi-1D geometry."""
        # Unpack the input and initialize
        x0, gamma = args
        x: Array = np.array([x0])
        r, ru, rE = y
        p: float = (gamma - 1.0) * (rE - 0.5 * ru**2.0 / r)
        rhs_compact: Array = np.zeros(3)

        # create quasi-1D right hand side
        if self.geometry.dlnA_dt is not None:
            dlnA_dt: Array = self.geometry.dlnA_dt(time, x)
            rhs_compact[0] -= r * dlnA_dt
            rhs_compact[1] -= ru * dlnA_dt
            rhs_compact[2] -= rE * dlnA_dt

        if self.geometry.dlnA_dx is not None:
            dlnA_dx: Array = self.geometry.dlnA_dx(time, x)
            rhs_compact[0] -= ru * dlnA_dx
            rhs_compact[1] -= (ru**2.0 / r) * dlnA_dx
            rhs_compact[2] -= (ru / r * (rE + p)) * dlnA_dx

        return rhs_compact
