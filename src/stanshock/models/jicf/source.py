from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import make_interp_spline

from stanshock.models.jicf.generate import JICModel
from stanshock.physics.flamelet import FPVTable
from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, TypeAlias, Unpack
from stanshock.system.base import PrecomputeSteps, RightHandSide
from stanshock.system.geometry import Box
from stanshock.utils.isentropic import (
    mach_from_pressure_ratio,
    mdot_from_pressure_ratio,
    property_ratios,
)

_ThrottleFunction: TypeAlias = Callable[[float], float]
_ThrottleFunctionLike: TypeAlias = _ThrottleFunction | tuple[Array, Array] | float


# Classes which turn scalars and arrays into spatiotemporal functions
class ConstantValue:
    def __init__(self, constant: float) -> None:
        self.constant: float = constant

    def __call__(self, _time: float) -> float:
        return self.constant


class LinearInterpolator:
    def __init__(self, time: Array, throttle: Array) -> None:
        self.xmin, self.xmax = float(time[0]), float(time[-1])
        self.ymin, self.ymax = float(throttle[0]), float(throttle[-1])
        self.interp = make_interp_spline(time, throttle, k=1)

    def __call__(self, time: float) -> float:
        if time <= self.xmin:
            return self.ymin
        if time >= self.xmax:
            return self.ymax
        return float(self.interp(time))


class FuelInjector(RightHandSide):
    """Source terms for a fuel injector based on a JICF model.

    The injected mass flow rate is driven by a runtime *throttle schedule*
    (``t_inj`` -> ``mdot_inj``/``u_inj``/``E_inj``), which is decoupled from the
    throttle-agnostic :class:`JICModel` tables. The original fixed-schedule
    behaviour is recovered by passing that schedule here. Table lookups are
    performed on the momentum-flux ratio ``J``, computed from the injected jet
    state and the crossflow inflow so that they remain correct if the inflow is
    allowed to vary.
    """

    def __init__(
        self,
        jicf: JICModel,
        mdot_max: float,
        throttle: _ThrottleFunctionLike = 1.0,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        """
        jicf: JICModel
            The underlying jet-in-crossflow model (tabulated over ``J``)
        throttle: Callable[[float], float] | tuple[Array, Array] | float
            Throttle setting between 0 and 1 as either a function of time, a tuple of
            (time, throttle) arrays to interpolate, or a constant value to use.
        geometry: Box
            The geometry object describing the mesh and cross-section
        physics: FPVTable
            The FPV table object, used for the chemical source terms
        """
        super().__init__(**precompute_steps)
        assert isinstance(self.geometry, Box)
        assert isinstance(self.physics, FPVTable)
        self.jicf = jicf
        self.mdot_max = mdot_max
        self.throttle = throttle

        # Restrict the injector source terms to the near-injector region
        L_src = 3e-2
        xf = self.geometry.xf
        self.idx_input = (
            np.where(
                np.logical_and(xf >= self.jicf.x_inj, xf < self.jicf.x_inj + L_src)
            )[0]
            + self.geometry.n_ghost_layers
        )
        # Cross flow properties defined just upstream of the injector
        self.idx_upstream: int = self.idx_input[0] - 1
        self.idx_input = np.concatenate(([self.idx_upstream], self.idx_input))
        self.shape_input = (len(self.idx_input), -1)
        self.shape_output = (len(self.idx_input), -1)

        # Extract some information about the geometry
        self.xc = self.geometry.xc[self.idx_input]
        self.cell_volumes: Array | None = None
        if self.geometry.dlnA_dt is None:
            # Constant volume
            self.cell_volumes = self.geometry.volume()[self.idx_input, None]

        # Weight injection distribution by cell widths - don't include upstream cell
        self.distribute_factor: Array | float = 1.0 / (len(self.idx_input) - 1)
        if isinstance(self.geometry.dx, np.ndarray):
            dx = self.geometry.dx[self.idx_input]
            self.distribute_factor = dx / np.sum(dx[1:])
            self.distribute_factor[0] = 0.0

        # Properties of the first injected fluid particle: [x, mdot, J, rho_inj/rho].
        # The mass flow rate (for reporting), the momentum-flux ratio (the table lookup
        # axis), and the fuel/air density ratio are convected with each parcel.
        self.fluid_tips = np.array([[self.jicf.x_inj, 0.0, 0.0, 1.0]])

    @property
    def throttle(self) -> _ThrottleFunction:
        return self._throttle

    @throttle.setter
    def throttle(self, throttle: _ThrottleFunctionLike) -> None:
        self._throttle: _ThrottleFunction
        if isinstance(throttle, float | int):
            self._throttle = ConstantValue(throttle)
        elif isinstance(throttle, tuple):
            self._throttle = LinearInterpolator(*throttle)
        else:
            self._throttle = throttle

    def inflow(self, t: float, state: FluidState) -> tuple[FluidState, float, float]:
        """Fluid properties at the plane of injection.

        Simple model: Use specified manifold temperature and crossflow pressure.
        """
        assert self.physics is not None
        inflow = FluidState(
            shape=(1,),
            temperature=np.asarray([self.jicf.T0]),
            pressure=np.mean(self.physics.get_pressure(state)[1:], keepdims=True),
            composition=self.jicf.manifold_state.composition,
        )
        rho = self.physics.get_density(inflow)
        mdot = self.mdot(t, state)
        inflow.velocity = mdot / (rho * self.jicf.n_inj * self.jicf.A_inj)
        J = self.J(state, inflow)
        return inflow, mdot, J

    def mdot(self, t: float, state: FluidState) -> float:
        """Injector mass flow rate at a given time and fluid state.

        Simple model: Throttle is simple percentage of specified maximum flow rate.
        """
        _ = state
        return self.throttle(t) * self.mdot_max

    def J(self, state: FluidState, inflow: FluidState) -> float:
        """Momentum-flux ratio ``J = rho_inj u_inj**2 / (rho u**2)`` at time ``t``."""
        assert self.physics is not None

        # Get cross flow momentum flux just upstream of the injector
        rho = float(self.physics.get_density(state)[0])
        u = float(self.physics.get_velocity(state)[0])

        # Get momentum flux at the injection plane
        rho_inj: float = float(self.physics.get_density(inflow)[0])
        u_inj = float(self.physics.get_velocity(inflow)[0])

        return (rho_inj * u_inj**2) / (rho * u**2)

    def update_fluid_tip_positions(
        self, dt: float, t: float, state: FluidState
    ) -> None:
        """
        This method updates the position of the fluid tips based on the velocity
        of the fluid.
        dt: float
            The time step
        t: float
            The current time
        state: FluidState
            The current state of the cross flow
        """
        assert self.geometry is not None
        assert self.physics is not None

        # Get inflow properties
        inflow, mdot, J = self.inflow(t, state[self.idx_input])
        rho_inj = float(self.physics.get_density(inflow)[0])

        # Get cross flow density just upstream of the injector
        rho = float(self.physics.get_density(state)[self.idx_upstream])

        # Update the fluid tip positions
        xc = self.geometry.xc
        u = self.physics.get_velocity(state)
        x = self.fluid_tips[:, 0]
        x += dt * make_interp_spline(xc, u, k=1)(x)

        # Drop fluid tips that have passed the end of the domain
        idx = x < xc[-1]

        # Emit a new fluid tip, carrying its mass flow rate, momentum-flux ratio, and density ratio
        next_tip = np.array([[self.jicf.x_inj, mdot, J, rho_inj / rho]])
        self.fluid_tips = np.concatenate([next_tip, self.fluid_tips[idx]], axis=0)

    def source_implementation(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Compute a fuel injector source term to target the desired mixture fraction profile."""
        _ = state_array_local, face_states, avg_face_states, face_gradients
        assert self.geometry is not None
        assert self.physics is not None
        assert state is not None
        rhs = np.zeros_like(state_array_local)

        inflow, mdot_total, _ = self.inflow(time, state)
        u_inj = self.physics.get_velocity(inflow)
        E_inj = self.physics.get_internal_energy(inflow) + 0.5 * u_inj**2

        mdot = mdot_total * self.distribute_factor

        # Compute the source term
        rhs[:, 0] = mdot * u_inj * np.cos(self.jicf.theta_inj)  # momentum
        rhs[:, 1] = mdot * E_inj  # total energy
        rhs[:, 2] = mdot  # density
        rhs[:, 3] = mdot  # mixture fraction
        # rhs[:, 4] = 0.0  # progress variable

        # Divide by the cell volumes
        if self.cell_volumes is None:
            assert self.geometry is not None
            vol = self.geometry.volume(time)[self.idx_input, None]
            return np.ravel(rhs / vol)

        return np.ravel(rhs / self.cell_volumes)


class NozzleInjector(FuelInjector):
    def mdot(self, t: float, state: FluidState) -> float:
        """Compute mdot from discharge coefficient."""
        assert self.physics is not None
        throttle = self.throttle(t)
        p0 = self.jicf.p0
        rho0 = self.jicf.rho0

        # Compute pressure drop across injector
        p = np.mean(self.physics.get_pressure(state))
        dp = p0 - p
        if dp <= 0.0:
            return 0.0
        return min(float(throttle * self.jicf.Ae * np.sqrt(rho0 * dp)), self.mdot_max)


class GasInjector(FuelInjector):
    """Injection computed using isentropic compressible flow relations."""

    def inflow(self, t: float, state: FluidState) -> tuple[FluidState, float, float]:
        """Fluid properties at the plane of injection.

        Simple model: Use specified manifold temperature and crossflow pressure.
        """
        assert self.physics is not None

        # Compute pressure drop across injector
        g = self.jicf.gamma0
        p0 = self.jicf.p0
        p = float(np.mean(self.physics.get_pressure(state)[1:]))
        pr = p / p0

        # Get mass flow rate
        mdot = self.mdot(t, state, pr)

        # Set inflow properties from isentropic flow relations
        mach = mach_from_pressure_ratio(pr, g)
        Tr, pr, rhor = property_ratios(mach, g)
        rho = self.jicf.rho0 * rhor

        inflow = FluidState(
            shape=(1,),
            temperature=np.asarray([self.jicf.T0 * Tr]),
            pressure=np.asarray([p0 * pr]),
            density=np.asarray([rho]),
            velocity=np.asarray([mdot / (rho * self.jicf.n_inj * self.jicf.A_inj)]),
            composition=self.jicf.manifold_state.composition,
        )
        J = self.J(state, inflow)
        return inflow, mdot, J

    def mdot(self, t: float, state: FluidState, pr: float | None = None) -> float:
        """Compute mdot from isentropic flow relations."""
        assert self.physics is not None
        throttle = self.throttle(t)
        g = self.jicf.gamma0
        p0 = self.jicf.p0
        T0 = self.jicf.T0
        R0 = self.jicf.R0

        p = (
            float(np.mean(self.physics.get_pressure(state)[1:]))
            if pr is None
            else pr * p0
        )
        mdot = mdot_from_pressure_ratio(p0, T0, R0, g, p, throttle * self.jicf.Ae)

        return min(mdot, self.mdot_max)


class JICFChemistrySource(RightHandSide):
    def __init__(
        self, injector: FuelInjector, **precompute_steps: Unpack[PrecomputeSteps]
    ) -> None:
        super().__init__(**precompute_steps)
        assert self.geometry is not None
        self.injector = injector
        xc = self.geometry.xc
        idx_inj = np.nonzero(xc > self.injector.jicf.x_inj)[0][0] - 1
        idx_noz = np.nonzero(xc > self.injector.jicf.x_noz)[0][0] + 1
        self.idx_input = slice(idx_inj, idx_noz)
        self.xc = xc[self.idx_input]

        # Reactions only directly affect progress variable
        self.idx_source = np.array([4])
        self.shape_input = (idx_noz - idx_inj, -1)
        self.shape_output = (idx_noz - idx_inj, 1)

    def source_implementation(
        self,
        time: float,
        state_array_local: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Compute the chemical source terms [1/s] using the FPV table."""
        _ = time, state_array_local, face_states, avg_face_states, face_gradients
        assert isinstance(self.physics, FPVTable)
        assert state is not None
        assert state.density is not None
        assert state.mixture_fraction is not None
        assert state.normalized_progress_variable is not None
        factor = self.physics.get_source_progress_variable_compressibility_factor(state)

        # Get the mixture fraction variance profile. The variance table is
        # indexed by momentum-flux ratio J (column 2 of the fluid tips), which
        # is convected with each injected parcel.
        result = make_interp_spline(
            self.injector.fluid_tips[:, 0], self.injector.fluid_tips[:, 2:], k=1
        )(self.xc)
        J = result[:, 0]
        rhor = result[:, 1]

        Zvar = rhor * self.injector.jicf.Z_var_profile_interp((self.xc, J))
        Zvar = np.maximum(Zvar, 10 ** self.injector.jicf.logsigma2_vec.min())

        return (
            factor
            * state.density
            * self.injector.jicf.omega_C_int_interp(
                (
                    state.mixture_fraction,
                    state.normalized_progress_variable,
                    np.log10(Zvar),
                )
            )
        )
