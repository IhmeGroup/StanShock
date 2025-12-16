from __future__ import annotations

from abc import abstractmethod

import cantera as ct
import numpy as np

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.backend import Array
from stanshock.system.geometry import Geometry
from stanshock.utils.isentropic import mach_from_area_ratio, property_ratios


def smoothing_function(
    x: Array, xShock: float, Delta: float, phiLeft: float, phiRight: float
) -> Array:
    """Returns the variable smoothed over the interface.

    inputs:
        x = numpy array of cell centers
        phiLeft = the value of the variable on the left side
        phiRight = the value of the variable on the right side
        xShock = the mean of the shock location
    """
    dphidx = (phiRight - phiLeft) / Delta
    phi = (phiLeft + phiRight) / 2.0 + dphidx * (x - xShock)
    phi[x < (xShock - Delta / 2.0)] = phiLeft
    phi[x > (xShock + Delta / 2.0)] = phiRight
    return phi


def smoothing_function_gradient(
    x: Array, xShock: float, Delta: float, phiLeft: float, phiRight: float
) -> Array:
    """Returns the derivative of the smoothing function.

    inputs:
        x = numpy array of cell centers
        phiLeft = the value of the variable on the left side
        phiRight = the value of the variable on the right side
        xShock = the mean of the shock location
    """
    dphidx = np.full_like(x, (phiRight - phiLeft) / Delta)
    dphidx[x < (xShock - Delta / 2.0)] = 0.0
    dphidx[x > (xShock + Delta / 2.0)] = 0.0
    return dphidx


class Initialization:
    def __init__(self, geometry: Geometry, physics: FluidPhysics) -> None:
        self.geometry = geometry
        self.physics = physics

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @geometry.setter
    def geometry(self, geometry: Geometry) -> None:
        self._geometry = geometry

    @abstractmethod
    def __call__(self) -> FluidState:
        """Compute an initial state for the fluid."""


class InitializeConstant(Initialization):
    def __init__(
        self,
        geometry: Geometry,
        physics: FluidPhysics,
        gas: ct.Solution,
        u: float,
    ) -> None:
        """Initializes a constant state.

        inputs:
            gas = Cantera solution object at the desired thermodynamic state
            u = velocity
        """
        super().__init__(geometry, physics)

        # Initialize state
        self.density: float = gas.density_mass
        self.pressure: float = gas.P
        self.gamma: float = gas.cp / gas.cv
        self.composition = self.physics.get_composition(gas.Y[None, :])
        self.u: float = u

    def __call__(self) -> FluidState:
        n = self.geometry.n_cells
        ones = np.ones(n)

        return FluidState(
            shape=(n,),
            density=ones * self.density,
            velocity=ones * self.u,
            pressure=ones * self.pressure,
            gamma=ones * self.gamma,
            composition=np.tile(self.composition, (n, 1)),
        )


class InitializeRiemannProblem(Initialization):
    def __init__(
        self,
        geometry: Geometry,
        physics: FluidPhysics,
        left_state: tuple[ct.Solution, float],
        right_state: tuple[ct.Solution, float],
        shock_location: float,
    ) -> None:
        """Initialize a Riemann Problem.

        inputs:
            left_state = a tuple containing the Cantera solution object at the
                            the desired thermodynamic state and the velocity:
                            (canteraSolution,u)
            right_state = a tuple containing the Cantera solution object at the
                            the desired thermodynamic state and the velocity:
                            (canteraSolution,u)
            shock_location = x-location of the shock within the domain
        """
        gas = physics.gas
        left_gas, u_left = left_state
        right_gas, u_right = right_state
        if (
            left_gas.species_names != gas.species_names
            or right_gas.species_names != gas.species_names
        ):
            msg = "Input gasses must be the same as the initialized gas."
            raise Exception(msg)

        self.shock_location = shock_location
        self.left = InitializeConstant(geometry, physics, left_gas, u_left)
        self.right = InitializeConstant(geometry, physics, right_gas, u_right)
        super().__init__(geometry, physics)

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @geometry.setter
    def geometry(self, geometry: Geometry) -> None:
        self._geometry = geometry
        self.left.geometry = geometry
        self.right.geometry = geometry

    def __call__(self) -> FluidState:
        # Initialize with left state
        state = self.left()
        assert state.density is not None
        assert state.velocity is not None
        assert state.pressure is not None
        assert state.composition is not None
        assert state.gamma is not None

        # Override with right state
        state_right = self.right()
        assert state_right.density is not None
        assert state_right.velocity is not None
        assert state_right.pressure is not None
        assert state_right.composition is not None
        assert state_right.gamma is not None

        index = np.where(self.geometry.xc >= self.shock_location)[0]
        state.density[index] = state_right.density[index]
        state.velocity[index] = state_right.velocity[index]
        state.pressure[index] = state_right.pressure[index]
        state.composition[index, :] = state_right.composition[index, :]
        state.gamma[index] = state_right.gamma[index]

        return state


class InitializeDiffuseInterface(Initialization):
    def __init__(
        self,
        geometry: Geometry,
        physics: FluidPhysics,
        left_state: tuple[ct.Solution, float],
        right_state: tuple[ct.Solution, float],
        shock_location: float,
        delta_smoothing: float,
    ) -> None:
        """Initialize an interface smoothed over a distance.

        inputs:
            left_state = a tuple containing the Cantera solution object at the
                        the desired thermodynamic state and the velocity:
                        (canteraSolution,u)
            right_state = a tuple containing the Cantera solution object at the
                        the desired thermodynamic state and the velocity:
                        (canteraSolution,u)
            shock_location = x-location of the shock within the domain
            delta_smoothing = distance over which the interface is smoothed linearly
        """
        super().__init__(geometry, physics)
        gas = physics.gas
        left_gas, u_left = left_state
        right_gas, u_right = right_state
        if (
            left_gas.species_names != gas.species_names
            or right_gas.species_names != gas.species_names
        ):
            msg = "Input gasses must be the same as the initialized gas."
            raise Exception(msg)

        self.shock_location = shock_location
        self.delta_smoothing = delta_smoothing
        self.left = (
            left_gas.density_mass,
            u_left,
            left_gas.P,
            left_gas.cp / left_gas.cv,
            physics.get_composition(left_gas.Y),
        )
        self.right = (
            right_gas.density_mass,
            u_right,
            right_gas.P,
            right_gas.cp / right_gas.cv,
            physics.get_composition(right_gas.Y),
        )

    def __call__(self) -> FluidState:
        # Smooth transition between left and right states
        xc = self.geometry.xc
        n_cells = self.geometry.n_cells
        n_scalars = self.physics.n_scalars

        rl, ul, pl, gl, Yl = self.left
        rr, ur, pr, gr, Yr = self.right
        r = smoothing_function(xc, self.shock_location, self.delta_smoothing, rl, rr)
        u = smoothing_function(xc, self.shock_location, self.delta_smoothing, ul, ur)
        p = smoothing_function(xc, self.shock_location, self.delta_smoothing, pl, pr)
        g = smoothing_function(xc, self.shock_location, self.delta_smoothing, gl, gr)

        composition = np.zeros((n_cells, n_scalars))
        for kSp in range(n_scalars):
            composition[:, kSp] = smoothing_function(
                xc, self.shock_location, self.delta_smoothing, Yl[kSp], Yr[kSp]
            )

        return FluidState(
            shape=(n_cells,),
            density=r,
            pressure=p,
            velocity=u,
            gamma=g,
            composition=composition,
        )


class InitializeIsentropic(Initialization):
    def __init__(
        self,
        geometry: Geometry,
        physics: FluidPhysics,
        inflow_state: ct.Solution,
        throat_area: float | None = None,
        subsonic_inflow: bool = True,
        subsonic_outflow: bool = False,
    ) -> None:
        """Applies isentropic flow relations to set initial conditions.

        Note that this formulation is only valid for a flow with constant specific
        heat ratio. It could be extended for non-ideal gas equations of state.
        """
        super().__init__(geometry, physics)
        self.throat_area = throat_area
        self.subsonic_inflow = subsonic_inflow
        self.subsonic_outflow = subsonic_outflow

        # Get inflow properties for the gas
        self.g = inflow_state.cp / inflow_state.cv
        self.P_in = inflow_state.P
        self.rho_in = inflow_state.density_mass
        self.composition = physics.get_composition(inflow_state.Y)

    def __call__(self) -> FluidState:
        # Get cross-sectional area from the geometry
        area = self.geometry.area(0.0, self.geometry.xf)
        assert isinstance(area, np.ndarray)

        # If throat area is not given, assume choked flow
        min_area: float = area.min()
        self.throat_area = (
            min_area if self.throat_area is None else min(min_area, self.throat_area)
        )
        area_ratio = area / self.throat_area
        inflow_area_ratio = area_ratio[0]
        area_ratio_min = area_ratio.min()
        choked_flow = area_ratio_min < 1.0
        if choked_flow:
            # Adjust throat area based on choked flow - will affect requested boundary conditions
            inflow_area_ratio /= area_ratio_min
            area_ratio /= area_ratio_min
            self.throat_area /= area_ratio_min
        elif self.subsonic_inflow != self.subsonic_outflow:
            print(
                "Warning: Subsonic/supersonic transition requested, but flow may not be choked.\n"
                + f"Minimum area ratio = {area_ratio_min}."
            )

        # Get area ratios at cell centers
        area = self.geometry.area(0.0, self.geometry.xc)
        assert isinstance(area, np.ndarray)
        area_ratio = area / self.throat_area

        # Solve for allowable Mach numbers corresponding to given area ratio
        subsonic_mach: Array = mach_from_area_ratio(area_ratio, self.g, subsonic=True)
        supersonic_mach: Array = mach_from_area_ratio(
            area_ratio, self.g, subsonic=False
        )

        # Combine into one mach profile
        mach = subsonic_mach
        if self.subsonic_inflow != self.subsonic_outflow:
            idx = np.argmin(area_ratio)

            if self.subsonic_inflow:
                mach[idx + 1 :] = supersonic_mach[idx + 1 :]
            else:
                mach[:idx] = supersonic_mach[:idx]
        elif not self.subsonic_inflow and not self.subsonic_outflow:
            mach = supersonic_mach

        # Get stagnation properties based on inflow
        inflow_mach: float = mach_from_area_ratio(
            inflow_area_ratio, self.g, subsonic=self.subsonic_inflow
        )[0]
        _, inflow_Pratio, inflow_rhoratio = property_ratios(inflow_mach, self.g)

        # Get properties throughout
        _, Pratio_profile, rhoratio_profile = property_ratios(mach, self.g)

        n = self.geometry.n_cells
        state = FluidState(
            shape=(n,),
            pressure=self.P_in / inflow_Pratio * Pratio_profile,
            density=self.rho_in / inflow_rhoratio * rhoratio_profile,
            gamma=self.g * np.ones((n,)),
            composition=np.broadcast_to(
                self.composition, (n, self.composition.shape[0])
            ).copy(),
        )
        state.velocity = mach * self.physics.get_sound_speed(state)
        state.temperature = self.physics.get_temperature(state)

        return state


class InitializeIsentropicTotal(Initialization):
    def __init__(
        self,
        geometry: Geometry,
        physics: FluidPhysics,
        total_state: ct.Solution,
        inflow_mach: float | None = None,
        inflow_pressure: float | None = None,
        outflow_pressure: float | None = None,
        throat_area: float | None = None,
        subsonic_inflow: bool = True,
        subsonic_outflow: bool = True,
    ) -> None:
        """Applies isentropic flow relations to set initial conditions.

        Note that this formulation is only valid for a flow with constant specific
        heat ratio. It could be extended for non-ideal gas equations of state
        """
        super().__init__(geometry, physics)
        self.inflow_mach = inflow_mach
        self.inflow_pressure = inflow_pressure
        self.outflow_pressure = outflow_pressure
        self.throat_area = throat_area
        self.subsonic_inflow = subsonic_inflow
        self.subsonic_outflow = subsonic_outflow

        # Get stagnation properties for the gas
        self.g = total_state.cp / total_state.cv
        self.Pt = total_state.P
        self.Tt = total_state.T
        self.rhot = total_state.density_mass
        self.composition = physics.get_composition(total_state.Y)

    def __call__(self) -> FluidState:
        # Get cross-sectional area from the geometry
        x = self.geometry.xc
        area = self.geometry.area(0.0, x)
        assert isinstance(area, np.ndarray)

        # Determine throat area from given inputs
        inputs_done = False
        too_many_inputs_prefix = "Too many inputs specified. "
        error_msg = "Must specify one of inflow_velocity, inflow_pressure, outflow_pressure, or throat area."

        if self.throat_area is not None:
            inputs_done = True

        if self.inflow_mach is not None:
            Tratio, _, _ = property_ratios(self.inflow_mach, self.g)
            inflow_area = area[0]
            self.throat_area = (
                inflow_area
                * self.inflow_mach
                * (2.0 * Tratio / (self.g + 1)) ** (0.5 * (self.g + 1) / (self.g - 1))
            )
            if inputs_done:
                raise ValueError(too_many_inputs_prefix + error_msg)
            inputs_done = True

        if self.inflow_pressure is not None:
            Pratio = self.inflow_pressure / self.Pt
            Tratio = Pratio ** ((self.g - 1.0) / self.g)
            inflow_mach = np.sqrt(2.0 * (1.0 / Tratio - 1.0) / (self.g - 1.0))
            inflow_area = area[0]
            self.throat_area = (
                inflow_area
                * inflow_mach
                * (2.0 * Tratio / (self.g + 1)) ** (0.5 * (self.g + 1) / (self.g - 1))
            )
            if inputs_done:
                raise ValueError(too_many_inputs_prefix + error_msg)
            inputs_done = True

        if self.outflow_pressure is not None:
            Pratio = self.outflow_pressure / self.Pt
            Tratio = Pratio ** ((self.g - 1.0) / self.g)
            outflow_mach = np.sqrt(2.0 * (1.0 / Tratio - 1.0) / (self.g - 1.0))
            outflow_area = area[-1]
            self.throat_area = (
                outflow_area
                * outflow_mach
                * (2.0 * Tratio / (self.g + 1)) ** (0.5 * (self.g + 1) / (self.g - 1))
            )
            if inputs_done:
                raise ValueError(too_many_inputs_prefix + error_msg)
            inputs_done = True

        if not inputs_done:
            raise ValueError(error_msg)

        # Get the permissible subsonic and supersonic Mach numbers throughout
        # the domain from the area ratio
        assert isinstance(self.throat_area, float)
        area_ratio = area / self.throat_area

        # Check if flow is choked
        area_ratio_min = area_ratio.min()
        choked_flow = area_ratio_min <= 1.0
        if choked_flow:
            # Adjust throat area based on choked flow - will affect requested boundary conditions
            area_ratio /= area_ratio_min
            self.throat_area /= area_ratio_min

        # Solve for allowable Mach numbers corresponding to given area ratio
        subsonic_mach: Array = mach_from_area_ratio(area_ratio, self.g, subsonic=True)
        supersonic_mach: Array = mach_from_area_ratio(
            area_ratio, self.g, subsonic=False
        )

        # Combine into one mach profile
        mach = subsonic_mach
        if self.subsonic_inflow != self.subsonic_outflow:
            if not choked_flow:
                msg = f"Subsonic-supersonic transition requested, but flow is not choked. {area_ratio_min = }"
                raise ValueError(msg)
            idx = np.argmin(area_ratio)

            if self.subsonic_inflow:
                mach[idx:] = supersonic_mach[idx:]
            else:
                mach[:idx] = supersonic_mach[:idx]
        elif not self.subsonic_inflow and not self.subsonic_outflow:
            mach = supersonic_mach

        # Get properties throughout
        Tratio_profile, Pratio_profile, rhoratio_profile = property_ratios(mach, self.g)

        n = self.geometry.n_cells
        state = FluidState(
            shape=(n,),
            temperature=self.Tt * Tratio_profile,
            pressure=self.Pt * Pratio_profile,
            density=self.rhot * rhoratio_profile,
            gamma=self.g * np.ones((n,)),
            composition=np.broadcast_to(
                self.composition, (n, self.composition.shape[0])
            ).copy(),
        )
        state.velocity = mach * self.physics.get_sound_speed(state)

        return state


class InitializeInterpolate(Initialization):
    def __init__(
        self,
        geometry: Geometry,
        x_init: Array,
        state_init: FluidState,
    ) -> None:
        """Interpolate properties onto a new mesh."""
        self.geometry = geometry
        self.x_init = x_init
        self.state_init = state_init

    def __call__(self) -> FluidState:
        x_new = self.geometry.xc
        shape = (len(x_new),)

        # Straightforward to interpolate 1D data
        interpolated_data = {
            key: np.interp(x_new, self.x_init, val)
            for key, val in self.state_init.__dict__.items()
            if isinstance(val, np.ndarray) and val.ndim == 1
        }

        # Special treatment for 2D data
        comp_init = self.state_init.composition
        if comp_init is not None:
            nsp = comp_init.shape[1]
            composition = np.zeros((*shape, nsp))
            for isp in range(nsp):
                composition[:, isp] = np.interp(x_new, self.x_init, comp_init[:, isp])
            interpolated_data["composition"] = composition

        return FluidState(shape=shape, _cache_valid=False, **interpolated_data)


class InitializeCanteraArray(InitializeInterpolate):
    def __init__(
        self,
        geometry: Geometry,
        sol: ct.SolutionArray[ct.Solution],
    ) -> None:
        """Interpolate properties from a Cantera SolutionArray."""
        self.geometry = geometry

        self.x_init = sol.grid
        self.state_init = FluidState(
            shape=(len(self.x_init),),
            density=sol.density_mass,
            pressure=sol.P,
            temperature=sol.T,
            composition=sol.Y,
            velocity=sol.velocity,
        )


class InitializeRestart(Initialization):
    def __init__(
        self,
        groupname: str = "0",
        filename: str = "fluid_state.hdf5",
    ) -> None:
        self.groupname = groupname
        self.filename = filename

    def __call__(self) -> FluidState:
        return FluidState.restore(self.groupname, self.filename)
