from __future__ import annotations

from typing import cast

import numpy as np
from scipy.optimize import root

from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Index, Unpack
from stanshock.system.base import PrecomputeSteps, RightHandSide


class SkinFriction:
    """
    This functor computes the skin friction function. Since the skin friction
    function is partially implicit, it interpolates from a table of values at
    outset.
        inputs:
            ReCrit = the critical Reynolds number for transition
            ReMax = the maximum value for the table
        outputs:
            cf = numpy array of the skin friction coefficient
    """

    def __init__(self, ReCrit: float = 2300, ReMax: float = 1e9) -> None:
        # store the values and compute the Reynolds number table
        self.ReMax = ReMax
        self.ReCrit = ReCrit
        self.ReTable = np.logspace(np.log10(self.ReCrit), np.log10(ReMax))

        # define the residual of the Karman-Nikuradse function and its derivative
        def f(x: Array) -> Array:
            return cast(Array, 2.46 * x * np.log(self.ReTable * x) + 0.3 * x - 1.0)

        def jac(x: Array) -> Array:
            dx = 2.46 * (np.log(self.ReTable * x) + 1.0) + 0.3
            return cast(Array, np.diagflat(dx))

        # use the scipy root finding method
        x0 = 1.0 / (2.236 * np.log(self.ReTable) - 4.639)  # use fit for initial value
        self.cfTable = (
            root(f, x0, jac=jac).x
        ) ** 2.0 * 2.0  # grid of values for interpolation

    def __call__(self, Re: Array) -> Array:
        cf = np.zeros_like(Re)
        laminarIndices = np.logical_and(Re > 0.0, Re <= self.ReCrit)
        cf[laminarIndices] = 16.0 / Re[laminarIndices]
        turbulentIndices = Re > self.ReCrit
        cf[turbulentIndices] = np.interp(
            Re[turbulentIndices], self.ReTable, self.cfTable
        )
        if np.any(Re > self.ReMax):
            msg = f"Error: Reynolds number exceeds the maximum value of {self.ReMax:f}: skinFriction Table bounds must be adjusted"
            raise Exception(msg)
        return cf


class BoundaryLayer(RightHandSide):
    def __init__(
        self,
        wall_temperature: Array | float | None = None,
        skin_friction_coefficient: SkinFriction | None = None,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        super().__init__(**precompute_steps)
        self.wall_temperature = wall_temperature

        if skin_friction_coefficient is None:
            self.skin_friction_coefficient = SkinFriction()  # initialize the functor
        else:
            self.skin_friction_coefficient = skin_friction_coefficient

        # Provides momentum and energy source terms
        self.idx_source: Index = np.array([0, 1])

    def get_nusselt_number(self, Re: Array, Pr: Array, cf: Array) -> Array:
        """
        This function defines the nusselt Number as a function of the
        Reynolds number. These functions are empirical correlations taken
        from Kayes. The selection of the correlations assumes that this solver
        will be used for gasses.
            inputs:
                Re=Reynolds number
                Pr=Prandtl number
                cf=skin friction
            outputs:
                Nu=Nusselt number
        """
        # define the transitional Reynolds number
        ReCrit = 2300
        ReLowTurbulent = 2e5  # taken frkom figure 14-5 of Kayes for Pr=0.7
        Nu = np.zeros_like(Re)

        # laminar portion of the flow
        laminarIndices: Index = np.logical_and(Re > 0.0, Re <= ReCrit)
        Nu[laminarIndices] = 3.657  # from the analytical solution

        # low turbulent portion of the flow (accounts for isothermal wall)
        lowTurbulentIndices: Index = np.logical_and(Re > ReCrit, Re <= ReLowTurbulent)
        ReLT, PrLT = Re[lowTurbulentIndices], Pr[lowTurbulentIndices]
        Nu[lowTurbulentIndices] = (
            0.021 * PrLT**0.5 * ReLT**0.8
        )  # empircal correlation for isothermal case

        # highly turbulent portion of the flow (data shows that boundary condition is less important)
        # highTurublentIndices = Re > ReLowTurbulent
        highTurublentIndices = Re > 2300.0
        ReHT, PrHT, cfHT = (
            Re[highTurublentIndices],
            Pr[highTurublentIndices],
            cf[highTurublentIndices],
        )
        Nu[highTurublentIndices] = (
            ReHT
            * PrHT
            * cfHT
            / 2.0
            / (0.88 + 13.39 * (PrHT ** (2.0 / 3.0) - 0.78) * np.sqrt(cfHT / 2.0))
        )
        return Nu

    def source_implementation(
        self,
        time: float,
        state_array: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        """Boundary layer contribution to RHS."""
        _ = time, state_array, face_states, avg_face_states, face_gradients
        assert self.physics is not None
        assert state is not None
        assert state.density is not None
        assert state.velocity is not None
        rhs = np.zeros((*state.shape, 2))

        x = self.geometry.xc[self.idx_domain]
        characteristic_length = self.geometry.characteristic_length(time, x)
        hydraulic_diameter = self.geometry.hydraulic_diameter(time, x)

        # Compute gas properties
        T = state.temperature = self.physics.get_temperature(state)
        mu = self.physics.get_mu(state)

        # Shear stress on wall
        Re = abs(state.density * state.velocity * characteristic_length / mu)
        cf = self.skin_friction_coefficient(Re)
        shear = (
            cf * (0.5 * state.density * state.velocity**2.0) * np.sign(state.velocity)
        )
        rhs[:, 0] = -4.0 / hydraulic_diameter * shear

        # Stanton number and heat transfer to wall
        if self.wall_temperature is not None:
            cp = self.physics.get_cp(state)
            k = self.physics.get_thermal_conductivity(state)
            Pr = cp * mu / k
            Nu = self.get_nusselt_number(Re, Pr, cf)
            qloss = Nu * k / characteristic_length * (T - self.wall_temperature)

            rhs[:, 1] = -4.0 / hydraulic_diameter * qloss

        return rhs
