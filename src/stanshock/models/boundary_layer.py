from __future__ import annotations

import numpy as np
from scipy.optimize import root

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.system.base import Array, RightHandSide


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

    def __init__(self, ReCrit=2300, ReMax=1e9):
        # store the values and compute the Reynolds number table
        self.ReMax = ReMax
        self.ReCrit = ReCrit
        self.ReTable = np.logspace(np.log10(self.ReCrit), np.log10(ReMax))

        # define the residual of the Karman-Nikuradse function and its derivative
        def f(x):
            return 2.46 * x * np.log(self.ReTable * x) + 0.3 * x - 1.0

        def jac(x):
            dx = 2.46 * (np.log(self.ReTable * x) + 1.0) + 0.3
            return np.diagflat(dx)

        # use the scipy root finding method
        x0 = 1.0 / (2.236 * np.log(self.ReTable) - 4.639)  # use fit for initial value
        self.cfTable = (
            root(f, x0, jac=jac).x
        ) ** 2.0 * 2.0  # grid of values for interpolation

    def __call__(self, Re):
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
        hydraulic_diameter,
        characteristic_length,
        wall_temperature=None,
        skin_friction_coefficient=None,
    ) -> None:
        self.hydraulic_diameter = hydraulic_diameter
        self.characteristic_length = characteristic_length
        self.wall_temperature = wall_temperature
        self.skin_friction_coefficient = skin_friction_coefficient

        if self.skin_friction_coefficient is None:
            self.skin_friction_coefficient = SkinFriction()  # initialize the functor

    def get_nusselt_number(self, Re, Pr, cf):
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
        laminarIndices = np.logical_and(Re > 0.0, Re <= ReCrit)
        Nu[laminarIndices] = 3.657  # from the analytical solution

        # low turbulent portion of the flow (accounts for isothermal wall)
        lowTurublentIndices = np.logical_and(Re > ReCrit, Re <= ReLowTurbulent)
        ReLT, PrLT = Re[lowTurublentIndices], Pr[lowTurublentIndices]
        Nu[lowTurublentIndices] = (
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

    def __call__(self, _time: float, state: FluidState, physics: FluidPhysics) -> Array:
        """Boundary layer contribution to RHS."""
        if self.hydraulic_diameter is None or self.characteristic_length is None:
            msg = "Combustor improperly initialized for boundary layer terms"
            raise Exception(msg)

        rhs = np.zeros((*state.shape, 3+physics.n_scalars))

        # Compute gas properties
        T = state.temperature = physics.get_temperature(state)
        mu = physics.get_mu(state)

        # Shear stress on wall
        Re = abs(state.density * state.velocity * self.characteristic_length / mu)
        cf = self.skin_friction_coefficient(Re)
        shear = (
            cf * (0.5 * state.density * state.velocity**2.0) * np.sign(state.velocity)
        )
        rhs[:, 1] = -4.0 / self.hydraulic_diameter * shear

        # Stanton number and heat transfer to wall
        if self.wall_temperature is not None:
            cp = physics.get_cp(state)
            k = physics.get_thermal_conductivity(state)
            Pr = cp * mu / k
            Nu = self.get_nusselt_number(Re, Pr, cf)
            qloss = Nu * k / self.characteristic_length * (T - self.wall_temperature)

            rhs[:, 2] = -4.0 / self.hydraulic_diameter * qloss

        return rhs
