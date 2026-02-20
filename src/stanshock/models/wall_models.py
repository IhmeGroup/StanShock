from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import newton

from stanshock.system.backend import Array


@dataclass(slots=True)
class WallState:
    Re: Array
    M: Array
    T: Array
    Tw: Array
    Pr: Array
    gamma: Array
    h_St: Array
    q: Array
    Cf: Array | None = None


def get_wall_state(
    rho: Array,
    U: Array,
    mu: Array,
    a: Array,
    cp: Array,
    k: Array,
    gamma: Array,
    Lc: Array | float,
    T: Array,
    wall_temperature: Array | float | None,
) -> WallState:
    Re = np.abs(rho * U * Lc / mu)
    M = np.abs(U / a)

    if wall_temperature is None:  # noqa: SIM108
        Tw = T.copy()  # adiabatic
    else:
        Tw = np.array((wall_temperature,))  # isothermal

    Pr = cp * mu / k
    h_St = rho * U * cp

    q = 0.5 * rho * U**2 * np.sign(U)
    return WallState(Re=Re, M=M, T=T, Tw=Tw, Pr=Pr, gamma=gamma, h_St=h_St, q=q)


"""
##### WALL HEAT FLUX MODELS #####
"""

class HeatFlux(ABC):
    @abstractmethod
    def __call__(self, wall: WallState) -> Array:
        pass

    def get_stanton_number(self, Pr: Array, cf: Array) -> Array:
        """
        Defines Stanton number for gas phase flow as func. of Pr and Cf
        Uses empirical correlations from "Convective Heat and Mass Transfer", Kays 1993
        """
        return (cf / 2.0) / (1 + 13 * (Pr ** (2 / 3) - 1) * np.sqrt(cf / 2.0))


class CompressibleHeatFlux(HeatFlux):
    """
    Implements recovery temperature model from Van Driest (1956)
    """

    def __call__(self, wall: WallState) -> Array:
        assert wall.Cf is not None
        T_r = self.recovery_temperature(wall.T, wall.M, wall.Pr, wall.gamma)
        St = self.get_stanton_number(wall.Pr, wall.Cf)
        h = St * wall.h_St
        return h * (T_r - wall.Tw)

    def recovery_temperature(
        self, T: Array, M: Array, Pr: Array, gamma: Array
    ) -> Array:
        return T * (1.0 + (Pr ** (1.0 / 3.0)) * 0.5 * (gamma - 1.0) * M**2)


class IncompressibleHeatFlux(HeatFlux):
    def __call__(self, wall: WallState) -> Array:
        assert wall.Cf is not None
        laminar = wall.Re < 1e3
        St = self.get_stanton_number(wall.Pr, wall.Cf)
        # Analytical soln for low Re flows.
        St[laminar] = 3.657 / (wall.Re[laminar] / wall.Pr[laminar])
        h = St * wall.h_St
        return h * (wall.T - wall.Tw)


"""
##### SKIN FRICTION MODELS #####
"""

class SkinFriction(ABC):
    @abstractmethod
    def __call__(self, wall: WallState) -> Array:
        """
        Returns the skin friction coefficient Cf for fully developed
        pipe flow based on wall state.
        """


class CompressibleInertSkinFriction(SkinFriction):
    """
    Tabulated implementation of DeChant & Tattar's model (1998).
    Valid for air; in low Re limit, reduces to Prandtl's law.
    """

    def __init__(
        self,
        Re_range: tuple[float, float] = (1e3, 1e7),
        Mach_range: tuple[float, float] = (0.01, 4.0),
        T_Tw_range: tuple[float, float] = (0.1, 8.0),
        gamma: float = 1.4,
        recovery: float = 0.88,
    ) -> None:
        self.N_Re = 30
        self.N_Mach = 30
        self.N_T_Tw = 30

        self.gamma = gamma
        self.recovery = recovery

        self.ReTable = np.logspace(
            np.log10(Re_range[0]), np.log10(Re_range[1]), self.N_Re
        )
        self.MachTable = np.linspace(*Mach_range, self.N_Mach)
        self.T_TwTable = np.linspace(*T_Tw_range, self.N_T_Tw)

        self.Mach_grid, self.T_Tw_grid = np.meshgrid(
            self.MachTable, self.T_TwTable, indexing="ij"
        )
        self.cfTable = np.zeros((self.N_Re, self.N_Mach, self.N_T_Tw))
        self._build_table()
        self.interp = RegularGridInterpolator(
            (self.ReTable, self.MachTable, self.T_TwTable),
            self.cfTable,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def _build_table(self) -> None:
        F_jk, G_jk = get_dechant_constants(
            self.Mach_grid, self.T_Tw_grid, self.gamma, self.recovery
        )
        for i, Re in enumerate(self.ReTable):
            self.cfTable[i, :, :] = cf_dechant_solve(Re, F_jk, G_jk)

    def __call__(self, wall: WallState) -> Array:
        T_Tw = wall.T / wall.Tw
        pts = np.column_stack((wall.Re, wall.M, T_Tw))
        return self.interp(pts)


class CompressibleReactingSkinFriction(SkinFriction):
    """
    Extension of the DeChant & Tattar model for variable
    gamma and Prandtl number.
    """

    def __call__(self, wall: WallState) -> Array:
        recovery = wall.Pr ** (1.0 / 3.0)
        T_Tw = wall.T / wall.Tw
        F, G = get_dechant_constants(
            wall.M,
            T_Tw,
            wall.gamma,
            recovery,
        )
        return cf_dechant_solve(wall.Re, F, G)


class IncompressibleInertSkinFriction(SkinFriction):
    """
    Karman-Nikuradse (1933) skin friction model for Re > 1000
    Analytical turbulent pipe-flow solution for Re < 1000
    """

    def __init__(self, Re_range: tuple[float, float] = (1e3, 1e7)) -> None:
        self.Re_range = Re_range
        self.N_Re = 100
        self.ReTable = np.logspace(
            np.log10(min(self.Re_range)), np.log10(max(self.Re_range)), self.N_Re
        )

        self._build_table()

    def _build_table(self) -> None:
        self.cfTable = cf_karman_solve(self.ReTable)

    def __call__(self, wall: WallState) -> Array:
        cf = np.zeros_like(wall.Re)
        laminar = wall.Re < self.Re_range[0]
        cf[laminar] = 16.0 / wall.Re[laminar]
        cf[~laminar] = np.interp(wall.Re[~laminar], self.ReTable, self.cfTable)
        return cf


"""
DeChant Skin Friction Model Functions
"""


def get_dechant_constants(
    M: Array, T_Tw: Array, gamma: Array | float, recovery: Array | float
) -> tuple[Array, Array]:
    k = 0.5 * (gamma - 1.0) * M**2
    A = (T_Tw - 1.0) + k * T_Tw
    B = np.sqrt(k * T_Tw * recovery)

    D = np.sqrt(A**2 + 4.0 * B**2)
    C = np.arcsin((2.0 * B**2 - A) / D) + np.arcsin(A / D)

    E = T_Tw * (1.505 / (1.0 + (0.505 / T_Tw)))

    F = C / np.sqrt(k * recovery)
    G = 1.77 * np.log(E) - 0.6005
    return F, G


def f_dechant(x: Array, Re: Array, F: Array, G: Array) -> Array:
    return 1.77 * np.log(x * Re) - (F / x) + G


def dfdx_dechant(x: Array, Re: Array, F: Array, G: Array) -> Array:
    return (1.77 / x) + (F / x**2)


def cf_dechant_solve(Re: Array, F: Array, G: Array) -> Array:
    x0 = np.sqrt(3e-3) * np.ones_like(F)
    cf_sqrt = newton(
        func=f_dechant,
        fprime=dfdx_dechant,
        x0=x0,
        args=(Re, F, G),
        tol=1e-10,
        maxiter=100,
    )
    return cf_sqrt * cf_sqrt


"""
Karman Nikuradse Skin Friction Model Functions
"""


def f_karman(x: Array, Re: Array) -> Array:
    return 2.46 * x * np.log(Re * x) + 0.3 * x - 1.0


def dfdx_karman(x: Array, Re: Array) -> Array:
    return 2.46 * (np.log(Re * x) + 1.0) + 0.3


def cf_karman_solve(Re: Array) -> Array:
    x0 = np.sqrt(3e-3) * np.ones_like(Re)
    cf_sqrt2 = newton(func=f_karman, fprime=dfdx_karman, x0=x0, args=(Re,))
    return 2.0 * cf_sqrt2**2
