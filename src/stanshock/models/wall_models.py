from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from scipy.optimize import  newton
from scipy.interpolate import RegularGridInterpolator
from stanshock.system.backend import Array

@dataclass(slots=True)
class WallState:
    Re: Array
    M: Array | None
    T: Array | None
    Tw: Array | float | None
    Pr: Array | None
    gamma: Array | None
    h_St: Array | None
    q: Array | None
    Cf: Array | None = None

def get_wall_state(
    rho: Array,
    U: Array,
    mu: Array,
    a: Array,
    cp: Array | None,
    k: Array | None,
    gamma: Array | None,
    Dh: Array | float,
    T: Array,
    wall_temperature: Array | float | None,
) -> WallState:
    
    Re = np.abs(rho * U * Dh / mu)
    M = np.abs(U / a)

    if wall_temperature is None:
        Tw = T.copy()
    else:
        Tw = wall_temperature

    if cp is not None and k is not None:
        Pr = cp * mu / k
    else:
        Pr = None

    h_St = rho * U * cp
    q = 0.5 * rho * U**2 * np.sign(U)
    return WallState(
        Re=Re,
        M=M,
        T=T,
        Tw=Tw,
        Pr=Pr,
        gamma=gamma,
        h_St=h_St,
        q=q,
    )


class WallModel(ABC):
    @abstractmethod
    def __call__(self, wall: WallState) -> Array:
        """
        This method will always return the shear stress tau_w or heat flux q_w per unit wall area dAw.
        """
        pass

class HeatFlux_Compressible(WallModel):
    def __call__(self, wall: WallState) -> Array:
        T_r = self.recovery_temperature(wall.T, wall.M, wall.Pr, wall.gamma)
        St = get_stanton_number(wall.Re, wall.Pr, wall.Cf)
        h = St * wall.h_St
        return h * (T_r - wall.Tw)

    def recovery_temperature(self, T: Array | float, M: Array | float, Pr: Array | float, gamma: Array | float) -> Array | float:
        return (T * (1.0 + (Pr ** (1.0 / 3.0)) * 0.5 * (gamma - 1.0) * M ** 2))


class HeatFlux_Incompressible(WallModel):
    def __call__(self, wall: WallState) -> Array:
        St = get_stanton_number(wall.Re, wall.Pr, wall.Cf)
        h = St * wall.h_St
        return h * (wall.T - wall.Tw)
    
class ShearStress_CompressibleInert(WallModel):
    def __init__(self,
        Re_range = [1e3, 1e7],
        Mach_range = [0.01, 4.0],
        T_Tw_range = [0.10, 8.0],
        gamma = 1.4,
        recovery = 0.88,
    ):
        self.Re_range = Re_range
        self.Mach_range = Mach_range
        self.T_Tw_range = T_Tw_range

        self.N_Re = 30
        self.N_Mach = 30
        self.N_T_Tw = 30

        self.gamma = gamma
        self.recovery = recovery

        self.ReTable = np.logspace(np.log10(min(self.Re_range)), np.log10(max(self.Re_range)), self.N_Re)
        self.MachTable = np.linspace(min(self.Mach_range), max(self.Mach_range), self.N_Mach)
        self.T_TwTable = np.linspace(min(self.T_Tw_range), max(self.T_Tw_range), self.N_T_Tw)

        self.Mach_grid, self.T_Tw_grid = np.meshgrid(
                self.MachTable, self.T_TwTable, indexing="ij"
            )
        self.cfTable = np.zeros((self.N_Re, self.N_Mach, self.N_T_Tw))
        self._build_table()
        self.interp = RegularGridInterpolator((self.ReTable, self.MachTable, self.T_TwTable), self.cfTable,
                                              method="linear", bounds_error=False,fill_value=None,)
        
    
    def _build_table(self):
        F_jk, G_jk = get_dechant_constants(self.Mach_grid, self.T_Tw_grid, self.gamma, self.recovery)
        for i, Re in enumerate(self.ReTable):
            self.cfTable[i, :, :] = cf_dechant_solve(Re, F_jk, G_jk)
        
    def __call__(self, wall: WallState) -> Array:
        T_Tw = wall.T / wall.Tw
        pts = np.column_stack((wall.Re, wall.M, T_Tw))
        cf = self.interp(pts)
        return wall.q * cf

class ShearStress_IncompressibleInert(WallModel):
    def __init__(self, 
                 Re_range = [1e3, 1e7],
    ):
        self.Re_range = Re_range
        self.N_Re = 100
        self.ReTable = np.logspace(np.log10(min(self.Re_range)), np.log10(max(self.Re_range)), self.N_Re)
        self.cfTable = np.zeros(self.Re_range)


    def _build_table(self):
        self.cfTable = cf_karman_solve(self.ReTable)

    def __call__(self, wall: WallState) -> Array:
        cf = np.interp(wall.Re, self.ReTable, self.cfTable)
        return wall.q * cf
    

class ShearStress_CompressibleReacting(WallModel):
    requires = {"Re", "M", "T_Tw", "Pr", "gamma"}
    def __call__(self, wall: WallState) -> Array:
        recovery = wall.Pr ** (1.0 / 3.0)
        T_Tw = wall.T / wall.Tw
        F, G = get_dechant_constants(
            wall.M,
            T_Tw,
            wall.gamma,
            recovery,
        )
        cf = cf_dechant_solve(wall.Re, F, G)
        return wall.q * cf
    






"""
DeChant Skin Friction Model Functions
"""

def get_dechant_constants(M: Array, T_Tw: Array, gamma: Array | float, recovery: Array | float):
    k = 0.5 * (gamma - 1.0) * M**2
    A = (T_Tw - 1.0) + k * T_Tw
    B = np.sqrt(k * T_Tw * recovery)

    D = np.sqrt(A**2 + 4.0 * B**2)
    C = np.arcsin((2.0 * B**2 - A) / D) + np.arcsin(A / D)

    E = T_Tw * (1.505 / (1.0 + (0.505 / T_Tw)))

    F = C / np.sqrt(k * recovery)
    G = 1.77 * np.log(E) - 0.6005
    return F, G

def f_dechant(x: Array, Re: Array | float, F: Array | float, G: Array | float) -> Array | float:
        return 1.77 * np.log(x * Re) - (F / x) + G

def dfdx_dechant(x: Array | float, Re: Array | float, F: Array | float, G: Array | float) -> Array | float:
        return (1.77 / x) + (F / x**2)

def cf_dechant_solve(Re: Array | float, F: Array | float, G: Array | float) -> Array | float:
    x0 = np.sqrt(3e-3) * np.ones_like(F)
    cf_sqrt = newton(func=f_dechant, fprime=dfdx_dechant, x0=x0, args=(Re, F, G,), tol=1e-10,maxiter=100)
    return cf_sqrt**2


"""
Karman Nikuradse Skin Friction Model Functions
"""

def f_karman(x: Array | float, Re: Array | float):
    return 2.46 * x * np.log(Re * x) + 0.3 * x - 1.0

def dfdx_karman(x: Array | float, Re: Array | float):
    return 2.46 * (np.log(Re * x) + 1.0) + 0.3

def cf_karman_solve(Re: Array | float) -> Array | float:
    x0 = np.sqrt(3e-3) * np.ones_like(Re)
    cf_sqrt2 = newton(func=f_karman, fprime=dfdx_karman, x0=x0, args=(Re,))
    return 2.0 * cf_sqrt2**2


"""
Wall Heat Flux Functions
"""
def get_stanton_number(Re: Array, Pr: Array, cf: Array) -> Array:
        """
        Defines Stanton number for gas phase flow as func. of Re, Pr, Cf
        Uses empirical correlations from Kays 
        """
        return ((cf / 2.0) / (1 + 13 * (Pr ** (2/3) - 1) * np.sqrt(cf / 2.0)))

def recovery_temperature(T: Array | float, M: Array | float, Pr: Array | float, gamma: Array | float) -> Array | float:
    return (T * (1.0 + (Pr ** (1.0 / 3.0)) * 0.5 * (gamma - 1.0) * M ** 2))



