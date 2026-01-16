from __future__ import annotations

import numpy as np
from scipy.optimize import newton
from scipy.interpolate import RegularGridInterpolator

class CompressibleSkinFriction:
    """
    Pre-tabulated implementation of the DeChant skin friction model (1998) as function of Re, Mach, T/T_w
    Currently only implemented for air.
    """
    def __init__(self, ReMin=100, ReMax=1e7, MachMin=0.01, MachMax=4, T_ratMin=0.01, T_ratMax=8):
        self.ReMax = ReMax
        self.ReMin = ReMin
        self.MachMin = MachMin
        self.MachMax = MachMax
        self.T_ratMin = T_ratMin
        self.T_ratMax = T_ratMax
        self.N_Re = 30
        self.N_Mach = 30
        self.N_T_rat=11
        self.gamma = 1.4

        self.ReTable = np.logspace(np.log10(self.ReMin), np.log10(self.ReMax), self.N_Re)
        self.MachTable = np.linspace(self.MachMin, self.MachMax, int(self.N_Mach))
        self.T_ratTable = np.linspace(self.T_ratMin, self.T_ratMax, self.N_T_rat)

        self.M_grid, self.T_grid = np.meshgrid(self.MachTable,self.T_ratTable,indexing="ij")
        self.F, self.G = self._precompute_FG(self.M_grid, self.T_grid)
        
        self.cfTable = np.zeros((self.N_Re, self.N_Mach, self.N_T_rat))
        self._build_table()
        self.interp = RegularGridInterpolator(
            (self.ReTable, self.MachTable, self.T_ratTable),
            self.cfTable,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def __call__(self, Re, M, T_rat):
        pts = np.array([Re, M, T_rat]).T
        return self.interp(pts)

    def _precompute_FG(self, M, T_rat):
        gamma = self.gamma
        recovery = 0.88
        C_axi = -0.6005

        k = 0.5 * (gamma - 1.0) * M**2
        A = (T_rat - 1) + k*T_rat
        B = np.sqrt(k * T_rat * recovery)

        D = np.sqrt(A**2 + 4.0 * B**2)
        C = np.arcsin((2.0 * B**2 - A) / D) + np.arcsin(A / D)

        E = T_rat * (1.505 / (1.0 + (0.505 / T_rat)))

        F = C / np.sqrt(k * recovery)
        G = 1.77 * np.log(E) + C_axi

        return F, G

    def _f(self, x, Re):
        return 1.77 * np.log(x * Re) - (self.F / x) + self.G

    def _dfdx(self, x, Re):
        return (1.77 / x) + (self.F / x**2)

        
    def _solve_cf_slice(self, Re):
        x0 = np.sqrt(3e-3) * np.ones_like(self.F)

        sqrt_cf = newton(
            func=self._f,
            fprime=self._dfdx,
            x0=x0,
            args=(Re,),
            tol=1e-10,
            maxiter=100,
        )
        return (sqrt_cf**2)
    
    def _build_table(self):
        for i, Re in enumerate(self.ReTable):
            self.cfTable[i, :, :] = self._solve_cf_slice(Re)