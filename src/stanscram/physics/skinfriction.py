"""
Copyright 2017 Kevin Grogan
Copyright 2024 Matthew Bonanni

This file is part of StanScram.

StanScram is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License.

StanScram is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with StanScram.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root


class skinFriction:
    """
    Functor: skinFriction
    ---------------------------------------------------------------------------
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
