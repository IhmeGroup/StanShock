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

# necessary modules
from __future__ import annotations

import numpy as np
from numba import double, njit

# Global variables (parameters) used by the solver
mt = 3  # number of ghost nodes
mn = 3  # number of 1D Euler equations

# Type signatures for numba
double1D = double[:]
double2D = double[:, :]
double3D = double[:, :, :]


@njit(double3D(double1D, double1D, double1D, double2D, double1D))
def weno5(r, u, p, Y, gamma):
    """
    This method implements the fifth-order WENO interpolation. This method
    follows that of Houim and Kuo (JCP2011)
        inputs:
            r=density
            u=velocity
            p=pressure
            Y=scalar variables matrix [x,scalars]
            gamma=specific heat ratio
        outputs:
            PLR=a matrix of the primitive variables [LR,]
    """
    nLR = 2
    nCells = len(r) - 2 * mt
    nFaces = nCells + 1
    nSc = len(Y[0])  # number of scalars
    nVar = mn + nSc  # [rho, rhou, rhoE, rhoY1, rhoY2, ...]
    nStencil = 2 * mt
    epWENO = 1.0e-06

    # Cell weight (WL(i,j,k); i=left(1) or right(2) j=stencil#,k=weight#)
    W = np.empty((2, 3, 3))
    W[0, 0, 0] = 0.333333333333333
    W[0, 0, 1] = 0.833333333333333
    W[0, 0, 2] = -0.166666666666667

    W[0, 1, 0] = -0.166666666666667
    W[0, 1, 1] = 0.833333333333333
    W[0, 1, 2] = 0.333333333333333

    W[0, 2, 0] = 0.333333333333333
    W[0, 2, 1] = -1.166666666666667
    W[0, 2, 2] = 1.833333333333333

    W[1, 0, 0] = W[0, 2, 2]
    W[1, 0, 1] = W[0, 2, 1]
    W[1, 0, 2] = W[0, 2, 0]

    W[1, 1, 0] = W[0, 1, 2]
    W[1, 1, 1] = W[0, 1, 1]
    W[1, 1, 2] = W[0, 1, 0]

    W[1, 2, 0] = W[0, 0, 2]
    W[1, 2, 1] = W[0, 0, 1]
    W[1, 2, 2] = W[0, 0, 0]

    # Stencil Weight (i=left(1) or right(2) j=stencil#)
    D = np.empty((2, 3))
    D[0, 0] = 0.3
    D[0, 1] = 0.6
    D[0, 2] = 0.1

    D[1, 0] = D[0, 2]
    D[1, 1] = D[0, 1]
    D[1, 2] = D[0, 0]

    B1 = 1.083333333333333
    B2 = 0.25

    B = np.empty(mt)
    PLR = np.empty((nLR, nFaces, nVar))
    YAverage = np.empty(nSc)
    U = np.empty(nVar)
    R = np.zeros((nVar, nVar))
    L = np.zeros((nVar, nVar))
    CStencil = np.empty(
        (nStencil, nVar)
    )  # all the characteristic values in the stencil

    for iFace in range(nFaces):  # iterate through each cell right edge
        iCell = iFace + 2  # face is on the right side of the cell

        # Face averages
        rAverage = 0.5 * (r[iCell] + r[iCell + 1])
        uAverage = 0.5 * (u[iCell] + u[iCell + 1])
        pAverage = 0.5 * (p[iCell] + p[iCell + 1])
        gammaAverage = 0.5 * (gamma[iCell] + gamma[iCell + 1])
        for kSc in range(nSc):
            YAverage[kSc] = 0.5 * (Y[iCell, kSc] + Y[iCell + 1, kSc])
        eAverage = pAverage / (rAverage * (gammaAverage - 1.0)) + 0.5 * uAverage**2.0
        hAverage = eAverage + pAverage / rAverage
        cAverage = np.sqrt(gammaAverage * pAverage / rAverage)

        # Right eigenvector matrix [rho, rhou, rhoE, rhoY1, rhoY2, ...]
        # Density wave
        R[0, 0] = 1.0
        R[1, 0] = uAverage - cAverage
        R[2, 0] = hAverage - uAverage * cAverage

        # Velocity wave
        R[0, 1] = 1.0
        R[1, 1] = uAverage
        R[2, 1] = 0.5 * uAverage**2.0

        # Energy wave
        R[0, 2] = 1.0
        R[1, 2] = uAverage + cAverage
        R[2, 2] = hAverage + uAverage * cAverage

        # Scalar waves
        for i in range(nSc):
            R[0, 3 + i] = 0.0
            R[1, 3 + i] = 0.0
            R[2, 3 + i] = 0.0
            for j in range(nSc):
                R[3 + j, 3 + i] = 1.0 if i == j else 0.0

        # Left eigenvector matrix
        gammaHat = gammaAverage - 1.0
        phi = 0.5 * gammaHat * uAverage**2.0

        # Acoustic waves
        L[0, 0] = 0.5 * (phi + cAverage * uAverage) / cAverage**2
        L[0, 1] = -0.5 * (gammaHat * uAverage + cAverage) / cAverage**2
        L[0, 2] = 0.5 * gammaHat / cAverage**2

        L[2, 0] = 0.5 * (phi - cAverage * uAverage) / cAverage**2
        L[2, 1] = -0.5 * (gammaHat * uAverage - cAverage) / cAverage**2
        L[2, 2] = 0.5 * gammaHat / cAverage**2

        # Velocity wave
        L[1, 0] = 1.0 - phi / cAverage**2
        L[1, 1] = gammaHat * uAverage / cAverage**2
        L[1, 2] = -gammaHat / cAverage**2

        # Scalar waves
        for i in range(nSc):
            L[3 + i, 3 + i] = 1.0

        for iVar in range(nVar):
            for iStencil in range(nStencil):
                iCellStencil = iStencil - 2 + iCell

                # Conservative variables [rho, rhou, rhoE, rhoY1, rhoY2, ...]
                U[0] = r[iCellStencil]  # density
                U[1] = r[iCellStencil] * u[iCellStencil]  # momentum
                U[2] = (
                    p[iCellStencil] / (gammaAverage - 1.0)
                    + 0.5 * r[iCellStencil] * u[iCellStencil] ** 2.0
                )  # energy
                for kSc in range(nSc):
                    U[3 + kSc] = (
                        r[iCellStencil] * Y[iCellStencil, kSc]
                    )  # scalar densities

                CStencil[iStencil, iVar] = 0.0
                for jVar in range(nVar):
                    CStencil[iStencil, iVar] += L[iVar, jVar] * U[jVar]

        # WENO interpolation in characteristic variables
        for N in range(nLR):
            for iVar in range(nVar):
                U[iVar] = 0.0
            for iVar in range(nVar):
                NO = N + 2

                # Smoothness parameters
                B[0] = (
                    B1
                    * (
                        CStencil[0 + NO, iVar]
                        - 2.0 * CStencil[1 + NO, iVar]
                        + CStencil[2 + NO, iVar]
                    )
                    ** 2.0
                    + B2
                    * (
                        3.0 * CStencil[0 + NO, iVar]
                        - 4.0 * CStencil[1 + NO, iVar]
                        + CStencil[2 + NO, iVar]
                    )
                    ** 2
                )
                B[1] = (
                    B1
                    * (
                        CStencil[-1 + NO, iVar]
                        - 2.0 * CStencil[0 + NO, iVar]
                        + CStencil[1 + NO, iVar]
                    )
                    ** 2.0
                    + B2 * (CStencil[-1 + NO, iVar] - CStencil[1 + NO, iVar]) ** 2
                )
                B[2] = (
                    B1
                    * (
                        CStencil[-2 + NO, iVar]
                        - 2.0 * CStencil[-1 + NO, iVar]
                        + CStencil[0 + NO, iVar]
                    )
                    ** 2.0
                    + B2
                    * (
                        CStencil[-2 + NO, iVar]
                        - 4.0 * CStencil[-1 + NO, iVar]
                        + 3.0 * CStencil[0 + NO, iVar]
                    )
                    ** 2
                )

                # Edge interpolation
                ATOT = 0.0
                CW = 0.0
                for iStencil in range(mt):
                    iStencilO = NO - iStencil
                    CINT = (
                        W[N, iStencil, 0] * CStencil[0 + iStencilO, iVar]
                        + W[N, iStencil, 1] * CStencil[1 + iStencilO, iVar]
                        + W[N, iStencil, 2] * CStencil[2 + iStencilO, iVar]
                    )
                    A = D[N, iStencil] / ((epWENO + B[iStencil]) ** 2)
                    ATOT += A
                    CW += CINT * A
                CiVar = CW / ATOT

                for jVar in range(nVar):
                    U[jVar] += R[jVar, iVar] * CiVar

            # Reconstruct primitives from conservatives
            rLR = U[0]
            uLR = U[1] / rLR
            eLR = U[2] / rLR
            pLR = rLR * (gammaAverage - 1.0) * (eLR - 0.5 * uLR**2.0)

            # Fill primitive matrix [rho, u, p, Y1, Y2, ...]
            PLR[N, iFace, 0] = rLR
            PLR[N, iFace, 1] = uLR
            PLR[N, iFace, 2] = pLR
            for kSc in range(nSc):
                PLR[N, iFace, 3 + kSc] = U[3 + kSc] / rLR

    # First order at boundaries
    for N in range(nLR):
        for iFace in range(mt):
            iCell = iFace + 2
            PLR[N, iFace, 0] = r[iCell + N]
            PLR[N, iFace, 1] = u[iCell + N]
            PLR[N, iFace, 2] = p[iCell + N]
            for kSc in range(nSc):
                PLR[N, iFace, 3 + kSc] = Y[iCell + N, kSc]
        for iFace in range(nFaces - mt, nFaces):
            iCell = iFace + 2
            PLR[N, iFace, 0] = r[iCell + N]
            PLR[N, iFace, 1] = u[iCell + N]
            PLR[N, iFace, 2] = p[iCell + N]
            for kSc in range(nSc):
                PLR[N, iFace, 3 + kSc] = Y[iCell + N, kSc]

    # Create primitive matrix for limiter
    P = np.zeros((nCells + 2 * mt, nVar))
    P[:, 0] = r[:]
    P[:, 1] = u[:]
    P[:, 2] = p[:]
    P[:, 3:] = Y[:, :]

    # Apply limiter
    alpha = 2.0
    threshold = 1e-6
    epsilon = 1.0e-15
    for N in range(nLR):
        for iFace in range(nFaces):
            for iVar in range(nVar):
                iCell = iFace + 2 + N
                iCellm1 = iCell - 1 + 2 * N
                iCellp1 = iCell + 1 - 2 * N
                iCellm2 = iCell - 2 + 4 * N
                iCellp2 = iCell + 2 - 4 * N
                # check the error threshold for smooth regions
                error = abs(
                    (
                        -P[iCellm2, iVar]
                        + 4.0 * P[iCellm1, iVar]
                        + 4.0 * P[iCellp1, iVar]
                        - P[iCellp2, iVar]
                        + epsilon
                    )
                    / (6.0 * P[iCell, iVar] + epsilon)
                    - 1.0
                )
                if error < threshold:
                    continue
                # compute limiter
                if P[iCell, iVar] != P[iCellm1, iVar]:
                    phi = min(
                        alpha,
                        alpha
                        * (P[iCellp1, iVar] - P[iCell, iVar])
                        / (P[iCell, iVar] - P[iCellm1, iVar]),
                    )
                    phi = min(
                        phi,
                        2.0
                        * (PLR[N, iFace, iVar] - P[iCell, iVar])
                        / (P[iCell, iVar] - P[iCellm1, iVar]),
                    )
                    phi = max(0.0, phi)
                else:
                    phi = alpha
                # apply limiter
                PLR[N, iFace, iVar] = P[iCell, iVar] + 0.5 * phi * (
                    P[iCell, iVar] - P[iCellm1, iVar]
                )

    return PLR
