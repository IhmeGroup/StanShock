from __future__ import annotations

import numpy as np
from numba import double, njit

# Global variables (parameters) used by the solver
mn = 3  # number of 1D Euler equations

# Type signatures for numba
double1D = double[:]
double2D = double[:, :]
double3D = double[:, :, :]


@njit(double2D(double2D, double2D, double2D, double3D, double1D))
def lax_friedrichs_flux(rLR, uLR, pLR, YLR, gamma):
    """
    This method computes the flux at each interface
        inputs:
            rLR=array containing left and right density states [nLR,nFaces]
            uLR=array containing left and right velocity states [nLR,nFaces]
            pLR=array containing left and right pressure states [nLR,nFaces]
            YLR=array containing left and right scalar states
                [nLR,nFaces,nSc]
            gamma= array containing the specific heat [nFaces]
        return:
            F=modeled Euler fluxes [nFaces,mn+nSc]
    """
    nLR = len(rLR)
    nFaces = len(rLR[0])
    nSc = YLR[0].shape[1]
    nDim = mn + nSc

    # find the maximum wave speed
    lambdaMax = 0.0
    for iFace in range(nFaces):
        a = max(
            np.sqrt(gamma[iFace] * pLR[0, iFace] / rLR[0, iFace]),
            np.sqrt(gamma[iFace] * pLR[1, iFace] / rLR[1, iFace]),
        )
        u = max(abs(uLR[0, iFace]), abs(uLR[1, iFace]))
        lambdaMax = max(lambdaMax, u + a)
    lambdaMax *= 0.9
    # find the regular flux
    FLR = np.empty((2, nFaces, nDim))
    for K in range(nLR):
        for iFace in range(nFaces):
            FLR[K, iFace, 0] = rLR[K, iFace] * uLR[K, iFace]
            FLR[K, iFace, 1] = rLR[K, iFace] * uLR[K, iFace] ** 2.0 + pLR[K, iFace]
            FLR[K, iFace, 2] = uLR[K, iFace] * (
                gamma[iFace] / (gamma[iFace] - 1) * pLR[K, iFace]
                + 0.5 * rLR[K, iFace] * uLR[K, iFace] ** 2.0
            )
            for kSc in range(nSc):
                FLR[K, iFace, mn + kSc] = (
                    rLR[K, iFace] * uLR[K, iFace] * YLR[K, iFace, kSc]
                )

    # compute the modeled flux
    F = np.empty((nFaces, mn + nSc))
    U = np.empty((nLR, mn + nSc))
    for iFace in range(nFaces):
        for K in range(nLR):
            U[K, 0] = rLR[K, iFace]
            U[K, 1] = rLR[K, iFace] * uLR[K, iFace]
            U[K, 2] = (
                pLR[K, iFace] / (gamma[iFace] - 1.0)
                + 0.5 * rLR[K, iFace] * uLR[K, iFace] ** 2.0
            )
            for kSc in range(nSc):
                U[K, mn + kSc] = rLR[K, iFace] * YLR[K, iFace, kSc]
        for iDim in range(nDim):
            FBar = 0.5 * (FLR[0, iFace, iDim] + FLR[1, iFace, iDim])
            F[iFace, iDim] = FBar - 0.5 * lambdaMax * (U[1, iDim] - U[0, iDim])
    return F


@njit(double2D(double2D, double2D, double2D, double3D, double1D))
def hllc_flux(rLR, uLR, pLR, YLR, gamma):
    """
    This method computes the flux at each interface
        inputs:
            rLR=array containing left and right density states [nLR,nFaces]
            uLR=array containing left and right velocity states [nLR,nFaces]
            pLR=array containing left and right pressure states [nLR,nFaces]
            YLR=array containing left and right scalar states
                [nLR,nFaces,nSc]
            gamma= array containing the specific heat [nFaces]
        return:
            F=modeled Euler fluxes [nFaces,mn+nSc]
    """
    nLR = len(rLR)
    nFaces = len(rLR[0])
    nSc = YLR[0].shape[1]
    nDim = mn + nSc

    # compute the wave speeds
    aLR = np.empty((2, nFaces))
    qLR = np.empty((2, nFaces))
    SLR = np.empty((2, nFaces))
    SStar = np.empty(nFaces)
    for iFace in range(nFaces):
        aLR[0, iFace] = np.sqrt(gamma[iFace] * pLR[0, iFace] / rLR[0, iFace])
        aLR[1, iFace] = np.sqrt(gamma[iFace] * pLR[1, iFace] / rLR[1, iFace])
        aBar = 0.5 * (aLR[0, iFace] + aLR[1, iFace])
        pBar = 0.5 * (pLR[0, iFace] + pLR[1, iFace])
        rBar = 0.5 * (rLR[0, iFace] + rLR[1, iFace])
        pPVRS = pBar - 0.5 * (uLR[1, iFace] - uLR[0, iFace]) * rBar * aBar
        pStar = max(0.0, pPVRS)
        qLR[0, iFace] = (
            np.sqrt(
                1.0
                + (gamma[iFace] + 1.0)
                / (2.0 * gamma[iFace])
                * (pStar / pLR[0, iFace] - 1.0)
            )
            if pStar > pLR[0, iFace]
            else 1.0
        )
        qLR[1, iFace] = (
            np.sqrt(
                1.0
                + (gamma[iFace] + 1.0)
                / (2.0 * gamma[iFace])
                * (pStar / pLR[1, iFace] - 1.0)
            )
            if pStar > pLR[1, iFace]
            else 1.0
        )
        SLR[0, iFace] = uLR[0, iFace] - aLR[0, iFace] * qLR[0, iFace]
        SLR[1, iFace] = uLR[1, iFace] + aLR[1, iFace] * qLR[1, iFace]
        SStar[iFace] = pLR[1, iFace] - pLR[0, iFace]
        SStar[iFace] += rLR[0, iFace] * uLR[0, iFace] * (SLR[0, iFace] - uLR[0, iFace])
        SStar[iFace] -= rLR[1, iFace] * uLR[1, iFace] * (SLR[1, iFace] - uLR[1, iFace])
        SStar[iFace] /= rLR[0, iFace] * (SLR[0, iFace] - uLR[0, iFace]) - rLR[
            1, iFace
        ] * (SLR[1, iFace] - uLR[1, iFace])

    # find the regular flux
    FLR = np.empty((2, nFaces, nDim))
    for K in range(nLR):
        for iFace in range(nFaces):
            FLR[K, iFace, 0] = rLR[K, iFace] * uLR[K, iFace]
            FLR[K, iFace, 1] = rLR[K, iFace] * uLR[K, iFace] ** 2.0 + pLR[K, iFace]
            FLR[K, iFace, 2] = uLR[K, iFace] * (
                gamma[iFace] / (gamma[iFace] - 1) * pLR[K, iFace]
                + 0.5 * rLR[K, iFace] * uLR[K, iFace] ** 2.0
            )
            for kSc in range(nSc):
                FLR[K, iFace, 3 + kSc] = (
                    rLR[K, iFace] * uLR[K, iFace] * YLR[K, iFace, kSc]
                )

    # compute the modeled flux
    F = np.empty((nFaces, mn + nSc))
    U = np.empty(mn + nSc)
    UStar = np.empty(mn + nSc)
    YFace = np.empty(nSc)
    for iFace in range(nFaces):
        if SLR[0, iFace] >= 0.0:
            for iDim in range(nDim):
                F[iFace, iDim] = FLR[0, iFace, iDim]
        elif SLR[1, iFace] <= 0.0:
            for iDim in range(nDim):
                F[iFace, iDim] = FLR[1, iFace, iDim]
        else:
            SStarFace = SStar[iFace]
            K = 0 if SStarFace >= 0.0 else 1
            rFace = rLR[K, iFace]
            uFace = uLR[K, iFace]
            pFace = pLR[K, iFace]
            for kSc in range(nSc):
                YFace[kSc] = YLR[K, iFace, kSc]
            gammaFace = gamma[iFace]
            SFace = SLR[K, iFace]
            # conservative variable vector
            U[0] = rFace
            U[1] = rFace * uFace
            U[2] = pFace / (gammaFace - 1.0) + 0.5 * rFace * uFace**2.0
            for kSc in range(nSc):
                U[mn + kSc] = rFace * YFace[kSc]
            # star conservative variable vector
            prefactor = rFace * (SFace - uFace) / (SFace - SStarFace)
            UStar[0] = prefactor
            UStar[1] = prefactor * SStarFace
            UStar[2] = prefactor * (
                U[2] / rFace
                + (SStarFace - uFace) * (SStarFace + pFace / (rFace * (SFace - uFace)))
            )
            for iSp in range(nSc):
                UStar[mn + iSp] = prefactor * YFace[iSp]
            # flux update
            for iDim in range(nDim):
                F[iFace, iDim] = FLR[K, iFace, iDim] + SFace * (UStar[iDim] - U[iDim])

    return F
