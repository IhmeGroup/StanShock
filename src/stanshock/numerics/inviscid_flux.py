from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numba import double, njit

from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, TypeAlias, Unpack
from stanshock.system.base import PrecomputeStepName, PrecomputeSteps, RightHandSide

# Global variables (parameters) used by the solver
mn = 2  # number of 1D Euler equations

# Type signatures for numba
double1D = double[:]
double2D = double[:, :]
double3D = double[:, :, :]

# Signature for Riemann solvers
RiemannSolver: TypeAlias = Callable[[Array, Array, Array, Array, Array, Array], Array]


@njit(double2D(double2D, double2D, double2D, double3D, double1D, double1D))
def lax_friedrichs_flux(
    rLR: Array, uLR: Array, pLR: Array, YLR: Array, gamma: Array, e0: Array
) -> Array:
    """
    This method computes the flux at each interface
        inputs:
            rLR=array containing left and right density states [nLR,nFaces]
            uLR=array containing left and right velocity states [nLR,nFaces]
            pLR=array containing left and right pressure states [nLR,nFaces]
            YLR=array containing left and right scalar states
                [nLR,nFaces,nSc]
            gamma=array containing the specific heat [nFaces]
            eo=array containing the reference internal energy [nFaces]
        return:
            F=modeled Euler fluxes [nFaces,mn+nSc]
    """
    nLR, nFaces, nSc = YLR.shape
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
            FLR[K, iFace, 0] = rLR[K, iFace] * uLR[K, iFace] ** 2.0 + pLR[K, iFace]
            FLR[K, iFace, 1] = uLR[K, iFace] * (
                gamma[iFace] / (gamma[iFace] - 1) * pLR[K, iFace]
                + rLR[K, iFace] * (e0[iFace] + 0.5 * uLR[K, iFace] ** 2.0)
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
            U[K, 0] = rLR[K, iFace] * uLR[K, iFace]
            U[K, 1] = pLR[K, iFace] / (gamma[iFace] - 1.0) + rLR[K, iFace] * (
                e0[iFace] + 0.5 * uLR[K, iFace] ** 2.0
            )
            for kSc in range(nSc):
                U[K, mn + kSc] = rLR[K, iFace] * YLR[K, iFace, kSc]
        for iDim in range(nDim):
            FBar = 0.5 * (FLR[0, iFace, iDim] + FLR[1, iFace, iDim])
            F[iFace, iDim] = FBar - 0.5 * lambdaMax * (U[1, iDim] - U[0, iDim])
    return F


@njit(double2D(double2D, double2D, double2D, double3D, double1D, double1D))
def hllc_flux(
    rLR: Array, uLR: Array, pLR: Array, YLR: Array, gamma: Array, e0: Array
) -> Array:
    """
    This method computes the flux at each interface
        inputs:
            rLR=array containing left and right density states [nLR,nFaces]
            uLR=array containing left and right velocity states [nLR,nFaces]
            pLR=array containing left and right pressure states [nLR,nFaces]
            YLR=array containing left and right scalar states
                [nLR,nFaces,nSc]
            gamma=array containing the specific heat [nFaces]
            eo=array containing the reference internal energy [nFaces]
        return:
            F=modeled Euler fluxes [nFaces,mn+nSc]
    """
    nLR, nFaces, nSc = YLR.shape
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
            FLR[K, iFace, 0] = rLR[K, iFace] * uLR[K, iFace] ** 2.0 + pLR[K, iFace]
            FLR[K, iFace, 1] = uLR[K, iFace] * (
                gamma[iFace] / (gamma[iFace] - 1) * pLR[K, iFace]
                + rLR[K, iFace] * (e0[iFace] + 0.5 * uLR[K, iFace] ** 2.0)
            )
            for kSc in range(nSc):
                FLR[K, iFace, mn + kSc] = (
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
            U[0] = rFace * uFace
            U[1] = pFace / (gammaFace - 1.0) + rFace * (e0[iFace] + 0.5 * uFace**2.0)
            for kSc in range(nSc):
                U[mn + kSc] = rFace * YFace[kSc]
            # star conservative variable vector
            prefactor = rFace * (SFace - uFace) / (SFace - SStarFace)
            UStar[0] = prefactor * SStarFace
            UStar[1] = prefactor * (
                U[1] / rFace
                + (SStarFace - uFace) * (SStarFace + pFace / (rFace * (SFace - uFace)))
            )
            for iSp in range(nSc):
                UStar[mn + iSp] = prefactor * YFace[iSp]
            # flux update
            for iDim in range(nDim):
                F[iFace, iDim] = FLR[K, iFace, iDim] + SFace * (UStar[iDim] - U[iDim])

    return F


def hllc_flux_vectorized(
    rLR: Array, uLR: Array, pLR: Array, YLR: Array, gamma: Array, e0: Array
) -> Array:
    """Vectorized version of HLLC flux based on DoubleFlux-1D implementation."""
    nLR, nFaces = rLR.shape

    aLR = np.sqrt(gamma[None, :] * pLR / rLR)
    ELR = pLR / (gamma[None, :] - 1.0) + rLR * (e0[None, :] + 0.5 * uLR**2)

    FLR = np.concatenate(
        (
            (rLR * uLR**2 + pLR)[..., None],
            (uLR * (ELR + pLR))[..., None],
            (rLR * uLR)[..., None] * YLR,
        ),
        axis=2,
    )

    WLR = np.concatenate(
        (
            (rLR * uLR)[..., None],
            ELR[..., None],
            rLR[..., None] * YLR,
        ),
        axis=2,
    )

    sLR = np.empty((2, nFaces))
    rBar = 0.5 * (rLR[0] + rLR[1])
    aBar = 0.5 * (aLR[0] + aLR[1])
    pStar = 0.5 * (pLR[0] + pLR[1]) - 0.5 * (uLR[1] - uLR[0]) * rBar * aBar
    qLR = np.ones((2, nFaces))
    for K in range(nLR):
        idx = np.where(pStar > pLR[K])[0]
        qLR[K, idx] = np.sqrt(
            1.0
            + (gamma[idx] + 1.0) / (2.0 * gamma[idx]) * (pStar[idx] / pLR[K, idx] - 1.0)
        )
    sLR[0] = uLR[0] - aLR[0] * qLR[0]
    sLR[1] = uLR[1] + aLR[1] * qLR[1]

    sM = (
        pLR[0]
        - pLR[1]
        - rLR[0] * uLR[0] * (sLR[0] - uLR[0])
        + rLR[1] * uLR[1] * (sLR[1] - uLR[1])
    ) / (rLR[1] * (sLR[1] - uLR[1]) - rLR[0] * (sLR[0] - uLR[0]))
    pM = rLR[1] * (uLR[1] - sLR[1]) * (uLR[1] - sM) + pLR[1]

    WMLR = (
        np.concatenate(
            (
                ((sLR - uLR) * rLR * uLR + (pM[None, :] - pLR))[..., None],
                ((sLR - uLR) * ELR - pLR * uLR + (pM * sM)[None, :])[..., None],
                ((sLR - uLR) * rLR)[..., None] * YLR,
            ),
            axis=2,
        )
        / (sLR - sM[None, :])[..., None]
    )

    sminus = np.minimum(sLR[0], 0.0)
    splus = np.maximum(sLR[1], 0.0)

    return 0.5 * (1.0 + np.sign(sM)[..., None]) * (
        FLR[0] + sminus[:, None] * (WMLR[0] - WLR[0])
    ) + 0.5 * (1.0 - np.sign(sM)[..., None]) * (
        FLR[1] + splus[:, None] * (WMLR[1] - WLR[1])
    )


class InviscidFlux(RightHandSide):
    PRECOMPUTE_STEPS: tuple[PrecomputeStepName, ...] = (
        "boundary_conditions",
        "face_extrapolator",
        "geometry",
        "physics",
    )

    def __init__(
        self,
        riemann_solver: RiemannSolver,
        **precompute_steps: Unpack[PrecomputeSteps],
    ) -> None:
        super().__init__(**precompute_steps)
        assert self.geometry is not None
        self.riemann_solver = riemann_solver
        self.dx: Array | float = self.geometry.dx
        if isinstance(self.dx, np.ndarray):
            self.dx = self.dx[self.geometry.idx_cells]

    def source_implementation(
        self,
        time: float,
        state_array: Array | None,
        state: FluidState | None,
        face_states: FluidState | None,
        avg_face_states: FluidState | None,
        face_gradients: FluidState | None,
    ) -> Array:
        _ = state_array, state, avg_face_states, face_gradients
        assert face_states is not None
        assert face_states.density is not None
        assert face_states.velocity is not None
        assert face_states.pressure is not None
        assert face_states.composition is not None
        assert face_states.gamma_star is not None
        assert face_states.e0_star is not None

        # Flux from face on cell's left side, using double-flux variables
        # from cell on the face's right side
        left_face_flux: Array = self.riemann_solver(
            face_states.density,
            face_states.velocity,
            face_states.pressure,
            face_states.composition,
            face_states.gamma_star[1, :],
            face_states.e0_star[1, :],
        )
        # Flux from face on cell's right side, using double-flux variables
        # from cell on the face's left side
        right_face_flux: Array = self.riemann_solver(
            face_states.density,
            face_states.velocity,
            face_states.pressure,
            face_states.composition,
            face_states.gamma_star[0, :],
            face_states.e0_star[0, :],
        )

        # Allow any SpecifiedFlux BCs to directly override face fluxes
        if self.boundary_conditions is not None:
            left_face_flux = self.boundary_conditions.update_face_flux(
                time, left_face_flux
            )
            right_face_flux = self.boundary_conditions.update_face_flux(
                time, right_face_flux
            )

        return cast(
            Array,
            (left_face_flux[:-1, :] - right_face_flux[1:, :])
            / self.dx,  # Central difference
        )
