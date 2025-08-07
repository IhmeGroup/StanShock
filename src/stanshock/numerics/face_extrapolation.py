from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numba import double, int16, njit

from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array, Index

# Global variables (parameters) used by the solver
mn = 2  # number of 1D Euler equations

# Type signatures for numba
double1D = double[:]
double2D = double[:, :]
double3D = double[:, :, :]


class FaceExtrapolator(ABC):
    minimum_ghost_layers: int = 1

    def __init__(
        self, n_scalars_rho_sum: int, n_ghost_layers: int = minimum_ghost_layers
    ) -> None:
        """Initialize the face extrapolator with the number of ghost nodes."""
        self.n_scalars_rho_sum = n_scalars_rho_sum
        self.n_ghost_layers = n_ghost_layers

        # Set up slices relating faces to the cells on their left and right
        mt: int = n_ghost_layers
        self.index_face_left: Index = np.s_[mt - 1 : -mt]
        right: int | None = None if mt == 1 else -mt + 1
        self.index_face_right: Index = np.s_[mt:right]

    @abstractmethod
    def __call__(self, state: FluidState) -> FluidState:
        """Extrapolate fluid state from the cells to left and right faces."""

    def add_ghost_layers(self, state: FluidState) -> FluidState:
        """Add ghost layers to the primitive variables."""
        assert state.density is not None
        assert state.velocity is not None
        assert state.pressure is not None
        assert state.composition is not None

        mt: int = self.n_ghost_layers

        gamma_star = (
            None
            if state.gamma_star is None
            else np.pad(state.gamma_star, mt, mode="edge")
        )
        e0_star = (
            None if state.e0_star is None else np.pad(state.e0_star, mt, mode="edge")
        )

        return FluidState(
            shape=(state.shape[0] + 2 * mt,),
            density=np.pad(state.density, mt, mode="edge"),
            velocity=np.pad(state.velocity, mt, mode="edge"),
            pressure=np.pad(state.pressure, mt, mode="edge"),
            composition=np.pad(
                state.composition,
                ((mt, mt), (0, 0)),
                mode="edge",
            ),
            gamma_star=gamma_star,
            e0_star=e0_star,
        )


class FirstOrder(FaceExtrapolator):
    minimum_ghost_layers: int = 1

    def __call__(self, state: FluidState) -> FluidState:
        """First order interpolation of primitive variables to the edge states."""
        assert state.density is not None
        assert state.velocity is not None
        assert state.pressure is not None
        assert state.composition is not None

        n_faces: int = state.shape[0] - 2 * self.n_ghost_layers + 1

        gamma_star = None
        if state.gamma_star is not None:
            gamma_star = np.stack(
                (
                    state.gamma_star[self.index_face_left],
                    state.gamma_star[self.index_face_right],
                ),
                axis=0,
            )

        e0_star = None
        if state.e0_star is not None:
            e0_star = np.stack(
                (
                    state.e0_star[self.index_face_left],
                    state.e0_star[self.index_face_right],
                ),
                axis=0,
            )

        return FluidState(
            shape=(2, n_faces),
            density=np.stack(
                (
                    state.density[self.index_face_left],
                    state.density[self.index_face_right],
                ),
                axis=0,
            ),
            velocity=np.stack(
                (
                    state.velocity[self.index_face_left],
                    state.velocity[self.index_face_right],
                ),
                axis=0,
            ),
            pressure=np.stack(
                (
                    state.pressure[self.index_face_left],
                    state.pressure[self.index_face_right],
                ),
                axis=0,
            ),
            composition=np.stack(
                (
                    state.composition[self.index_face_left],
                    state.composition[self.index_face_right],
                ),
                axis=0,
            ),
            gamma_star=gamma_star,
            e0_star=e0_star,
        )


@njit(double3D(double1D, double1D, double1D, double2D, double1D, int16, int16))
def weno5(
    r: Array,
    u: Array,
    p: Array,
    Y: Array,
    gamma: Array,
    n_ghost_layers: int,
    n_scalars_rho_sum: int,
) -> Array:
    """
    This method implements the fifth-order WENO interpolation. This method
    follows that of Houim and Kuo (JCP2011)
        inputs:
            r=density
            u=velocity
            p=pressure
            Y=scalar variables matrix [x,scalars]
            gamma=specific heat ratio
            n_ghost_layers=number of ghost layers
            n_scalars_rho_sum=number of scalars that are summed into density
        outputs:
            PLR=a matrix of the primitive variables [LR,]
    """
    nLR = 2
    nCells = len(r) - 2 * n_ghost_layers
    nFaces = nCells + 1
    nSc = len(Y[0])  # number of scalars
    nVar = mn + nSc  # [rhou, rhoE, rhoY1, rhoY2, ...]
    nStencil = 2 * n_ghost_layers
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

    B = np.zeros(n_ghost_layers)
    PLR = np.empty((nLR, nFaces, nVar + 1))
    YAverage = np.empty(nSc)
    U = np.empty(nVar)
    R = np.zeros((nVar, nVar))
    L = np.zeros((nVar, nVar))
    CStencil = np.empty((nStencil, nVar))
    # ^ all the characteristic values in the stencil

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

        # Right eigenvector matrix [rhou, rhoE, rhoY1, rhoY2, ...]
        # Acoustic waves (columns 0 and -1)
        R[0, 0] = uAverage - cAverage  # momentum, left acoustic
        R[1, 0] = hAverage - uAverage * cAverage  # energy, left acoustic
        R[0, -1] = uAverage + cAverage  # momentum, right acoustic
        R[1, -1] = hAverage + uAverage * cAverage  # energy, right acoustic

        # Entropy waves (columns 1 to nSp)
        for i in range(nSc):
            R[0, i + 1] = uAverage  # momentum, entropy wave i
            R[1, i + 1] = 0.5 * uAverage**2.0  # energy, entropy wave i
            R[mn + i, i + 1] = 1.0  # scalar i density, entropy wave i

        # Scalar densities for acoustic waves
        for i in range(nSc):
            R[mn + i, 0] = YAverage[i]  # scalar i, left acoustic
            R[mn + i, -1] = YAverage[i]  # scalar i, right acoustic

        # Left eigenvector matrix [rhou, rhoE, rhoY1, rhoY2, ...]
        gammaHat = gammaAverage - 1.0
        phi = 0.5 * gammaHat * uAverage**2.0
        firstRowConstant = 0.5 * (phi + uAverage * cAverage)
        lastRowConstant = 0.5 * (phi - uAverage * cAverage)

        # Acoustic wave rows (rows 0 and -1)
        L[0, 0] = -0.5 * (gammaHat * uAverage + cAverage)  # left acoustic, momentum
        L[0, 1] = gammaHat / 2.0  # left acoustic, energy
        L[-1, 0] = -0.5 * (gammaHat * uAverage - cAverage)  # right acoustic, momentum
        L[-1, 1] = gammaHat / 2.0  # right acoustic, energy

        # Acoustic wave interactions with scalars
        for i in range(nSc):
            L[0, mn + i] = firstRowConstant  # left acoustic, scalar i
            L[-1, mn + i] = lastRowConstant  # right acoustic, scalar i

        # Entropy wave rows (rows 1 to nSp)
        for i in range(nSc):
            L[i + 1, 0] = YAverage[i] * gammaHat * uAverage  # entropy wave i, momentum
            L[i + 1, 1] = -YAverage[i] * gammaHat  # entropy wave i, energy

            # Entropy wave interactions with scalars
            for j in range(nSc):
                L[i + 1, mn + j] = -YAverage[i] * phi  # entropy wave i, scalar j
            L[i + 1, mn + i] = L[i + 1, mn + i] + cAverage**2.0  # diagonal correction

        L /= cAverage**2.0

        # Perform WENO interpolation in characteristic variables
        for iVar in range(nVar):
            for iStencil in range(nStencil):
                iCellStencil = iStencil - 2 + iCell

                # Conservative variables [rhou, rhoE, rhoY1, rhoY2, ...]
                U[0] = r[iCellStencil] * u[iCellStencil]  # momentum
                U[1] = (
                    p[iCellStencil] / (gammaAverage - 1.0)
                    + 0.5 * r[iCellStencil] * u[iCellStencil] ** 2.0
                )  # energy
                for kSc in range(nSc):
                    U[mn + kSc] = (
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
                for iStencil in range(n_ghost_layers):
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
            rLR = 0.0
            for kSc in range(n_scalars_rho_sum):
                rLR += U[mn + kSc]
            uLR = U[0] / rLR
            eLR = U[1] / rLR
            pLR = rLR * (gammaAverage - 1.0) * (eLR - 0.5 * uLR**2.0)

            # Fill primitive matrix [rho, u, p, Y1, Y2, ...]
            PLR[N, iFace, 0] = rLR
            PLR[N, iFace, 1] = uLR
            PLR[N, iFace, 2] = pLR
            for kSc in range(nSc):
                PLR[N, iFace, 3 + kSc] = U[mn + kSc] / rLR

    # First order at boundaries
    for N in range(nLR):
        for iFace in range(n_ghost_layers):
            iCell = iFace + 2
            PLR[N, iFace, 0] = r[iCell + N]
            PLR[N, iFace, 1] = u[iCell + N]
            PLR[N, iFace, 2] = p[iCell + N]
            for kSc in range(nSc):
                PLR[N, iFace, 3 + kSc] = Y[iCell + N, kSc]
        for iFace in range(nFaces - n_ghost_layers, nFaces):
            iCell = iFace + 2
            PLR[N, iFace, 0] = r[iCell + N]
            PLR[N, iFace, 1] = u[iCell + N]
            PLR[N, iFace, 2] = p[iCell + N]
            for kSc in range(nSc):
                PLR[N, iFace, 3 + kSc] = Y[iCell + N, kSc]

    # Create primitive matrix for limiter
    P = np.zeros((nCells + 2 * n_ghost_layers, nVar + 1))
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
            for iVar in range(nVar + 1):
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


class FifthOrderWeno(FaceExtrapolator):
    minimum_ghost_layers: int = 3

    def __call__(self, state: FluidState) -> FluidState:
        """First order interpolation to the edge states."""
        assert state.density is not None
        assert state.velocity is not None
        assert state.pressure is not None
        assert state.composition is not None
        assert state.gamma_star is not None
        assert state.e0_star is not None

        face_states_array = weno5(
            state.density,
            state.velocity,
            state.pressure,
            state.composition,
            state.gamma_star,
            self.n_ghost_layers,
            self.n_scalars_rho_sum,
        )

        n_faces: int = state.shape[0] - 2 * self.n_ghost_layers + 1

        return FluidState(
            shape=(2, n_faces),
            density=face_states_array[:, :, 0],
            velocity=face_states_array[:, :, 1],
            pressure=face_states_array[:, :, 2],
            composition=face_states_array[:, :, 3:],
            gamma_star=np.stack(
                (
                    state.gamma_star[self.index_face_left],
                    state.gamma_star[self.index_face_right],
                ),
                axis=0,
            ),
            e0_star=np.stack(
                (
                    state.e0_star[self.index_face_left],
                    state.e0_star[self.index_face_right],
                ),
                axis=0,
            ),
        )
