from __future__ import annotations

import numpy as np

from stanshock.physics.fluid_base import FluidState


def viscous_flux(domain, rLR, uLR, pLR, YLR):
    """
    This method computes the viscous flux at each interface
        inputs:
            rLR=array containing left and right density states [nLR,nFaces]
            uLR=array containing left and right velocity states [nLR,nFaces]
            pLR=array containing left and right pressure states [nLR,nFaces]
            YLR=array containing left and right scalar states
                [nLR,nFaces,nSp]
        return:
            f=modeled viscous fluxes [nFaces,mn+nSp]
    """
    # get the temperature, pressure, and composition for each cell (including the two ghosts)
    nT = domain.n + 2
    mn = domain.mn

    physics = domain.physics
    state = FluidState(
        shape=nT,
        density=np.concatenate((rLR[0, :], rLR[1, [-1]])),
        pressure=np.concatenate((pLR[0, :], pLR[1, [-1]])),
        velocity=np.concatenate((uLR[0, :], uLR[1, [-1]])),
        composition=np.concatenate((YLR[0, :, :], YLR[1, [-1], :]), axis=0),
    )
    state.temperature = T = physics.get_temperature(state)
    state.density = None

    F = np.ones(nT)
    F[1:-1] = domain.F
    F[0], F[-1] = domain.F[0], domain.F[-1]  # no gradient in F at boundary

    mu = physics.get_mu(state)
    k = physics.get_thermal_conductivity(state) * F
    diff = physics.get_mass_diffusivity(state) * F[:, None]

    # compute the gas properties at the face
    viscosity = (mu[1:] + mu[:-1]) / 2.0
    conductivity = (k[1:] + k[:-1]) / 2.0
    diffusivities = (diff[1:, :] + diff[:-1, :]) / 2.0
    r = ((rLR[0, :] + rLR[1, :]) / 2.0).reshape(-1, 1)
    # get the central differences
    dudx = (uLR[1, :] - uLR[0, :]) / domain.dx
    dTdx = (T[1:] - T[:-1]) / domain.dx
    dYdx = (YLR[1, :, :] - YLR[0, :, :]) / domain.dx
    # compute the fluxes
    f = np.zeros((nT - 1, mn + domain.n_scalars))
    f[:, 1] = 4.0 / 3.0 * viscosity * dudx
    f[:, 2] = conductivity * dTdx
    f[:, mn:] = r * diffusivities * dYdx
    return f
