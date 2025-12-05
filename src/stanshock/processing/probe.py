from __future__ import annotations

import numpy as np


def interpolate(xArray, qArray, x):
    """
    helper function for the probe
    """
    xUpper = (xArray[xArray >= x])[0]
    xLower = (xArray[xArray < x])[-1]
    qUpper = (qArray[xArray >= x])[0]
    qLower = (qArray[xArray < x])[-1]
    return qLower + (qUpper - qLower) / (xUpper - xLower) * (x - xLower)


class Probe:
    """
    This class is used to store the relevant data for the probe
    """

    def __init__(self, domain, probeLocation, skipSteps=0, probeName=None):
        self.probeLocation = probeLocation
        geometry = domain.geometry
        if probeLocation > np.max(geometry.xf) or probeLocation < np.min(geometry.xf):
            msg = "Invalid Probe Location"
            raise Exception(msg)

        self.skipSteps = skipSteps  # number of timesteps to skip
        self.name = probeName
        if probeName is None:
            self.name = "probe" + str(len(domain.probes))

        self.t = []
        self.r = []  # density
        self.u = []  # velocity
        self.p = []  # pressure
        self.gamma = []  # specific heat ratio
        self.Y = []  # scalars

    def update(self, domain):
        state = domain.state
        geometry = domain.geometry
        self.t.append(domain.t)
        self.r.append(interpolate(geometry.xc, state.density, self.probeLocation))
        self.u.append(interpolate(geometry.xc, state.velocity, self.probeLocation))
        self.p.append(interpolate(geometry.xc, state.pressure, self.probeLocation))
        self.gamma.append(interpolate(geometry.xc, state.gamma, self.probeLocation))
        YProbe = np.array(
            [
                (
                    interpolate(
                        geometry.xc,
                        state.composition[:, kSp],
                        self.probeLocation,
                    )
                )
                for kSp in range(domain.physics.n_scalars)
            ]
        )
        self.Y.append(YProbe)
