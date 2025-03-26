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
        if probeLocation > np.max(domain.x) or probeLocation < np.min(domain.x):
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
        self.t.append(domain.t)
        self.r.append(interpolate(domain.x, domain.r, self.probeLocation))
        self.u.append(interpolate(domain.x, domain.u, self.probeLocation))
        self.p.append(interpolate(domain.x, domain.p, self.probeLocation))
        self.gamma.append(interpolate(domain.x, domain.gamma, self.probeLocation))
        YProbe = np.array(
            [
                (interpolate(domain.x, domain.Y[:, kSp], self.probeLocation))
                for kSp in range(domain.n_scalars)
            ]
        )
        self.Y.append(YProbe)
