from __future__ import annotations

from typing import ClassVar

import numpy as np

from stanshock.physics.fluid_base import FluidState
from stanshock.system.backend import Array
from stanshock.system.geometry import Geometry


def interpolate(x_array: Array, q_array: Array, x: float) -> float:
    """
    helper function for the probe
    """
    idx: int = np.where(x_array < x)[0][-1]
    x_upper = float(x_array[idx + 1])
    x_lower = float(x_array[idx])
    q_upper = float((q_array[x_array >= x])[0])
    q_lower = float((q_array[x_array < x])[-1])
    return q_lower + (q_upper - q_lower) / (x_upper - x_lower) * (x - x_lower)


class Probe:
    """
    This class is used to store the relevant data for the probe
    """

    n_probes: ClassVar[int] = 0

    def __init__(
        self,
        geometry: Geometry,
        probe_location: float,
        skip_steps: int = 0,
        name: str | None = None,
    ) -> None:
        self.geometry = geometry
        self.probe_location = probe_location
        if probe_location > np.max(self.geometry.xf) or probe_location < np.min(
            self.geometry.xf
        ):
            msg = "Invalid Probe Location"
            raise ValueError(msg)

        self.skip_steps = skip_steps  # number of timesteps to skip
        self.name = f"probe{Probe.n_probes:03d}" if name is None else name

        # Increment the number of probes currently instantiated
        Probe.n_probes += 1

        self.t: list[float] = []  # time
        self.r: list[float] = []  # density
        self.u: list[float] = []  # velocity
        self.p: list[float] = []  # pressure
        self.gamma: list[float] = []  # specific heat ratio
        self.Y: list[Array] = []  # scalars

    def update(self, time: float, state: FluidState) -> None:
        assert state.density is not None
        assert state.velocity is not None
        assert state.pressure is not None
        assert state.gamma is not None
        assert state.composition is not None
        xc = self.geometry.xc
        n_scalars = state.composition.shape[1]
        self.t.append(time)
        self.r.append(interpolate(xc, state.density, self.probe_location))
        self.u.append(interpolate(xc, state.velocity, self.probe_location))
        self.p.append(interpolate(xc, state.pressure, self.probe_location))
        self.gamma.append(interpolate(xc, state.gamma, self.probe_location))
        YProbe = np.array(
            [
                interpolate(
                    xc,
                    state.composition[:, kSp],
                    self.probe_location,
                )
                for kSp in range(n_scalars)
            ]
        )
        self.Y.append(YProbe)
