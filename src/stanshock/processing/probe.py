from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np

from stanshock.physics.fluid_base import FluidPhysics, FluidState
from stanshock.processing.plot import VariableInfo, get_variable_info_map
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
    interpolator: Callable[[Array], float]

    def __init__(
        self,
        geometry: Geometry,
        physics: FluidPhysics,
        probe_location: float,
        skip_steps: int = 0,
        name: str | None = None,
        plot_variables: list[str] | None = None,
        variable_info_map: dict[str, VariableInfo] | None = None,
    ) -> None:
        self.probe_location = probe_location
        if probe_location > np.max(geometry.xf) or probe_location < np.min(geometry.xf):
            msg = "Invalid Probe Location"
            raise ValueError(msg)

        self.skip_steps = skip_steps  # number of timesteps to skip
        self.name = f"probe{Probe.n_probes:03d}" if name is None else name

        # Increment the number of probes currently instantiated
        Probe.n_probes += 1

        # Set default variables to plot
        if plot_variables is None:
            plot_variables = ["r", "u", "p", "g"] + [
                scalar for scalar in physics.scalar_names if scalar != "density"
            ]
        if variable_info_map is None:
            variable_info_map = get_variable_info_map(physics)
        self.plot_variables = ["t", *plot_variables]
        self.variable_info = [variable_info_map[k] for k in plot_variables]

        # Set the initial time index
        self.idx = 0
        self.data = np.full((1000, 1 + len(self.plot_variables)), np.nan)

        # Set up the interpolating function
        self.interpolator = lambda x: interpolate(geometry.xc, x, self.probe_location)

    def __getattr__(self, name) -> Array:
        if name in self.plot_variables:
            return self.data[: self.idx, self.plot_variables.index(name)]
        msg = f"Variable {name} not requested for probe."
        raise AttributeError(msg)

    def update(self, time: float, state: FluidState) -> None:
        # Expand the data array if needed
        if self.idx >= self.data.shape[0]:
            self.data = np.pad(self.data, ((0, 1000), (0, 0)), constant_values=np.nan)

        # Fill the current row
        self.data[self.idx, 0] = time
        for i, vinfo in enumerate(self.variable_info):
            self.data[self.idx, 1 + i] = self.interpolator(vinfo.fun(state))

        self.idx += 1
