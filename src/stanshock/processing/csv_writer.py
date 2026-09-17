from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stanshock.processing.plot import (
    VariableInfo,
    get_plot_cell_mask,
    get_variable_info_map,
)


class CSVWriter:
    """
    Writes simulation data to CSV file at specified intervals.
    Creates a new numbered file for each output.
    """

    def __init__(
        self,
        combustor,
        filename: str | Path,
        interval: int = 100,
        variables: list[str] | None = None,
        variable_info_map: dict[str, VariableInfo] | None = None,
        region: str | None = None,
    ) -> None:
        """
        Initialize CSV writer.

        Args:
            combustor: Combustor instance
            filename: Base output CSV file path (will be numbered: filename_00000.csv, etc.)
            interval: Write every 'interval' iterations (0 = never)
            wall_temperature: Wall temperature for heat flux calculation
        """
        self.combustor = combustor
        self.base_filename = Path(filename)

        self.interval = interval
        self.output_counter = 0
        idx_cells = np.arange(combustor.geometry.n_cells)[combustor.geometry.idx_cells]
        idx_plot = get_plot_cell_mask(combustor.geometry, region)
        self.idx = idx_cells[idx_plot]
        self.x = combustor.geometry.xc[self.idx]
        if variables is None:
            variables = ["x", "rho", "u", "p", "a", "T"]
        self.headers = variables
        if variable_info_map is None:
            variable_info_map = get_variable_info_map(combustor.physics)
        self.variable_info_map = variable_info_map
        self.parent = self.base_filename.parent
        self.stem = self.base_filename.stem
        self.suffix = self.base_filename.suffix
        self.fmt = ["%.4e"] + [variable_info_map[x].fmt for x in self.headers[1:]]

    def _get_variable_values(self, variable: str, state: Any) -> np.ndarray:
        variable_info = self.variable_info_map[variable]
        domain_fun = getattr(variable_info, "domain_fun", None)
        if domain_fun is None:
            value = variable_info.fun(state)
        else:
            value = domain_fun(self.combustor, state)

        value = np.asarray(value)
        if value.ndim == 0:
            return np.full(self.x.shape, value)
        return value

    def update(self, iteration: int) -> None:
        """Update CSV with current state if iteration matches interval."""
        # Same condition as plot_state in combustor.py
        if self.interval <= 0:
            return
        if iteration % self.interval != 0:
            return

        self.write_current_state()

    def write_current_state(self) -> None:
        """Write current state to a new numbered CSV file."""

        combustor = self.combustor
        state = combustor.state[self.idx]

        state_matrix = np.column_stack(
            (self.x, *[self._get_variable_values(v, state) for v in self.headers[1:]])
        )

        filename = self.parent / f"{self.stem}_{self.output_counter:05d}{self.suffix}"
        filename.parent.mkdir(parents=True, exist_ok=True)

        np.savetxt(
            filename,
            state_matrix,
            delimiter=",",
            header=",".join(self.headers),
            fmt=self.fmt,
            comments="",
        )

        self.output_counter += 1
