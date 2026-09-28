from __future__ import annotations

from pathlib import Path

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
        domain,
        filename: str | Path,
        interval: int = 100,
        variables: list[str] | None = None,
        variable_info_map: dict[str, VariableInfo] | None = None,
        region: str = "domain",
    ) -> None:
        """
        Initialize CSV writer.

        Args:
            domain: Component instance
            filename: Base output CSV file path (will be numbered: filename_00000.csv, etc.)
            interval: Write every 'interval' iterations (0 = never)
            wall_temperature: Wall temperature for heat flux calculation
        """
        self.domain = domain
        self.base_filename = Path(filename)

        self.interval = interval
        self.output_counter = 0
        idx_cells = np.arange(domain.geometry.n_cells)[domain.geometry.idx_cells]
        idx_plot = get_plot_cell_mask(domain.geometry, region)
        self.idx = idx_cells[idx_plot]
        self.x = domain.geometry.xc[self.idx]
        if variables is None:
            variables = ["x", "rho", "u", "p", "a", "T"]
        self.headers = variables
        if variable_info_map is None:
            variable_info_map = get_variable_info_map(domain.physics)
        self.variable_info_map = variable_info_map
        self.parent = self.base_filename.parent
        self.stem = self.base_filename.stem
        self.suffix = self.base_filename.suffix
        self.fmt = ["%.4e"] + [variable_info_map[x].fmt for x in self.headers[1:]]

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
        state = self.domain.state[self.idx]

        state_matrix = np.column_stack(
            (self.x, *[self.variable_info_map[v].fun(state) for v in self.headers[1:]])
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
