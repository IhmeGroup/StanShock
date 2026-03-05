from __future__ import annotations

from pathlib import Path

import numpy as np


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
        self.idx = combustor.geometry.idx_cells
        self.x = combustor.geometry.xc[combustor.geometry.idx_cells]
        self.headers = ["x", "rho", "u", "p", "a", "T"]
        self.parent = self.base_filename.parent
        self.stem = self.base_filename.stem
        self.suffix = self.base_filename.suffix
        self.fmt = [
            "%.4e",  # x
            "%.4e",  # rho
            "%.4e",  # u
            "%.3e",  # p
            "%.3e",  # a
            "%.3e",  # T
        ]

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
        state = combustor.state
        physics = combustor.physics

        rho = state.density[self.idx]
        u = state.velocity[self.idx]
        p = state.pressure[self.idx]

        T = physics.get_temperature(state)[self.idx]
        a = physics.get_sound_speed(state)[self.idx]

        state_matrix = np.column_stack(
            (
                self.x,
                rho,
                u,
                p,
                a,
                T,
            )
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
