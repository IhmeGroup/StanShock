from __future__ import annotations

from pathlib import Path

import numpy as np
from cantera import Solution

from stanshock.models.area_change import AreaChange
from stanshock.numerics.time_integration import FastSlowIntegrator
from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.processing.initialize import InitializeConstant
from stanshock.system.geometry import initialize_geometry


def test_area_change_handles_partial_moving_area_region() -> None:
    xf = np.linspace(0.0, 1.0, 21, dtype=np.float64)

    def dlnA_dt(_time: float, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        out[(x > 0.25) & (x < 0.75)] = 10.0
        return out

    geometry = initialize_geometry(xf=xf, area=1.0, dlnA_dt=dlnA_dt)
    gas = Solution(
        Path(__file__).resolve().parent / ".." / "data" / "mechanisms" / "Nitrogen.yaml"
    )
    gas.TP = 300.0, 101325.0
    physics = CanteraInterface(gas)
    state = InitializeConstant(geometry, physics, gas, 50.0)()
    state_array = np.ravel(physics.primitive_to_conservative(state))
    gamma_star, e0_star = physics.get_double_flux_variables(state)

    area_change = AreaChange(geometry=geometry, physics=physics)
    time_integrator = FastSlowIntegrator(area_change)
    _, state_array_next, _, _ = time_integrator.advance(
        1e-7,
        0.0,
        state_array,
        gamma_star,
        e0_star,
    )

    assert 0 < len(area_change.idx_output_implicit) < geometry.n_cells_interior
    assert state_array_next.shape == state_array.shape
    assert np.all(np.isfinite(state_array_next))
