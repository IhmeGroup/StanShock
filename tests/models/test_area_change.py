from __future__ import annotations

from pathlib import Path

import numpy as np
from cantera import Solution

from stanshock.models.area_change import AreaChange
from stanshock.numerics.time_integration import FastSlowIntegrator
from stanshock.physics.cantera_interface import CanteraInterface
from stanshock.processing.initialize import InitializeConstant
from stanshock.system.backend import Array
from stanshock.system.geometry import initialize_geometry

data_dir = Path(__file__).resolve().parent / ".." / ".." / "data"


def test_area_change_handles_partial_moving_area_region() -> None:
    nf = 21
    nc = nf - 1
    xf = np.linspace(0.0, 1.0, nf, dtype=np.float64)

    def dlnA_dt(_time: float, x: Array) -> Array:
        out = np.zeros_like(x)
        out[(x > 0.3) & (x < 0.75)] = 10.0
        return out

    geometry = initialize_geometry(xf=xf, area=1.0, dlnA_dt=dlnA_dt)
    gas = Solution(data_dir / "mechanisms" / "N2O2HeAr.yaml")
    gas.TPX = 300.0, 101325.0, {"N2": 0.79, "O2": 0.21}
    physics = CanteraInterface(gas)
    state = InitializeConstant(geometry, physics, gas, 50.0)()
    state_array = np.ravel(physics.primitive_to_conservative(state))
    gamma_star, e0_star = physics.get_double_flux_variables(state)

    # Initialize area change source terms
    area_change = AreaChange(geometry=geometry, physics=physics)
    assert len(area_change.x) == nc

    # Get reference values
    area_change.update_indices(0.0, state)
    source = dlnA_dt(0.0, area_change.x)
    nfast = int(np.count_nonzero(source))
    nslow = nc - nfast

    # Check fast source terms
    area_change.mode = "fast"
    state_compact, gamma_star_in, e0_star_in = area_change.before_time_integration(
        0.0, state_array, gamma_star, e0_star
    )
    assert len(state_compact) == nfast * 3
    rhs_fast = area_change.source_full(0.0, state_compact, gamma_star_in, e0_star_in)
    assert len(rhs_fast) == nfast * 3

    # Check slow source terms
    area_change.mode = "slow"
    state_compact, gamma_star_in, e0_star_in = area_change.before_time_integration(
        0.0, state_array, gamma_star, e0_star
    )
    assert len(state_compact) == nslow * 3
    rhs_slow = area_change.source_full(0.0, state_compact, gamma_star_in, e0_star_in)
    assert len(rhs_slow) == nslow * 3

    # Check single time integration step
    time_integrator = FastSlowIntegrator(area_change)
    _, state_array_next, _, _ = time_integrator.advance(
        1e-7,
        0.0,
        state_array,
        gamma_star,
        e0_star,
    )

    assert isinstance(area_change.idx_input_implicit, np.ndarray)
    assert 0 < len(area_change.idx_input_implicit) < geometry.n_cells_interior
    assert state_array_next.shape == state_array.shape
    assert np.all(np.isfinite(state_array_next))
