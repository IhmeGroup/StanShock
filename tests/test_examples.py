from __future__ import annotations

from pathlib import Path

import numpy as np

from examples import laminar_flame

_directory_prefix = Path("tests/resources")


def test_laminar_flame():
    flame = laminar_flame.main(sim_time=1e-5, plot_results=False, results_location=None)
    results = {
        "position": flame.x,
        "temperature": flame.thermoTable.get_temperature(flame.r, flame.p, flame.Y),
    }
    baseline = np.load(_directory_prefix / "laminarFlame.npz")
    assert all(np.allclose(results[name], baseline[name]) for name in baseline)
