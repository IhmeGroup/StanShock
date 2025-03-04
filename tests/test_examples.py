from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from examples import laminar_flame

_directory_prefix = Path("tests/resources")


def test_laminar_flame():
    with TemporaryDirectory(dir=_directory_prefix) as temp_dir:
        laminar_flame.main(sim_time=1e-5, plot_results=False, results_location=temp_dir)
        results = np.load(Path(temp_dir) / "laminarFlame.npz")
        baseline = np.load(_directory_prefix / "laminarFlame.npz")
        assert all(np.allclose(results[name], baseline[name]) for name in baseline)
