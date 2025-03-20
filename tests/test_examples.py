from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from examples import laminar_flame, optimization
from examples.validation import case1, case2, case3, case4

_directory_prefix = Path("tests/resources")

IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skip validation tests.")
def test_validation_case1():
    results = case1.main(plot_results=False, results_location=None)
    baseline = np.load(_directory_prefix / "case1.npz")
    assert all(np.allclose(results[name], baseline[name]) for name in baseline)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skip validation tests.")
def test_validation_case2():
    results = case2.main(plot_results=False, results_location=None)
    baseline = np.load(_directory_prefix / "case2.npz")
    assert all(np.allclose(results[name], baseline[name]) for name in baseline)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skip validation tests.")
def test_validation_case3():
    results = case3.main(plot_results=False, results_location=None)
    baseline = np.load(_directory_prefix / "case3.npz")
    assert all(np.allclose(results[name], baseline[name]) for name in baseline)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skip validation tests.")
def test_validation_case4():
    results = case4.main(plot_results=False, results_location=None)
    baseline = np.load(_directory_prefix / "case4.npz")
    assert all(np.allclose(results[name], baseline[name]) for name in baseline)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skip examples.")
def test_laminar_flame():
    results = laminar_flame.main(
        sim_time=1e-5, plot_results=False, results_location=None
    )
    baseline = np.load(_directory_prefix / "laminarFlame.npz")
    assert all(np.allclose(results[name], baseline[name]) for name in baseline)


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Skip examples.")
def test_optimization():
    results = optimization.main(plot_results=False, results_location=None)
    baseline = np.load(_directory_prefix / "optimization.npz")
    assert all(np.allclose(results[name], baseline[name]) for name in baseline)
