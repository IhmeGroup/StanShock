import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np

from examples.validation import case1, case2, case3, case4
from examples import laminarFlame, optimization


class TestExamples(unittest.TestCase):
    _directory_prefix = "test/resources"

    def test_validation_case1(self):
        with TemporaryDirectory(dir=self._directory_prefix) as temp_dir:
            case1.main(show_results=False, results_location=temp_dir)
            results_filename = os.path.join(temp_dir, "case1.npz")
            baseline_filename = os.path.join(self._directory_prefix, "case1.npz")
            self.assertResultsCloseToBaseline(results_filename, baseline_filename)

    def test_validation_case2(self):
        with TemporaryDirectory(dir=self._directory_prefix) as temp_dir:
            case2.main(show_results=False, results_location=temp_dir)
            results_filename = os.path.join(temp_dir, "case2.npz")
            baseline_filename = os.path.join(self._directory_prefix, "case2.npz")
            self.assertResultsCloseToBaseline(results_filename, baseline_filename)

    def test_validation_case3(self):
        with TemporaryDirectory(dir=self._directory_prefix) as temp_dir:
            case3.main(show_results=False, results_location=temp_dir)
            results_filename = os.path.join(temp_dir, "case3.npz")
            baseline_filename = os.path.join(self._directory_prefix, "case3.npz")
            self.assertResultsCloseToBaseline(results_filename, baseline_filename)

    def test_validation_case4(self):
        with TemporaryDirectory(dir=self._directory_prefix) as temp_dir:
            case4.main(show_results=False, results_location=temp_dir)
            results_filename = os.path.join(temp_dir, "case4.npz")
            baseline_filename = os.path.join(self._directory_prefix, "case4.npz")
            self.assertResultsCloseToBaseline(results_filename, baseline_filename)

    def test_laminar_flame(self):
        with TemporaryDirectory(dir=self._directory_prefix) as temp_dir:
            laminarFlame.main(show_results=False, results_location=temp_dir)
            results_filename = os.path.join(temp_dir, "laminarFlame.npz")
            baseline_filename = os.path.join(self._directory_prefix, "laminarFlame.npz")
            self.assertResultsCloseToBaseline(results_filename, baseline_filename)

    def test_optimization(self):
        with TemporaryDirectory(dir=self._directory_prefix) as temp_dir:
            optimization.main(show_results=False, results_location=temp_dir)
            results_filename = os.path.join(temp_dir, "optimization.npz")
            baseline_filename = os.path.join(self._directory_prefix, "optimization.npz")
            self.assertResultsCloseToBaseline(results_filename, baseline_filename)

    def assertResultsCloseToBaseline(self, results_filename: str, baseline_filename: str, msg=None):
        # Loads the results and baseline npz files and checks that they are close
        results = np.load(results_filename)
        baseline = np.load(baseline_filename)
        for name in baseline:
            if name not in results:
                msg = self._formatMessage(msg, f"'{name}' not found in results.")
                raise self.failureException(msg)
            if not np.allclose(baseline[name], results[name]):
                msg = self._formatMessage(msg, f"'{name}' is not found to be close to the baseline.")
                raise self.failureException(msg)


if __name__ == '__main__':
    unittest.main()
