"""Invariant checks for the paper-comparison metrics."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ssamrn.metrics import fr_metrics, q2n, rr_metrics  # noqa: E402


class MetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.image = rng.uniform(100, 900, (4, 64, 64))
        self.pan = rng.uniform(100, 900, (64, 64))

    def test_rr_identity(self):
        result = rr_metrics(self.image, self.image, peak=1023)
        self.assertAlmostEqual(result["SAM"], 0, places=5)
        self.assertEqual(result["ERGAS"], 0)
        self.assertAlmostEqual(result["SCC"], 1)
        self.assertAlmostEqual(result["Q2n"], 1)

    def test_q2n_sensitive_to_change(self):
        altered = self.image.copy()
        altered[0] = np.roll(altered[0], 8, axis=0)
        self.assertLess(q2n(self.image, altered), 1)

    def test_fr_identity_has_zero_spectral_distortion(self):
        result = fr_metrics(self.image, self.image, self.image[:, ::4, ::4], self.pan)
        self.assertAlmostEqual(result["D_lambda"], 0)
        self.assertAlmostEqual(result["QNR"], (1 - result["D_s"]))


if __name__ == "__main__":
    unittest.main()
