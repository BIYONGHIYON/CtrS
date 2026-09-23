"""Small offline checks for data contracts and the repaired forward path."""

import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssamrn.data.pancollection import PanCollectionH5
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet


class ReproductionTests(unittest.TestCase):
    def test_h5_shapes_and_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.h5"
            with h5py.File(path, "w") as file:
                file["pan"] = np.full((2, 1, 16, 16), 2047.0)
                file["lms"] = np.full((2, 4, 16, 16), 1023.5)
                file["ms"] = np.full((2, 4, 4, 4), 1023.5)
                file["gt"] = np.full((2, 4, 16, 16), 1023.5)
            dataset = PanCollectionH5(path, sensor="QB")
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["pan"].shape, (1, 16, 16))
            self.assertAlmostEqual(dataset[0]["pan"].max().item(), 1.0)
            self.assertAlmostEqual(dataset[0]["gt"].max().item(), 0.5)

    def test_model_forward_and_backward(self):
        for channels in (4, 8):
            with self.subTest(channels=channels):
                model = RestoredPansharpeningNet(channels=channels)
                pan = torch.rand(1, 1, 16, 16)
                lms = torch.rand(1, channels, 16, 16)
                ms = torch.rand(1, channels, 4, 4)
                result = model(pan, lms, ms)
                self.assertEqual(result.shape, lms.shape)
                self.assertTrue(torch.isfinite(result).all())
                result.mean().backward()
                self.assertIsNotNone(model.cov2t64.weight.grad)

    def test_model_rejects_misaligned_inputs(self):
        model = RestoredPansharpeningNet(channels=4)
        with self.assertRaises(ValueError):
            model(torch.rand(1, 1, 16, 16), torch.rand(1, 4, 16, 16),
                  torch.rand(1, 4, 5, 5))


if __name__ == "__main__":
    unittest.main()
