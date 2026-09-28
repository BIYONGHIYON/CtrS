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
from ssamrn.models.ssa_mrn import DirectMLPReLU, RestoredPansharpeningNet


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
            dataset.close()

    def test_model_forward_and_backward(self):
        for channels, dimension in ((4, 4), (4, 6), (8, 6)):
            with self.subTest(channels=channels, dimension=dimension):
                model = RestoredPansharpeningNet(channels=channels, ssai_dimension=dimension)
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

    def test_directml_prelu_matches_standard_prelu(self):
        standard = torch.nn.PReLU()
        compatible = DirectMLPReLU()
        compatible.load_state_dict(standard.state_dict())
        x1 = torch.randn(2, 4, 4, 4)
        x1[0, 0, 0, 0] = 0  # Match PReLU's derivative at the boundary.
        x1.requires_grad_()
        x2 = x1.detach().clone().requires_grad_()
        standard_output = standard(x1)
        compatible_output = compatible(x2)
        torch.testing.assert_close(standard_output, compatible_output)
        standard_output.sum().backward()
        compatible_output.sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad)
        torch.testing.assert_close(standard.weight.grad, compatible.weight.grad)


if __name__ == "__main__":
    unittest.main()
