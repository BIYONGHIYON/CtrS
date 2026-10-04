"""Invariants for x4 synthetic-area data consistency."""

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ssamrn.models.area_consistency import project_area_consistency


class AreaConsistencyTests(unittest.TestCase):
    def test_complete_blocks_match_lr_and_masked_mse_cannot_increase(self):
        generator = torch.Generator().manual_seed(7)
        gt = torch.rand((2, 5, 16, 16), generator=generator)
        lr = F.avg_pool2d(gt, 4)
        prediction = gt + 0.1 * torch.randn(gt.shape, generator=generator)
        valid = torch.ones((2, 16, 16), dtype=torch.bool)
        valid[0, 0, 0] = False
        valid[1, 5, 8] = False

        corrected = project_area_consistency(prediction, lr, valid)
        complete = F.avg_pool2d(valid[:, None].float(), 4).eq(1)
        self.assertTrue(torch.allclose(F.avg_pool2d(corrected, 4)[complete.expand_as(lr)],
                                       lr[complete.expand_as(lr)], atol=1e-6))
        incomplete = (~complete).repeat_interleave(4, -2).repeat_interleave(4, -1)
        self.assertTrue(torch.equal(corrected[incomplete.expand_as(corrected)],
                                    prediction[incomplete.expand_as(prediction)]))
        before = ((prediction - gt).square() * valid[:, None]).sum()
        after = ((corrected - gt).square() * valid[:, None]).sum()
        self.assertLessEqual(after.item(), before.item() + 1e-5)

    def test_zero_correction_and_invalid_shapes(self):
        gt = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8) / 256
        lr = F.avg_pool2d(gt, 4)
        self.assertTrue(torch.allclose(project_area_consistency(gt, lr), gt, atol=1e-6))
        with self.assertRaises(ValueError):
            project_area_consistency(gt, lr[:, :, :1])
        with self.assertRaises(ValueError):
            project_area_consistency(gt, lr, torch.ones((8, 8)))


if __name__ == "__main__":
    unittest.main()
