"""Checks for the joint 17-feature decoder and masked LR projection."""
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ssamrn.models.rgb_grouped import build_rgb_hsi_model
from ssamrn.models.area_consistency import project_area_consistency


class Joint17ConsistencyTest(unittest.TestCase):
    def test_joint_decoder_uses_all_rgb_branches_and_backpropagates(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = build_rgb_hsi_model(dict(model_type='rgb_triple_joint17_23tap',
                                         latent_channels=17, ssai_dimension=4, upsampler='23tap'))
        self.assertEqual(model.decoder[0].in_channels, 51)
        self.assertEqual(model.decoder[-1].out_channels, 204)
        rgb = torch.rand(1, 3, 64, 64)
        lr = torch.rand(1, 204, 16, 16)
        # The residual starts at zero, so each RGB branch sees a gradient after
        # the first decoder update. Prime that last layer for this test.
        with torch.no_grad():
            model.decoder[-1].weight.fill_(1e-4)
        output = model(rgb, lr)
        self.assertEqual(tuple(output.shape), (1, 204, 64, 64))
        output.square().mean().backward()
        self.assertTrue(all(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in core.parameters()) for core in model.cores))

    def test_projection_preserves_valid_lr_blocks_and_gradient(self):
        lr = torch.rand(1, 204, 2, 2)
        prediction = torch.rand(1, 204, 8, 8, requires_grad=True)
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        mask[:, 0, 0] = False
        corrected = project_area_consistency(prediction, lr, mask)
        pooled = F.avg_pool2d(corrected, 4)
        self.assertTrue(torch.allclose(pooled[:, :, 1:, :], lr[:, :, 1:, :], atol=1e-6))
        self.assertTrue(torch.allclose(pooled[:, :, :1, 1:], lr[:, :, :1, 1:], atol=1e-6))
        corrected.square().mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == '__main__':
    unittest.main()
