"""Regression checks for three independent guides covering twelve groups each."""
import unittest
import torch
from ssamrn.models.rgb_grouped import RGBTripleGroupedSsaMRN
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet


class TripleGuideTests(unittest.TestCase):
    def test_all_guides_receive_gradient(self):
        torch.manual_seed(7)
        model = RGBTripleGroupedSsaMRN()
        rgb = torch.rand(1, 3, 32, 32, requires_grad=True)
        output = model(rgb, torch.rand(1, 204, 8, 8))
        self.assertEqual(output.shape, (1, 204, 32, 32))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        for i, core in enumerate(model.cores):
            self.assertGreater(rgb.grad[:, i].abs().sum().item(), 0)
            self.assertGreater(core.SSA_blocks[11].conv1t6.weight.grad.abs().sum().item(), 0)

    def test_vectorized_scalar_guide_matches_independent_branches(self):
        core = RGBTripleGroupedSsaMRN().cores[0]
        guide = torch.rand(1, 1, 16, 16)
        features, ms = torch.rand(1, 12, 16, 16), torch.rand(1, 12, 16, 16)
        args = (guide, features, ms, core.SSA_blocks, core.conv48tnum1, core.prelu1)
        torch.testing.assert_close(core._attend(*args),
                                   RestoredPansharpeningNet._attend(core, *args),
                                   rtol=1e-5, atol=1e-6)

    def test_fusion_keeps_band_order(self):
        model = RGBTripleGroupedSsaMRN()
        bands = torch.arange(204.).view(1, 204, 1, 1)
        inputs = torch.stack([bands, bands + 3, bands + 6], dim=2).reshape(1, 612, 1, 1)
        torch.testing.assert_close(model.fusion(inputs), bands + 3)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
