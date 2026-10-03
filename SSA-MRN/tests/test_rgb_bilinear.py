"""Match the reproduced PAN-MS core resampling while retaining 23-tap LMS."""
import unittest
import torch
from ssamrn.models.rgb_grouped import ScalarGrouped12Core, ScalarGrouped34Core, build_rgb_hsi_model
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
from ssamrn.models.interp23 import interp23tap


class BilinearProtocolTests(unittest.TestCase):
    channels = 34
    core_type = ScalarGrouped34Core
    def test_all_used_resizers_equal_pan_ms_reproduction(self):
        actual = self.core_type('bilinear')
        reference = RestoredPansharpeningNet(self.channels, 4, guide_channels=1)
        x = torch.rand(1, 2, 12, 16)
        for name in ('upsample1', 'upsample100', 'upsample101', 'upsample102',
                     'downsample1', 'downsample2', 'downsample100', 'downsample200'):
            torch.testing.assert_close(getattr(actual, name)(x), getattr(reference, name)(x), rtol=0, atol=0)

    def test_full_core_matches_reproduced_operations(self):
        torch.manual_seed(5)
        actual = self.core_type('bilinear').eval()
        reference = RestoredPansharpeningNet(self.channels, 4, guide_channels=1).eval()
        reference.load_state_dict(actual.state_dict())
        guide = torch.rand(1, 1, 24, 32, requires_grad=True)
        lms = torch.rand(1, self.channels, 24, 32, requires_grad=True)
        ms = torch.rand(1, self.channels, 6, 8, requires_grad=True)
        a, b = actual(guide, lms, ms), reference(guide, lms, ms)
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        ga = torch.autograd.grad(a.square().mean(), (guide, lms, ms), retain_graph=True)
        gb = torch.autograd.grad(b.square().mean(), (guide, lms, ms))
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-6)

    def test_baseline_stays_23tap_and_forward_backward_is_finite(self):
        config = dict(model_type=f'rgb_triple_grouped{self.channels}_bilinear', latent_channels=self.channels,
                      ssai_dimension=4, upsampler='23tap')
        model = build_rgb_hsi_model(config)
        rgb, lr = torch.rand(1, 3, 32, 32), torch.rand(1, 204, 8, 8)
        output = model(rgb, lr)
        self.assertEqual(output.shape, (1, 204, 32, 32))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        for core in model.cores:
            self.assertGreater(core.SSA_blocks[self.channels-1].conv1t6.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            model.fusion.weight.zero_()
            model.fusion.bias.zero_()
            torch.testing.assert_close(model(rgb, lr), interp23tap(lr, 4), rtol=0, atol=0)


class Bilinear12ProtocolTests(BilinearProtocolTests):
    channels = 12
    core_type = ScalarGrouped12Core


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
