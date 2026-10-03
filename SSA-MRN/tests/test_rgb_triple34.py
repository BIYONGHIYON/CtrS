"""Check six-band group boundaries and all 34 scalar-guide branches."""
import unittest
import torch
from ssamrn.models.rgb_grouped import RGBTripleGroupedSsaMRN, build_rgb_hsi_model
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet


class Triple34Tests(unittest.TestCase):
    def test_six_band_groups_keep_spectral_order(self):
        model = RGBTripleGroupedSsaMRN(34)
        with torch.no_grad():
            model.encoder.weight.fill_(1)
            model.encoder.bias.zero_()
        for band in (0, 5, 6, 11, 198, 203):
            cube = torch.zeros(1, 204, 1, 1)
            cube[:, band] = 1
            expected = torch.zeros(1, 34, 1, 1)
            expected[:, band // 6] = 1
            torch.testing.assert_close(model.encoder(cube), expected)
        for decoder in model.decoders:
            with torch.no_grad():
                decoder.weight.fill_(1)
                decoder.bias.zero_()
            features = torch.arange(34.).view(1, 34, 1, 1)
            torch.testing.assert_close(decoder(features), features.repeat_interleave(6, dim=1))

    def test_all_guides_and_last_branch_receive_gradient(self):
        torch.manual_seed(34)
        model = RGBTripleGroupedSsaMRN(34)
        rgb = torch.rand(1, 3, 32, 32, requires_grad=True)
        output = model(rgb, torch.rand(1, 204, 8, 8))
        self.assertEqual(output.shape, (1, 204, 32, 32))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        for i, core in enumerate(model.cores):
            self.assertGreater(rgb.grad[:, i].abs().sum().item(), 0)
            for blocks in (core.SSA_blocks, core.SSA_blocks1, core.SSA_blocks2):
                self.assertEqual(len(blocks), 34)
                self.assertGreater(blocks[33].conv1t6.weight.grad.abs().sum().item(), 0)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_split_projection_matches_original_forward_and_backward(self):
        torch.manual_seed(2)
        core = RGBTripleGroupedSsaMRN(34).cores[0]
        guide = torch.rand(1, 1, 12, 16, requires_grad=True)
        features = torch.rand(1, 34, 12, 16, requires_grad=True)
        ms = torch.rand(1, 34, 12, 16, requires_grad=True)
        args = (guide, features, ms, core.SSA_blocks, core.conv48tnum1, core.prelu1)
        actual = core._attend(*args)
        expected = RestoredPansharpeningNet._attend(core, *args)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        targets = (guide, features, ms, core.SSA_blocks[33].conv7t6_3.weight)
        a = torch.autograd.grad(actual.square().sum(), targets, retain_graph=True)
        b = torch.autograd.grad(expected.square().sum(), targets)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-6)

    def test_factory_requires_matching_protocol(self):
        config = dict(model_type='rgb_triple_grouped34_23tap', latent_channels=34,
                      ssai_dimension=4, upsampler='23tap')
        self.assertEqual(build_rgb_hsi_model(config).encoder.groups, 34)
        with self.assertRaises(ValueError):
            build_rgb_hsi_model(dict(config, latent_channels=12))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
