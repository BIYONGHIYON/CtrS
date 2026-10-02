"""Check the new interpolation, single-color SSA wiring and scene geometry."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ssamrn.metrics import _interp23tap
from ssamrn.models.interp23 import interp23tap
from ssamrn.models.rgb_grouped import RGBGroupedSsaMRN, RGBGroupedCore
from ssamrn.data.lib_hsi import LIBHSI, downsample
from ssamrn.data.registration import estimate_projective, warp_projective, edges, score


class GroupedProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_23tap_matches_reference_channelwise_and_backward(self):
        x = torch.rand(2, 3, 7, 9, requires_grad=True)
        for ratio in (2, 4):
            actual = interp23tap(x, ratio)
            expected = np.stack([np.stack([_interp23tap(a.detach().numpy(), ratio) for a in item]) for item in x])
            np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-6, rtol=2e-6)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        isolated = torch.zeros(1, 3, 8, 8)
        isolated[:, 1, 3, 4] = 1
        enlarged = interp23tap(isolated)
        self.assertEqual(enlarged[:, [0, 2]].abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_23tap_and_area_cuda_match_cpu(self):
        x = torch.rand(1, 12, 16, 16)
        torch.testing.assert_close(interp23tap(x.cuda()).cpu(), interp23tap(x), atol=3e-6, rtol=2e-5)
        gt = torch.rand(204, 32, 32)
        torch.testing.assert_close(downsample(gt.cuda(), 'area').cpu(), downsample(gt, 'area'))

    def test_single_color_grouped_attention_matches_twelve_branches_and_gradients(self):
        torch.manual_seed(12)
        model = RGBGroupedCore()
        loop = copy.deepcopy(model)
        rgb, features, ms = torch.rand(2, 3, 8, 12), torch.rand(2, 12, 8, 12), torch.rand(2, 12, 8, 12)
        actual = model._attend(rgb, features, ms, model.SSA_blocks, model.conv48tnum1, model.prelu1)
        outputs = [block(rgb[:, band//4:band//4+1], torch.cat([features, ms[:, band:band+1]], 1))
                   for band, block in enumerate(loop.SSA_blocks)]
        expected = loop.prelu1(loop.conv48tnum1(torch.cat(outputs, 1)))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for (name, a), (_, b) in zip(model.named_parameters(), loop.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=3e-4, msg=name)

    def test_full_forward_has_204_outputs_and_no_resize_interpolation(self):
        model = RGBGroupedSsaMRN()
        rgb = torch.rand(1, 3, 16, 16, requires_grad=True)
        lr = torch.rand(1, 204, 4, 4, requires_grad=True)
        with patch('torch.nn.functional.interpolate', side_effect=AssertionError('Unexpected interpolation')):
            result = model(rgb, lr)
            self.assertEqual(result.shape, (1, 204, 16, 16))
            result.square().mean().backward()
        for gradient in (rgb.grad, lr.grad, model.encoder.weight.grad, model.decoder.weight.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        # Each compressed feature uses exactly its corresponding 17 spectral bands.
        with torch.no_grad():
            model.encoder.weight.fill_(1)
            model.encoder.bias.zero_()
            lr.zero_(); lr[:, 17:34] = 1
            encoded = model.encoder(lr)
        self.assertTrue((encoded[:, 1] == 17).all())
        self.assertEqual(encoded[:, [0, *range(2, 12)]].abs().sum().item(), 0)

    def test_four_training_quadrants_and_one_whole_scene_evaluation(self):
        field = np.repeat(np.repeat(np.array([[.1, .3], [.5, .7]], dtype='float32'), 256, 0), 256, 1)
        cube = np.broadcast_to(field[:, :, None], (512, 512, 204))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ('train', 'validation'):
                (root/split/'rgb').mkdir(parents=True)
                Image.fromarray((np.repeat(field[:, :, None], 3, 2)*255).astype('uint8')).save(root/split/'rgb/scene.png')
            with patch('ssamrn.data.lib_hsi.cube_view', return_value=cube), patch.object(LIBHSI, '_read_scene', return_value=cube):
                options = dict(root=root, patch_size=256, allow_incomplete=True, train_layout='quadrants',
                               eval_layout='full256', degradation='area')
                train, val = LIBHSI(split='train', **options), LIBHSI(split='validation', **options)
                self.assertEqual(len(train), 4)
                self.assertEqual(len(val), 1)
                for index, level in enumerate((.1, .3, .5, .7)):
                    sample = train[(index, 0)]
                    self.assertEqual(sample['lr_hsi'].shape, (204, 64, 64))
                    self.assertAlmostEqual(sample['gt'].mean().item(), level, places=5)
                sample = val[0]
                self.assertEqual(sample['gt'].shape, (204, 256, 256))
                for (y, x), level in zip(((0, 0), (0, 255), (255, 0), (255, 255)), (.1, .3, .5, .7)):
                    self.assertAlmostEqual(sample['gt'][0, y, x].item(), level, places=5)
                self.assertEqual(sample['lr_hsi'].shape, (204, 64, 64))

    def test_homography_direction_and_tilt_recovery(self):
        rng = np.random.default_rng(31)
        texture = gaussian_filter(rng.random((256, 256)).astype('float32'), 1)
        reference = np.repeat(texture[:, :, None], 3, 2)
        source_to_target = np.array([[1.002, -.014, 2.5], [.012, .997, -2.1], [.00002, -.00003, 1.]])
        moving = cv2.warpPerspective(reference, np.linalg.inv(source_to_target), (256, 256), flags=cv2.INTER_LANCZOS4)
        record = estimate_projective(reference, moving)
        self.assertTrue(record['projective_accepted'], record)
        aligned, valid = warp_projective(moving, record['matrix'])
        self.assertGreater(score(edges(reference)[24:-24, 24:-24], edges(aligned)[24:-24, 24:-24]), .95)
        self.assertGreater(valid.mean(), .85)
        # A positive x translation maps moving x=50 to fixed x=53.
        marker = np.zeros((64, 64, 3), 'float32'); marker[30, 50] = 1
        shifted, mask = warp_projective(marker, [[1, 0, 3], [0, 1, -2], [0, 0, 1]])
        self.assertAlmostEqual(float(shifted[28, 53, 0]), 1., places=5)
        self.assertTrue(mask[28, 53]); self.assertFalse(mask[0].any())

    def test_unrelated_images_do_not_get_projective_warp(self):
        rng = np.random.default_rng(42)
        result = estimate_projective(rng.random((256, 256, 3)), rng.random((256, 256, 3)))
        self.assertFalse(result['projective_accepted'])
        self.assertEqual(result['transform'], 'identity')


if __name__ == '__main__':
    unittest.main()
