"""Pilot compatibility, gradients, masking and controlled comparison checks."""
import json
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from ssamrn.models.rgb_grouped import RGBTripleJointSsaMRN
from ssamrn.models.rgb_pilot import RGBPilotJoint17, GroupGlobalEncoder, LocalRGBAlignment
from ssamrn.models.area_consistency import project_area_consistency
from ssamrn.pilot_metrics import gradient_error_stats
from diagnose_rgb_pilot import perturb_rgb, CommonMask


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_unchanged_pilot_matches_original_after_loading_weights(self):
        torch.manual_seed(7)
        original = RGBTripleJointSsaMRN().eval()
        with torch.no_grad():
            original.decoder[-1].weight.normal_(std=.001)
        candidate = RGBPilotJoint17().eval()
        candidate.load_state_dict(original.state_dict(), strict=True)
        rgb, lr = torch.rand(1, 3, 32, 32), torch.rand(1, 204, 8, 8)
        with torch.no_grad():
            self.assertTrue(torch.equal(original(rgb, lr), candidate(rgb, lr)))

    def test_variants_have_finite_output_and_module_gradients(self):
        for option, module_name in [({'attention_mode': 'aligned_spatial'}, 'cores'),
                                   ({'detail_path': True}, 'detail_path'),
                                   ({'global_encoder': True}, 'encoder'),
                                   ({'local_alignment': True}, 'local_alignment')]:
            with self.subTest(option=option):
                model = RGBPilotJoint17(option)
                with torch.no_grad():
                    model.decoder[-1].weight.normal_(std=.001)
                    if model.detail_path is not None:
                        model.detail_path.decode.weight.normal_(std=.001)
                    if isinstance(model.encoder, GroupGlobalEncoder):
                        model.encoder.mix.weight[:, 17:].normal_(std=.001)
                rgb, lr = torch.rand(1, 3, 32, 32), torch.rand(1, 204, 8, 8)
                mask = torch.ones(1, 32, 32, dtype=torch.bool)
                output = project_area_consistency(model(rgb, lr), lr, mask)
                self.assertEqual(tuple(output.shape), (1, 204, 32, 32))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.allclose(torch.nn.functional.avg_pool2d(output, 4), lr, atol=1e-6))
                output.square().mean().backward()
                params = list(getattr(model, module_name).parameters())
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in params))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_mask_excludes_differences_touching_invalid_pixels(self):
        gt = torch.zeros(1, 204, 4, 4)
        pred = gt.clone()
        pred[:, :, 0, 0] = 100
        mask = torch.ones(1, 4, 4, dtype=torch.bool)
        mask[:, 0, 0] = False
        sse, count = gradient_error_stats(pred, gt, mask)
        self.assertEqual(sse.item(), 0)
        self.assertEqual(count.item(), (24 - 2) * 204)

    def test_common_mask_and_rgb_shift_do_not_wrap(self):
        rgb = torch.arange(4.).reshape(1, 1, 1, 4).expand(1, 3, 4, 4)
        self.assertEqual(perturb_rgb(rgb, 'shift_right1')[0, 0, 0].tolist(), [0., 0., 1., 2.])
        dataset = [{'gt': torch.zeros(204, 16, 16), 'rgb': torch.zeros(3, 16, 16),
                    'valid_mask': torch.ones(16, 16, dtype=torch.bool)}]
        mask = CommonMask(dataset)[0]['valid_mask']
        self.assertEqual(mask.sum().item(), 64)
        self.assertFalse(mask[0].any())

    def test_each_scratch_ablation_changes_one_option(self):
        folder = ROOT / 'configs/pilot'
        base = json.loads((folder / '00_baseline10.json').read_text(encoding='utf-8-sig'))
        for prefix, key in [('03', 'attention_mode'), ('04', 'detail_path'), ('05', 'global_encoder'), ('06', 'local_alignment')]:
            cfg = json.loads(next(folder.glob(prefix + '*.json')).read_text(encoding='utf-8-sig'))
            for field in ('seed', 'epochs', 'patch_size', 'latent_channels', 'ssai_dimension',
                          'train_layout', 'eval_layout', 'degradation', 'learning_rate',
                          'spectral_weight', 'train_area_consistency', 'alignment_manifest'):
                self.assertEqual(cfg[field], base[field])
            options = dict(cfg['pilot_options'])
            options.pop('max_shift', None)
            self.assertEqual(list(options), [key])
        control = json.loads((folder / '02_warm_control10.json').read_text(encoding='utf-8-sig'))
        low = json.loads((folder / '02_warm_low10.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(control['init_checkpoint'], low['init_checkpoint'])
        self.assertEqual(control['learning_rate'] / low['learning_rate'], 10.)


if __name__ == '__main__':
    unittest.main()
