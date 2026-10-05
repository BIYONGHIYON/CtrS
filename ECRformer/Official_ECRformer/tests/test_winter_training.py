"""CPU-only checks: no Trainer.fit, model construction, or TIFF reads."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import Subset
import train
from config import find_config_using_name
from data.sen12mscr_dataset import SEN12MSCR


class WinterTrainingTests(unittest.TestCase):
    def check_layout(self, nested):
        with tempfile.TemporaryDirectory() as folder:
            for kind in ('s1', 's2', 's2_cloudy'):
                seed = 'ROIs1158_spring_' + kind
                root = Path(folder) / seed
                if nested:
                    root = root / seed
                roi = root / (kind + '_106')
                roi.mkdir(parents=True)
                for patch in range(100, 105):
                    (roi / f'{seed}_106_p{patch}.tif').touch()
                (roi / 'ignored.txt').touch()
                (roi / 'ignored_dir').mkdir()
            dataset = SEN12MSCR(folder, split='test', season='spring')
            self.assertEqual(len(dataset), 5)
            self.assertEqual([Path(p['S1']).stem.rsplit('_', 1)[1] for p in dataset.paths],
                             ['p100', 'p101', 'p102', 'p103', 'p104'])
            self.assertTrue(all(Path(p[k]).is_file() for p in dataset.paths
                                for k in ('S1', 'S2', 'S2_cloudy')))

    def test_flat_layout(self):
        self.check_layout(False)

    def test_nested_layout(self):
        self.check_layout(True)

    def test_subset_fixed_and_size_checked(self):
        dataset = list(range(20))
        first, indices = train.select_fixed_subset(dataset, 6, 42)
        second, other = train.select_fixed_subset(dataset, 6, 42)
        self.assertEqual(indices, other)
        self.assertEqual(list(first), list(second))
        self.assertEqual(len(set(indices)), 6)
        for size in (0, -1, 21):
            with self.assertRaises(ValueError):
                train.select_fixed_subset(dataset, size, 42)
        self.assertIs(train.select_fixed_subset(dataset, None, 42)[0], dataset)

    def test_nested_subset_paths(self):
        raw = SimpleNamespace(paths=[{'S1': str(i)} for i in range(10)])
        wrapped = Subset(Subset(raw, [8, 3, 5]), [2, 0])
        self.assertEqual(train.get_subset_paths(wrapped, [0, 1]),
                         [{'S1': '5'}, {'S1': '8'}])

    def test_manifest_rejects_changed_selection(self):
        dataset = list(range(20))
        config = SimpleNamespace(seed=42, dataset=SimpleNamespace(root='example'))
        with tempfile.TemporaryDirectory() as folder:
            train.save_subset_manifest(dataset, [1, 3], config, {'train': 'train'}, folder)
            train.save_subset_manifest(dataset, [1, 3], config, {'train': 'train'}, folder)
            self.assertEqual(json.loads((Path(folder) / 'train_subset.json').read_text())['indices'], [1, 3])
            with self.assertRaises(ValueError):
                train.save_subset_manifest(dataset, [2, 3], config, {'train': 'train'}, folder)

    def test_plateau_after_three_bad_epochs(self):
        dummy = SimpleNamespace(net=torch.nn.Linear(1, 1), lr=4e-4)
        result = train.CloudRemovalModel.configure_optimizers(dummy)
        scheduler = result['lr_scheduler']['scheduler']
        self.assertEqual(result['lr_scheduler']['monitor'], 'valid_loss')
        self.assertEqual(result['lr_scheduler']['interval'], 'epoch')
        scheduler.step(1.0)
        for _ in range(3):
            scheduler.step(1.0)
        self.assertAlmostEqual(result['optimizer'].param_groups[0]['lr'], 4e-4)
        scheduler.step(1.0)
        self.assertAlmostEqual(result['optimizer'].param_groups[0]['lr'], 2e-4)
        scheduler.step(0.5)
        self.assertEqual(scheduler.num_bad_epochs, 0)

    def test_previous_server_settings(self):
        config = find_config_using_name('ecrformer_spring')()
        self.assertEqual(config.train.lr, 4e-4)
        self.assertEqual(config.train.max_epoch, 100)
        self.assertEqual(config.train.max_train_samples, 6000)
        self.assertEqual(config.train.train_bs * config.optim.accumulate_grad_batches, 16)
        self.assertEqual(config.optim.precision, '16-mixed')
        self.assertFalse(hasattr(config.train, 'model_class'))
        self.assertFalse(hasattr(config.train, 'extra_callbacks'))


if __name__ == '__main__':
    unittest.main()
