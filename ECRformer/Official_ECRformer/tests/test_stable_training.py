"""CPU-only tests. No Trainer.fit, GPU allocation, or real dataset access."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import Subset
import train_stable as stable


class Toy(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.9))

    def forward(self, x):
        y = x[:, 2:] * self.weight
        return y, ([y], [y])


class RawDataset:
    paths = [{'S1': 'one.tif'}, {'S1': 'two.tif'}]
    def __len__(self): return 2
    def __getitem__(self, index): return {'target': torch.zeros(13, 16, 16)}


class StableTests(unittest.TestCase):
    def config(self):
        return stable.parse_config(['--data-root', 'unused', '--name', 'cpu_test'])

    def model(self):
        with patch.object(stable.train, 'find_model_using_name', return_value=Toy), \
                patch.object(stable.train, 'initialize_weights'):
            model = stable.StableModel(self.config())
        model.train_augment = SimpleNamespace(augment=lambda batch: batch)
        model.log = lambda *args, **kwargs: None
        return model

    def test_defaults_and_precision(self):
        c = self.config()
        self.assertEqual(c.train.lr, 1e-4)
        self.assertTrue(c.no_resume)
        self.assertEqual(c.train.train_bs * c.optim.accumulate_grad_batches, 16)
        self.assertEqual(stable.parse_config(['--data-root', 'x', '--name', 'y',
                                            '--precision', '32-true']).optim.precision, '32-true')

    def test_path_mapping(self):
        dataset = stable.PathDataset(Subset(Subset(RawDataset(), [1, 0]), [0]))
        self.assertEqual(json.loads(dataset[0]['sample_paths']), {'S1': 'two.tif'})

    def test_extension_resolution(self):
        c = self.config()
        self.assertIs(stable.train.resolve_training_extension(c.train.model_class), stable.StableModel)
        self.assertIsInstance(stable.train.resolve_training_extension(c.train.extra_callbacks[0]),
                              stable.StabilityLog)

    def test_config_serialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.pt'
            torch.save(self.config(), path)
            restored = torch.load(path, weights_only=False)
            self.assertEqual(restored.train.model_class, 'train_stable.StableModel')

    def test_validation_uses_float32_metrics(self):
        model = self.model()
        model.eval()
        model._trainer = SimpleNamespace(optimizers=[torch.optim.SGD(model.parameters(), lr=1e-4)])
        def metrics(prediction, target, **kwargs):
            self.assertEqual(prediction.dtype, torch.float32)
            self.assertEqual(target.dtype, torch.float32)
            return {'MAE': torch.tensor(0.1), 'SSIM': torch.tensor(0.9)}
        with patch.object(stable.train, 'compute_metric', side_effect=metrics):
            model.validation_step({'SAR': torch.zeros(1, 2, 16, 16),
                                   'cloudy': torch.zeros(1, 13, 16, 16),
                                   'target': torch.zeros(1, 13, 16, 16)}, 0)

    def test_loss_and_backward_cpu(self):
        model = self.model()
        target = torch.rand(1, 13, 16, 16)
        loss = model.training_step({'SAR': torch.rand(1, 2, 16, 16),
                                    'cloudy': target, 'target': target,
                                    'sample_paths': ['example.tif']}, 0)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(model.net.weight.grad))
        self.assertTrue(model.batch_diagnostics['prediction']['finite'])

    def test_scheduler_reduces_before_epoch120(self):
        optim = self.model().configure_optimizers()
        scheduler = optim['lr_scheduler']['scheduler']
        for _ in range(5): scheduler.step(1.0)
        self.assertAlmostEqual(optim['optimizer'].param_groups[0]['lr'], 5e-5)

    def test_nonfinite_logging_and_stop(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as folder:
            model._trainer = SimpleNamespace(log_dir=folder, current_epoch=0, global_step=0)
            with self.assertRaises(FloatingPointError):
                model.training_step({'SAR': torch.zeros(1, 2, 16, 16),
                                     'cloudy': torch.full((1, 13, 16, 16), float('nan')),
                                     'target': torch.zeros(1, 13, 16, 16)}, 2)
            record = json.loads((Path(folder) / 'stability_diagnostics.jsonl').read_text())
            self.assertEqual(record['event'], 'NONFINITE_BATCH')
            self.assertIsNone(record['loss'])


if __name__ == '__main__': unittest.main()
