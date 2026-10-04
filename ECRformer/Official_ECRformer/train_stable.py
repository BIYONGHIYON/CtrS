"""Opt-in stable training; never resumes optimizer state automatically."""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import Callback

import train
from config.ecrformer_config import ECRformerConfig


class PathDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        paths = train.get_subset_paths(self.dataset, [index])
        sample['sample_paths'] = json.dumps(paths[0] if paths else
                                          {'dataset_index': index}, ensure_ascii=True)
        return sample


class StableModel(train.CloudRemovalModel):
    def __init__(self, config):
        super().__init__(config)
        self.batch_diagnostics = {}
        # Protect both L1 and SSIM, including multi-scale losses, from autocast.
        self.loss_fn = [self.fp32_l1, self.fp32_ssim_loss]

    def fp32_l1(self, prediction, target):
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            return torch.nn.functional.l1_loss(prediction.float(), target.float())

    def fp32_ssim_loss(self, prediction, target):
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            return 1 - self.SSIM(prediction.float(), target.float())

    def fuse_input(self, batch):
        batch, merged, target = super().fuse_input(batch)
        if self.training:
            self.diag_input = merged.detach()
            self.batch_diagnostics = {
                'sample_paths': batch.get('sample_paths', []),
                'input': self.stats(merged), 'target': self.stats(target),
            }
        return batch, merged, target

    @staticmethod
    def stats(tensor):
        tensor = tensor.detach().float()
        finite = torch.isfinite(tensor)
        return {'finite': bool(finite.all()),
                'min': float(tensor.min()) if finite.all() else None,
                'max': float(tensor.max()) if finite.all() else None}

    def forward(self, x, *args, **kwargs):
        result = super().forward(x, *args, **kwargs)
        if self.training:
            self.batch_diagnostics['prediction'] = self.stats(result[0])
        return result

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        loss_value = float(loss.detach())
        self.batch_diagnostics['loss'] = loss_value if math.isfinite(loss_value) else None
        self.batch_diagnostics['batch'] = batch_idx + 1
        if not torch.isfinite(loss) or not all(
                self.batch_diagnostics[key]['finite']
                for key in ('input', 'target', 'prediction')):
            self.write_diagnostic('NONFINITE_BATCH', **self.batch_diagnostics)
            raise FloatingPointError('Non-finite training batch; see stability_diagnostics.jsonl')
        return loss

    def write_diagnostic(self, event, **values):
        row = dict(time=datetime.now().astimezone().isoformat(), event=event,
                   epoch=self.current_epoch, global_step=self.global_step, **values)
        path = Path(self.trainer.log_dir) / 'stability_diagnostics.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False, default=str) + '\n')

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch, merged, target = self.fuse_input(batch)
        prediction, _ = self.forward(merged)
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            metrics = train.compute_metric(prediction.float(), target.float(), size_average=True)
        for name, value in metrics.items():
            self.log('valid_' + name, value, on_epoch=True, prog_bar=name in ('MAE', 'SSIM'))
        self.log('valid_loss', metrics['MAE'] * self.loss_weight[0] +
                 (1 - metrics['SSIM']) * self.loss_weight[1], on_epoch=True)
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'], on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7)
        return {'optimizer': optimizer, 'lr_scheduler': {
            'scheduler': scheduler, 'monitor': 'valid_loss', 'interval': 'epoch'}}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        before = gradient_norm(self.net)
        self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val,
                            gradient_clip_algorithm=gradient_clip_algorithm)
        self.write_diagnostic('OPTIMIZER_GRADIENT', grad_norm_before_clip=before,
                              grad_norm_after_clip=gradient_norm(self.net),
                              lr=optimizer.param_groups[0]['lr'])


def gradient_norm(net):
    norms = [p.grad.detach().float().norm() for p in net.parameters() if p.grad is not None]
    value = float(torch.stack(norms).norm()) if norms else 0.0
    return value if math.isfinite(value) else None


class StabilityLog(Callback):
    def __init__(self, probe_every=200):
        self.probe_every = probe_every

    def on_train_start(self, trainer, module):
        module.write_diagnostic('RUN_SETTINGS', precision=str(trainer.precision),
                                loss_precision='float32', lr=module.lr,
                                scheduler='ReduceLROnPlateau', patience=3,
                                probe_every=self.probe_every, checkpoint=str(trainer.ckpt_path))

    def on_before_optimizer_step(self, trainer, module, optimizer):
        # Lightning calls this after AMP unscale; clipping is logged by the module.
        norm = gradient_norm(module.net)
        if norm is None:
            module.write_diagnostic('NONFINITE_GRADIENT', **module.batch_diagnostics)
            raise FloatingPointError('Non-finite unscaled gradient; stopping before update')

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        metrics = {}
        for key in ('train_MAE', 'train_SSIM', 'down_proj', 'up_proj'):
            value = float(trainer.callback_metrics[key].detach())
            metrics[key] = value if math.isfinite(value) else None
        module.write_diagnostic('TRAIN_BATCH',
                                lr=trainer.optimizers[0].param_groups[0]['lr'],
                                **metrics, **module.batch_diagnostics)
        if self.probe_every and (batch_idx + 1) % self.probe_every == 0:
            self.probe(module, batch_idx)
        # Do not retain the augmented batch beyond this callback.
        module.diag_input = None

    @torch.no_grad()
    def probe(self, module, batch_idx):
        if module.device.type != 'cuda':
            module.write_diagnostic('PRECISION_PROBE_SKIPPED', reason='CUDA required')
            return
        started = time.perf_counter()
        modes = [(layer, layer.training) for layer in module.net.modules()]
        try:
            # Paired eval probes avoid changing BatchNorm statistics or dropout state.
            module.net.eval()
            with torch.random.fork_rng(devices=[module.device.index or 0]):
                with torch.autocast(device_type='cuda', enabled=False):
                    full = module.net(module.diag_input.float())[0].float()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    mixed = module.net(module.diag_input.float())[0].float()
            difference = (full - mixed).abs()
            finite = bool(torch.isfinite(difference).all())
            module.write_diagnostic('PRECISION_PROBE', batch=batch_idx + 1,
                                    fp32=module.stats(full), fp16=module.stats(mixed),
                                    abs_diff_mean=float(difference.mean()) if finite else None,
                                    abs_diff_max=float(difference.max()) if finite else None,
                                    elapsed_seconds=time.perf_counter() - started)
        finally:
            for layer, mode in modes:
                layer.training = mode

    def on_validation_end(self, trainer, module):
        if not trainer.sanity_checking:
            values = {key: float(value.detach()) for key, value in trainer.callback_metrics.items()
                      if key.startswith('valid_')}
            values = {key: value if math.isfinite(value) else None for key, value in values.items()}
            module.write_diagnostic('VALIDATION', **values)

    def on_exception(self, trainer, module, exception):
        module.write_diagnostic('EXCEPTION', error=repr(exception))


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--name', required=True, help='New experiment name, distinct from old runs')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--precision', choices=['16-mixed', '32-true'], default='16-mixed')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--max-train-samples', type=int, default=6000)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--probe-every', type=int, default=200, help='0 disables paired eval probes')
    parser.add_argument('--init-weights', help='Weights only; starts a fresh optimizer and scheduler')
    args = parser.parse_args(argv)
    if args.lr <= 0 or args.probe_every < 0 or args.num_workers < 0 or args.max_epochs <= 0:
        parser.error('Invalid learning rate, probe interval, workers, or epochs')
    config = ECRformerConfig()
    config.name = args.name
    config.no_resume = True
    config.dataset.root = args.data_root
    config.train.gpu = [args.gpu]
    config.train.lr = args.lr
    config.train.train_bs = 2
    config.train.valid_bs = 1
    config.train.num_workers = args.num_workers
    config.train.max_epoch = args.max_epochs
    config.train.max_train_samples = args.max_train_samples
    config.train.init_weights_path = args.init_weights
    config.train.model_class = 'train_stable.StableModel'
    config.train.dataset_wrapper = 'train_stable.PathDataset'
    config.train.extra_callbacks = [{'class': 'train_stable.StabilityLog',
                                    'kwargs': {'probe_every': args.probe_every}}]
    config.optim.precision = args.precision
    config.optim.accumulate_grad_batches = 8
    return config


if __name__ == '__main__':
    train.config_name = 'ecrformer_stable'
    train.main(parse_config())
