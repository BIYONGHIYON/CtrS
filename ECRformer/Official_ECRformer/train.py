import os
import json
import importlib
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser, Namespace

from util.pytorch_ssim import SSIM
from util.util import count_parameters, initialize_weights, compute_metric
from util.augment import TestAugment, TrainAugment
from util.checkpoint import (
    extract_state_dict,
    find_latest_checkpoint,
    load_checkpoint_file,
)
from util.data_split import build_train_valid_datasets

from models import find_model_using_name
from data import find_dataset_using_name
from config import find_config_using_name


class CloudRemovalModel(pl.LightningModule):
    """PyTorch Lightning module for cloud removal with ECRformer.

    Supports multi-scale projection loss (down_proj + up_proj).
    """

    def __init__(self, config: Namespace):
        super().__init__()
        self.config = config
        self.input = config.net.input
        self.output = config.net.output

        net_class = find_model_using_name(config.net.name)
        self.net = net_class(**config.net.cfg)

        self.SSIM = SSIM()
        self.downsample = nn.AvgPool2d(2, 2)
        def loss_SSIM(x1, x2): return 1 - self.SSIM(x1, x2)
        self.loss_fn = [nn.L1Loss(), loss_SSIM]

        self.loss_weight = list(config.train.loss_weight[:2])
        if len(self.loss_weight) != 2:
            raise ValueError(
                f'loss_weight must provide at least two entries, got {len(config.train.loss_weight)}'
            )

        self.proj_weight = config.train.proj_weight

        self.lr = config.train.lr
        self.train_augment = TrainAugment(crop_size=config.dataset.crop_size)
        self.save_hyperparameters()

        initialize_weights(self.net)

        down_weight, up_weight = self.proj_weight
        if down_weight == 0.:
            self.net.down_proj.requires_grad_(False)
        if up_weight == 0.:
            self.net.up_proj.requires_grad_(False)

    def forward(self, x, *args, **kwargs):
        return self.net(x, *args, **kwargs)

    @torch.no_grad()
    def fuse_input(self, batch):
        """Fuse input channels and prepare target tensor."""
        if self.training:
            batch = self.train_augment.augment(batch)
        target = torch.cat([batch[key] for key in self.output], dim=1)
        merged = torch.cat([batch[key] for key in self.input], dim=1)
        return batch, merged, target

    def training_step(self, batch, batch_idx):
        batch, merged, target = self.fuse_input(batch)
        pred, projs = self.forward(merged)

        down_projs, up_projs = projs

        # Multi-Scale Feature Regularization
        down_loss_sum, up_loss_sum = 0, 0
        mid_target = target
        for down_proj, up_proj in zip(down_projs, up_projs[::-1]):
            down_loss = [fn(down_proj, mid_target) for fn in self.loss_fn]
            down_loss = sum(l * w for l, w in zip(down_loss, [0.1, 0.9]))
            down_loss_sum = down_loss_sum + down_loss

            up_loss = [fn(up_proj, mid_target) for fn in self.loss_fn]
            up_loss = sum(l * w for l, w in zip(up_loss, [0.9, 0.1]))
            up_loss_sum = up_loss_sum + up_loss

            mid_target = self.downsample(mid_target)

        down_loss = down_loss_sum / (len(down_projs) + 1e-3)
        up_loss = up_loss_sum / (len(up_projs) + 1e-3)

        # Main loss
        loss_list = [fn(pred, target) for fn in self.loss_fn]
        loss = sum(l * w for l, w in zip(loss_list, self.loss_weight))

        # Combined loss
        down_weight, up_weight = self.proj_weight
        loss = loss * (1 - down_weight - up_weight) + \
            down_loss * down_weight + up_loss * up_weight

        self.log('train_MAE', loss_list[0], prog_bar=True)
        self.log('train_SSIM', 1 - loss_list[1], prog_bar=True)
        self.log('down_proj', down_loss, prog_bar=True)
        self.log('up_proj', up_loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch, merged, target = self.fuse_input(batch)
        pred, projs = self.forward(merged)

        metrics = compute_metric(pred, target, size_average=True)
        self.log('valid_RMSE', metrics['RMSE'], prog_bar=False, on_epoch=True)
        self.log('valid_MAE', metrics['MAE'], prog_bar=True, on_epoch=True)
        self.log('valid_PSNR', metrics['PSNR'], prog_bar=False, on_epoch=True)
        self.log('valid_SAM', metrics['SAM'], prog_bar=False, on_epoch=True)
        self.log('valid_SSIM', metrics['SSIM'], prog_bar=True, on_epoch=True)
        self.log('valid_LPIPS', metrics['LPIPS'], prog_bar=False, on_epoch=True)
        valid_loss = metrics['MAE'] * self.loss_weight[0] + \
            (1 - metrics['SSIM']) * self.loss_weight[1]
        self.log('valid_loss', valid_loss, prog_bar=False, on_epoch=True)
        if len(self.trainer.optimizers) > 0:
            self.log('learning_rate',
                     self.trainer.optimizers[0].param_groups[0]['lr'],
                     prog_bar=False, on_epoch=True)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        batch, merged, target = self.fuse_input(batch)
        pred_result = self.forward(merged)
        return batch, pred_result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5, patience=10)
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'valid_loss',
        #         'interval': 'epoch',
        #         'frequency': 1,
        #     },
        # }

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[120, 150, 170, 180, 190, 200], gamma=0.5)
        return [optimizer], [scheduler]


# ---------------------------------------------------------------------------
# Training Entry Point
# ---------------------------------------------------------------------------

def select_fixed_subset(dataset, max_samples, seed):
    if max_samples is None:
        return dataset, None
    if not 0 < max_samples <= len(dataset):
        raise ValueError(
            f'max_train_samples must be between 1 and {len(dataset)}, '
            f'got {max_samples}'
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices), indices


def get_subset_paths(dataset, indices):
    # Resolve indices through single-split and other nested Subset wrappers.
    while isinstance(dataset, Subset):
        indices = [int(dataset.indices[i]) for i in indices]
        dataset = dataset.dataset
    raw_dataset = getattr(dataset, 'dataset', dataset)
    paths = getattr(raw_dataset, 'paths', None)
    return [paths[i] for i in indices] if paths is not None else None


def save_subset_manifest(dataset, indices, config, split_info, log_dir):
    if indices is None:
        return
    manifest = {
        'seed': config.seed,
        'data_root': str(config.dataset.root),
        'split': split_info['train'],
        'num_samples': len(indices),
        'indices': indices,
        'paths': get_subset_paths(dataset, indices),
    }
    manifest_path = Path(log_dir) / 'train_subset.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing != manifest:
            raise ValueError(
                'The saved training subset differs from this run. '
                'Use the original dataset and sample count, or start a new experiment.'
            )
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Training subset manifest: {manifest_path}')


def resolve_training_extension(spec):
    """Keep optional extension specifications serializable in checkpoints."""
    if isinstance(spec, str):
        module, name = spec.rsplit('.', 1)
        return getattr(importlib.import_module(module), name)
    if isinstance(spec, dict):
        return resolve_training_extension(spec['class'])(**spec.get('kwargs', {}))
    return spec


def main(config):
    torch.set_float32_matmul_precision("highest")
    pl.seed_everything(config.seed)

    print("\nBuilding model...")
    model_class = resolve_training_extension(
        getattr(config.train, 'model_class', CloudRemovalModel))
    model = model_class(config)
    init_weights_path = getattr(config.train, 'init_weights_path', None)
    if init_weights_path:
        if config.train.ckpt_path is not None:
            raise ValueError('Use either init_weights_path or ckpt_path, not both.')
        checkpoint = load_checkpoint_file(init_weights_path, map_location='cpu')
        model.load_state_dict(extract_state_dict(checkpoint), strict=True)
        print(f'Initialized model weights from: {init_weights_path}')
    print(f"Model class: {model.net.__class__.__name__}")
    count_parameters(model)

    print("\nLoading dataset...")
    dataset_class = find_dataset_using_name(config.dataset.name)
    train_dataset, valid_dataset, split_info = build_train_valid_datasets(
        config, dataset_class)
    full_train_dataset = train_dataset
    train_dataset, selected_train_indices = select_fixed_subset(
        train_dataset, getattr(config.train, 'max_train_samples', None), config.seed)
    print(f'Training samples used: {len(train_dataset)}')
    print(f'Validation samples used: {len(valid_dataset)}')
    print(f"Training split: {split_info['train']}")
    print(f"Validation split: {split_info['valid']}")

    dataset_wrapper = getattr(config.train, 'dataset_wrapper', None)
    if dataset_wrapper is not None:
        train_dataset = resolve_training_extension(dataset_wrapper)(train_dataset)

    num_workers = config.train.num_workers
    train_loader = DataLoader(
        train_dataset, batch_size=config.train.train_bs, drop_last=True,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0)
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.train.valid_bs,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0)

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss', verbose=False, mode='min',
        auto_insert_metric_name=True, save_last=True, save_top_k=1)
    early_stop_callback = EarlyStopping(
        monitor='valid_loss', patience=config.train.early_stop,
        verbose=True, mode='min')

    log_name = config_name
    if config.name:
        log_name = f"{log_name}_{config.name}"

    save_dir = config.train.save_dir

    ckpt_path = config.train.ckpt_path
    resume_version = None

    if init_weights_path:
        print('Starting a new fine-tuning run with fresh optimizer and callbacks.')
    elif ckpt_path is None and not getattr(config, 'no_resume', False):
        auto_ckpt_path, version_num = find_latest_checkpoint(
            save_dir, log_name)
        if auto_ckpt_path is not None:
            ckpt_path = auto_ckpt_path
            resume_version = version_num
            print(f"Auto-resume checkpoint: {ckpt_path}")
        else:
            print("No checkpoint found. Training from scratch.")
    elif getattr(config, 'no_resume', False):
        print("Auto-resume is disabled.")
        ckpt_path = None

    if resume_version is not None:
        tb_logger = TensorBoardLogger(
            save_dir=save_dir, name=log_name, version=resume_version)
    else:
        tb_logger = TensorBoardLogger(save_dir=save_dir, name=log_name)

    save_subset_manifest(
        full_train_dataset, selected_train_indices, config, split_info, tb_logger.log_dir)

    print("\nCreating trainer...")
    trainer = pl.Trainer(
        max_epochs=config.train.max_epoch,
        gradient_clip_val=.5,
        callbacks=[checkpoint_callback, early_stop_callback] +
                  [resolve_training_extension(spec) for spec in
                   getattr(config.train, 'extra_callbacks', [])],
        logger=[tb_logger],
        devices=config.train.gpu,
        **config.optim.__dict__,
    )

    print("\nStarting training...")
    print(f"Experiment directory: {os.path.join(save_dir, log_name)}")
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=valid_loader, ckpt_path=ckpt_path)
    print("\nTraining finished.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default='ecrformer', help="Config name")
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--no-resume', action='store_true',
                        help="Disable automatic checkpoint resume")
    parser.add_argument('--data-root', type=str,
                        help='Override the dataset root directory')
    parser.add_argument('--init-weights', type=str,
                        help='Initialize model weights without resuming optimizer or callbacks')
    parser.add_argument('--max-epochs', type=int,
                        help='Override the maximum number of epochs')
    parser.add_argument('--max-train-samples', type=int,
                        help='Select a fixed number of training samples for this run')
    parser.add_argument('--num-workers', type=int,
                        help='Override the number of data-loading workers')
    parser.add_argument('--lr', type=float,
                        help='Override the initial learning rate')
    args = parser.parse_args()

    config_name = args.config
    print(f"Config: {config_name}")
    config_class = find_config_using_name(config_name)
    config = config_class()
    config.name = args.name
    config.train.gpu = [args.gpu]
    config.no_resume = args.no_resume
    if args.data_root is not None:
        config.dataset.root = args.data_root
    if args.init_weights is not None:
        config.train.init_weights_path = args.init_weights
    if args.max_epochs is not None:
        config.train.max_epoch = args.max_epochs
    if args.max_train_samples is not None:
        config.train.max_train_samples = args.max_train_samples
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers
    if args.lr is not None:
        config.train.lr = args.lr

    main(config)
