import csv
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from config import find_config_using_name
from data import find_dataset_using_name
from train import CloudRemovalModel
from util.checkpoint import (
    extract_state_dict,
    find_latest_checkpoint,
    load_checkpoint_file,
)
from util.data_split import build_eval_dataset, resolve_eval_split_name
from util.util import compute_metric


DEFAULT_PNG_BRIGHTNESS = 3.0


def sanitize_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in {'-', '_', '.'} else '_' for ch in name)


def resolve_log_name(config_name, run_name=None):
    return f"{config_name}_{run_name}" if run_name else config_name


def resolve_sample_name(dataset, idx: int) -> str:
    if isinstance(dataset, Subset):
        return resolve_sample_name(dataset.dataset, dataset.indices[idx])

    source = None

    if hasattr(dataset, 'paths_list'):
        source = dataset.paths_list[idx]
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'paths'):
        source = dataset.dataset.paths[idx]

    if isinstance(source, np.ndarray):
        source = source.item() if source.ndim == 0 else source.tolist()

    if isinstance(source, dict):
        source = source.get('S2') or source.get('S2_cloudy') or source.get('S1')
    elif isinstance(source, (list, tuple)):
        source = next((item for item in source if isinstance(item, str) and item), None)

    if isinstance(source, str) and source:
        return sanitize_name(Path(source).stem)

    return f"sample_{idx:05d}"


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample['index'] = idx
        sample['sample_id'] = resolve_sample_name(self.dataset, idx)
        return sample


def get_device(gpu: int):
    if torch.cuda.is_available() and gpu >= 0:
        return torch.device(f'cuda:{gpu}')
    return torch.device('cpu')


def move_batch_to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_model(config, ckpt_path: str, device: torch.device):
    checkpoint = load_checkpoint_file(ckpt_path, map_location='cpu')
    state_dict = extract_state_dict(checkpoint)
    model = CloudRemovalModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def to_rgb_image(image: np.ndarray, bands, brightness=1.0):
    if image.ndim != 3:
        raise ValueError(f'Expected CHW image, got shape {image.shape}')
    if max(bands) >= image.shape[0]:
        raise ValueError(f'RGB bands {bands} exceed channel count {image.shape[0]}')

    rgb = np.transpose(image[list(bands)], (1, 2, 0))
    rgb = np.clip(rgb * brightness, 0.0, 1.0)
    return (rgb * 255.0).round().astype(np.uint8)


def save_prediction_npz(file_path: Path, pred: np.ndarray, index: int, sample_id: str):
    np.savez_compressed(
        file_path,
        pred=pred.astype(np.float32),
        index=np.int32(index),
        sample_id=np.array(sample_id),
    )


def save_prediction_png(file_path: Path, pred: np.ndarray, rgb_bands, brightness=DEFAULT_PNG_BRIGHTNESS):
    Image.fromarray(to_rgb_image(pred, rgb_bands, brightness=brightness)).save(file_path)


def write_metrics_csv(file_path: Path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with file_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(file_path: Path, summary):
    with file_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def build_dataloader(config, split_override=None, batch_size=None, num_workers=None, max_samples=None):
    dataset_class = find_dataset_using_name(config.dataset.name)
    split_name, dataset = build_eval_dataset(
        config,
        dataset_class,
        split_override=split_override,
    )

    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    dataset = IndexedDataset(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size or config.train.valid_bs,
        shuffle=False,
        num_workers=num_workers if num_workers is not None else config.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers if num_workers is not None else config.train.num_workers) > 0,
    )
    return split_name, dataset, loader


def run_evaluation(args):
    config_class = find_config_using_name(args.config)
    config = config_class()
    config.name = args.name
    config.train.gpu = [args.gpu]

    split = resolve_eval_split_name(config, args.split)
    log_name = resolve_log_name(args.config, args.name)
    output_dir = Path(args.output_dir or Path('results') / log_name / split)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation split: {split}")
    print(f"Output directory: {output_dir}")

    ckpt_path = args.ckpt_path
    if ckpt_path is None:
        ckpt_path, _ = find_latest_checkpoint(config.train.save_dir, log_name)
    if ckpt_path is None:
        raise FileNotFoundError('Checkpoint not found. Please specify one with --ckpt-path.')
    print(f"Using checkpoint: {ckpt_path}")

    device = get_device(args.gpu)
    _, _, dataloader = build_dataloader(
        config,
        split_override=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(dataloader.dataset)} samples for evaluation.")
    model = load_model(config, ckpt_path, device)

    npz_dir = output_dir / 'pred_npz'
    png_dir = output_dir / 'pred_rgb'
    if args.export_format in {'npz', 'both'}:
        npz_dir.mkdir(parents=True, exist_ok=True)
    if args.export_format in {'png', 'both'}:
        png_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    metric_names = ['RMSE', 'MAE', 'PSNR', 'SAM', 'SSIM', 'LPIPS']
    summary_accumulator = {name: [] for name in metric_names}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            batch = move_batch_to_device(batch, device)
            _, merged, target = model.fuse_input(batch)
            pred_result = model(merged)
            pred = pred_result[0] if isinstance(pred_result, tuple) else pred_result

            metrics = compute_metric(pred, target, size_average=False)
            pred_np = tensor_to_numpy(pred)
            indices = batch['index'].detach().cpu().tolist()
            sample_ids = batch['sample_id']

            for sample_idx, sample_id, sample_pred in zip(indices, sample_ids, pred_np):
                file_stem = f"{sample_idx:05d}_{sanitize_name(sample_id)}"
                if args.export_format in {'npz', 'both'}:
                    save_prediction_npz(npz_dir / f'{file_stem}.npz', sample_pred, sample_idx, sample_id)
                if args.export_format in {'png', 'both'}:
                    save_prediction_png(
                        png_dir / f'{file_stem}.png',
                        sample_pred,
                        args.rgb_bands,
                        brightness=args.png_brightness,
                    )

            batch_size = pred.shape[0]
            for item_idx in range(batch_size):
                row = {
                    'index': int(indices[item_idx]),
                    'sample_id': sample_ids[item_idx],
                }
                for key in metric_names:
                    row[key] = float(metrics[key][item_idx].item())
                metric_rows.append(row)
                for key in summary_accumulator:
                    summary_accumulator[key].append(row[key])

    summary = {
        'config': args.config,
        'run_name': args.name,
        'split': split,
        'checkpoint': str(ckpt_path),
        'num_samples': len(metric_rows),
        'metrics': {
            key: float(np.mean(values)) if values else None
            for key, values in summary_accumulator.items()
        },
        'export_format': args.export_format,
        'rgb_bands': list(args.rgb_bands),
        'png_brightness': float(args.png_brightness),
    }

    write_metrics_csv(output_dir / 'metrics.csv', metric_rows)
    write_summary_json(output_dir / 'summary.json', summary)

    print(f"Saved metrics to: {output_dir / 'metrics.csv'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='ecrformer', help='Config name')
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--split', type=str, default=None, help='Dataset split to evaluate')
    parser.add_argument('--ckpt-path', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory for evaluation outputs')
    parser.add_argument('--export-format', choices=['none', 'npz', 'png', 'both'], default='both',
                        help='Prediction export format')
    parser.add_argument('--rgb-bands', nargs=3, type=int, default=[3, 2, 1],
                        help='RGB band indices for PNG export')
    parser.add_argument('--png-brightness', type=float, default=DEFAULT_PNG_BRIGHTNESS,
                        help='Brightness multiplier for exported PNG images')
    parser.add_argument('--batch-size', type=int, default=None, help='Evaluation batch size')
    parser.add_argument('--num-workers', type=int, default=None, help='Number of dataloader workers')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit the number of evaluated samples')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU index, use a negative value for CPU')
    return parser.parse_args()


if __name__ == '__main__':
    run_evaluation(parse_args())