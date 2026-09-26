"""Convert raw SEN12MS-CR patches to ECRformer NPZ files.

The exported NPZ files are intentionally uncompressed for faster loading and use
float16 arrays to reduce disk usage. A sampling ratio can be provided to create
smaller debug subsets.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from data.sen12mscr_dataset import SEN12MSCR
except ImportError:
    from sen12mscr_dataset import SEN12MSCR


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--root', required=True, help='Path to the raw SEN12MS-CR root directory')
    parser.add_argument('--output-dir', required=True, help='Directory to store generated NPZ files')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Dataset splits to convert')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                        help='Sampling ratio in (0, 1], useful for smaller debug subsets')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for subset sampling')
    parser.add_argument('--data-range', type=float, default=1.0,
                        help='Value scaling factor applied before export')
    parser.add_argument('--season', type=str, default='all',
                        choices=['all', 'spring', 'summer', 'fall', 'winter'],
                        help='Optional season filter')
    parser.add_argument('--rescale-method', type=str, default='default',
                        choices=['default', 'resnet'],
                        help='Preprocessing method used by the raw dataset loader')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output files')
    return parser.parse_args()


def ratio_to_suffix(sample_ratio: float) -> str:
    ratio_str = f"{sample_ratio:.4f}".rstrip('0').rstrip('.')
    return ratio_str.replace('.', 'p')


def build_output_path(output_dir: Path, split: str, sample_ratio: float) -> Path:
    if sample_ratio >= 1.0:
        file_name = f'{split}.npz'
    else:
        file_name = f'{split}_ratio_{ratio_to_suffix(sample_ratio)}.npz'
    return output_dir / file_name


def select_indices(num_samples: int, sample_ratio: float, seed: int):
    if not 0 < sample_ratio <= 1:
        raise ValueError(f'sample_ratio must be in (0, 1], got {sample_ratio}')

    if num_samples == 0:
        return np.empty((0,), dtype=np.int64)

    if sample_ratio >= 1.0:
        return np.arange(num_samples, dtype=np.int64)

    subset_size = max(1, int(round(num_samples * sample_ratio)))
    rng = np.random.default_rng(seed)
    indices = rng.choice(num_samples, size=subset_size, replace=False)
    return np.sort(indices.astype(np.int64))


def export_split(dataset, split: str, indices, output_path: Path, data_range: float):
    if len(indices) == 0:
        raise RuntimeError(f'No sample selected for split: {split}')

    first_sample = dataset[int(indices[0])]
    sar_shape = first_sample['input']['S1'].shape
    cloudy_shape = first_sample['input']['S2'].shape
    target_shape = first_sample['target']['S2'].shape

    sar_array = np.empty((len(indices), *sar_shape), dtype=np.float16)
    cloudy_array = np.empty((len(indices), *cloudy_shape), dtype=np.float16)
    target_array = np.empty((len(indices), *target_shape), dtype=np.float16)
    path_list = []

    for export_idx, dataset_idx in enumerate(tqdm(indices, desc=f'Converting {split}', unit='sample')):
        sample = dataset[int(dataset_idx)]
        sar_array[export_idx] = (sample['input']['S1'] * data_range).astype(np.float16)
        cloudy_array[export_idx] = (sample['input']['S2'] * data_range).astype(np.float16)
        target_array[export_idx] = (sample['target']['S2'] * data_range).astype(np.float16)
        path_list.append(dataset.paths[int(dataset_idx)]['S2'])

    np.savez(
        output_path,
        s1=sar_array,
        s2=cloudy_array,
        label=target_array,
        paths=np.asarray(path_list),
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_idx, split in enumerate(args.splits):
        print(f'Preparing split: {split}')
        dataset = SEN12MSCR(
            args.root,
            split=split,
            season=args.season,
            rescale_method=args.rescale_method,
        )
        selected_indices = select_indices(
            len(dataset),
            args.sample_ratio,
            args.seed + split_idx,
        )
        output_path = build_output_path(output_dir, split, args.sample_ratio)

        if output_path.exists() and not args.force:
            raise FileExistsError(
                f'Output file already exists: {output_path}. Use --force to overwrite it.'
            )

        print(f'Selected {len(selected_indices)} / {len(dataset)} samples.')
        print(f'Writing uncompressed float16 NPZ to: {output_path}')
        export_split(dataset, split, selected_indices, output_path, args.data_range)
        print(f'Finished split: {split}')


if __name__ == '__main__':
    main()