from typing import Sequence

import torch
from torch.utils.data import Subset


DEFAULT_TRAIN_RATIO = 0.8


def normalize_dataset_splits(split) -> list[str]:
    if isinstance(split, str):
        splits = [split]
    elif isinstance(split, Sequence):
        splits = [item for item in split if item]
    else:
        raise TypeError(f'Unsupported split configuration type: {type(split)!r}')

    if not 1 <= len(splits) <= 3:
        raise ValueError(f'dataset.split must contain 1 to 3 entries, got {splits!r}')
    return splits


def get_train_ratio(config) -> float:
    train_ratio = float(getattr(config.dataset, 'train_ratio', DEFAULT_TRAIN_RATIO))
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f'dataset.train_ratio must be in (0, 1), got {train_ratio}')
    return train_ratio


def build_single_split_indices(dataset_len: int, train_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if dataset_len < 2:
        raise ValueError('At least two samples are required when splitting a single dataset branch.')

    train_size = int(round(dataset_len * train_ratio))
    train_size = min(max(train_size, 1), dataset_len - 1)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_len, generator=generator).tolist()
    train_indices = permutation[:train_size]
    eval_indices = permutation[train_size:]
    return train_indices, eval_indices


def resolve_eval_split_name(config, split_override: str | None = None) -> str:
    if split_override is not None:
        return split_override

    splits = normalize_dataset_splits(config.dataset.split)
    if len(splits) == 1:
        return f'{splits[0]}_holdout'
    return splits[-1]


def build_train_valid_datasets(config, dataset_class):
    splits = normalize_dataset_splits(config.dataset.split)
    root = config.dataset.root
    data_range = config.dataset.data_range
    crop_size = config.dataset.crop_size

    if len(splits) >= 2:
        train_dataset = dataset_class(
            root,
            split=splits[0],
            data_range=data_range,
            crop_size=crop_size,
        )
        valid_dataset = dataset_class(
            root,
            split=splits[1],
            data_range=data_range,
        )
        split_info = {
            'train': splits[0],
            'valid': splits[1],
        }
        return train_dataset, valid_dataset, split_info

    base_split = splits[0]
    train_ratio = get_train_ratio(config)
    train_dataset_full = dataset_class(
        root,
        split=base_split,
        data_range=data_range,
        crop_size=crop_size,
    )
    valid_dataset_full = dataset_class(
        root,
        split=base_split,
        data_range=data_range,
    )
    train_indices, eval_indices = build_single_split_indices(
        len(train_dataset_full),
        train_ratio=train_ratio,
        seed=config.seed,
    )
    train_dataset = Subset(train_dataset_full, train_indices)
    valid_dataset = Subset(valid_dataset_full, eval_indices)
    split_info = {
        'train': f'{base_split}_train_split',
        'valid': f'{base_split}_holdout',
    }
    return train_dataset, valid_dataset, split_info


def build_eval_dataset(config, dataset_class, split_override: str | None = None):
    root = config.dataset.root
    data_range = config.dataset.data_range

    if split_override is not None:
        dataset = dataset_class(root, split=split_override, data_range=data_range)
        return split_override, dataset

    splits = normalize_dataset_splits(config.dataset.split)
    if len(splits) >= 2:
        eval_split = splits[-1]
        dataset = dataset_class(root, split=eval_split, data_range=data_range)
        return eval_split, dataset

    base_split = splits[0]
    dataset = dataset_class(root, split=base_split, data_range=data_range)
    _, eval_indices = build_single_split_indices(
        len(dataset),
        train_ratio=get_train_ratio(config),
        seed=config.seed,
    )
    return f'{base_split}_holdout', Subset(dataset, eval_indices)