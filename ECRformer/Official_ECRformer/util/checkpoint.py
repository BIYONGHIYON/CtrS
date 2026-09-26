"""Checkpoint utilities for automatic resume."""

import os
import glob
from pathlib import Path

import torch


def find_latest_checkpoint(save_dir, log_name):
    """Find the latest checkpoint file for automatic resume.

    Args:
        save_dir: Base directory for experiments.
        log_name: Experiment name.

    Returns:
        tuple: (checkpoint_path, version_number), or (None, None) if not found.
    """
    exp_dir = os.path.join(save_dir, log_name)

    if not os.path.exists(exp_dir):
        print(f"Experiment directory does not exist: {exp_dir}")
        return None, None

    version_dirs = glob.glob(os.path.join(exp_dir, "version_*"))
    if not version_dirs:
        print(f"No version directory found under: {exp_dir}")
        return None, None

    def extract_version_number(path):
        try:
            return int(os.path.basename(path).split('_')[-1])
        except (ValueError, IndexError):
            return -1

    version_dirs.sort(key=extract_version_number)
    latest_version_dir = version_dirs[-1]
    version_num = extract_version_number(latest_version_dir)

    checkpoint_dir = os.path.join(latest_version_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None, None

    # Prefer last.ckpt
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        print(f"Found latest checkpoint: {last_ckpt} (version: {version_num})")
        return last_ckpt, version_num

    # Fall back to latest epoch checkpoint
    ckpt_files = glob.glob(os.path.join(checkpoint_dir, "epoch=*.ckpt"))
    if not ckpt_files:
        print(f"No checkpoint file found in: {checkpoint_dir}")
        return None, None

    def extract_epoch(filename):
        basename = os.path.basename(filename)
        try:
            return int(basename.split('epoch=')[1].split('-')[0])
        except (ValueError, IndexError):
            return -1

    ckpt_files.sort(key=extract_epoch)
    latest_ckpt = ckpt_files[-1]
    print(f"Found latest checkpoint: {latest_ckpt} (version: {version_num})")

    return latest_ckpt, version_num


def load_checkpoint_file(ckpt_path, map_location='cpu'):
    """Load a checkpoint file with backward-compatible torch.load behavior."""
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=map_location)


def extract_state_dict(checkpoint):
    """Extract a model state_dict from a Lightning checkpoint or a raw weights file."""
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
        return checkpoint['state_dict']
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f'Unsupported checkpoint type: {type(checkpoint)!r}')


def describe_checkpoint(checkpoint, max_keys=10):
    """Summarize the structure of a checkpoint for inspection or conversion."""
    if not isinstance(checkpoint, dict):
        return {
            'format': type(checkpoint).__name__,
            'top_level_keys': [],
            'num_tensors': 0,
            'sample_state_keys': [],
        }

    top_level_keys = list(checkpoint.keys())
    state_dict = extract_state_dict(checkpoint)

    if 'state_dict' in checkpoint:
        ckpt_format = 'lightning-full' if 'pytorch-lightning_version' in checkpoint else 'wrapped-state-dict'
    else:
        ckpt_format = 'weights-only-state-dict'

    return {
        'format': ckpt_format,
        'top_level_keys': top_level_keys,
        'num_tensors': len(state_dict),
        'sample_state_keys': list(state_dict.keys())[:max_keys],
    }


def default_weights_output_path(input_path):
    """Return the default path for a converted weights-only checkpoint."""
    input_path = Path(input_path)
    suffix = ''.join(input_path.suffixes) or '.ckpt'
    stem = input_path.name[:-len(suffix)] if suffix else input_path.name
    return input_path.with_name(f'{stem}.weights{suffix}')
