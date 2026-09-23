"""Read PanCollection H5 files without changing their original arrays."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SENSOR_MAX = {"QB": 2047.0, "WV3": 2047.0, "GF2": 1023.0, "WV2": 2047.0}
REQUIRED_KEYS = ("pan", "lms", "ms", "gt")


def inspect_h5(path):
    """Return array metadata; do not load the complete dataset into RAM."""
    with h5py.File(path, "r") as file:
        return {
            key: {"shape": tuple(value.shape), "dtype": str(value.dtype)}
            for key, value in file.items()
            if isinstance(value, h5py.Dataset)
        }


class PanCollectionH5(Dataset):
    """PanCollection reduced-resolution pairs in NCHW layout."""

    def __init__(self, path, sensor="QB", limit=None):
        self.path = Path(path)
        self.sensor = sensor.upper()
        if self.sensor not in SENSOR_MAX:
            raise ValueError(f"Unknown sensor: {sensor}")
        metadata = inspect_h5(self.path)
        missing = set(REQUIRED_KEYS) - metadata.keys()
        if missing:
            raise ValueError(f"Missing H5 datasets: {sorted(missing)}")

        shapes = {key: metadata[key]["shape"] for key in REQUIRED_KEYS}
        if any(len(shape) != 4 for shape in shapes.values()):
            raise ValueError(f"Expected NCHW arrays, got {shapes}")
        n, channels, h, w = shapes["gt"]
        if not n or min(h, w) <= 0 or h % 4 or w % 4:
            raise ValueError(f"Invalid GT shape: {shapes['gt']}")
        if shapes["lms"] != shapes["gt"]:
            raise ValueError(f"LMS and GT shapes differ: {shapes}")
        if shapes["pan"] != (n, 1, h, w):
            raise ValueError(f"PAN shape differs from GT: {shapes}")
        if shapes["ms"] != (n, channels, h // 4, w // 4):
            raise ValueError(f"MS should be 4x lower-resolution: {shapes}")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        self.length = min(n, limit) if limit is not None else n
        self.channels = channels
        self._file = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise IndexError(index)
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        scale = SENSOR_MAX[self.sensor]
        sample = {}
        for key in REQUIRED_KEYS:
            array = np.asarray(self._file[key][index], dtype=np.float32)
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite {key} at index {index}")
            sample[key] = torch.from_numpy(array / scale)
        return sample

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()
