"""Read LIB ENVI cubes with one-scene cache; rotate as supplied create_patches.py."""
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

EXPECTED = {"train": 393, "validation": 45, "test": 75}


def envi_metadata(path):
    text = Path(path).read_text(encoding="utf-8")
    def field(name):
        match = re.search(r"^\s*" + re.escape(name) + r"\s*=\s*([^\r\n]+)", text, re.I | re.M)
        if not match:
            raise ValueError(f"Missing ENVI field {name}: {path}")
        return match[1].strip()
    meta = {key: int(field(key)) for key in
            ("samples", "lines", "bands", "data type", "byte order", "header offset")}
    if meta["data type"] != 4 or field("interleave").lower() != "bil":
        raise ValueError(f"Expected LIB float32 BIL cube: {path}")
    if meta["byte order"] not in (0, 1) or meta["header offset"] < 0:
        raise ValueError(f"Invalid ENVI storage: {path}")
    return meta


def cube_view(hdr):
    meta = envi_metadata(hdr)
    dat = hdr.with_suffix(".dat")
    expected_bytes = meta["header offset"] + 4 * meta["lines"] * meta["samples"] * meta["bands"]
    if not dat.is_file() or dat.stat().st_size != expected_bytes:
        raise ValueError(f"Incomplete or missing cube (copy still running?): {dat}")
    raw = np.memmap(dat, mode="r", dtype="<f4" if meta["byte order"] == 0 else ">f4",
                    offset=meta["header offset"], shape=(meta["lines"], meta["bands"], meta["samples"]))
    return np.rot90(raw.transpose(0, 2, 1), 3)


def downsample(hsi):
    """Synthetic x4 degradation: antialiased bicubic, fixed reflectance range."""
    return F.interpolate(hsi.unsqueeze(0), scale_factor=0.25, mode="bicubic",
                         align_corners=False, antialias=True).squeeze(0).clamp(0, 1)


class LIBHSI(Dataset):
    def __init__(self, root, split, patch_size=128, patches_per_scene=4,
                 limit=None, allow_incomplete=False, compute_lr=True, crop_seed=42, read_mode="whole"):
        if patch_size < 8 or patch_size % 4 or 512 % patch_size:
            raise ValueError("patch_size must divide 512 and be a multiple of 4 (>=8)")
        if split not in EXPECTED or patches_per_scene < 1:
            raise ValueError("Invalid split or patches_per_scene")
        self.root, self.split = Path(root), split
        self.patch_size, self.patches_per_scene = patch_size, patches_per_scene
        self.compute_lr = compute_lr
        self.crop_seed = crop_seed
        if read_mode not in ("whole", "stripes"):
            raise ValueError("read_mode must be whole or stripes")
        self.read_mode = read_mode
        base = self.root / split
        self.files = sorted(p for p in (base / "rgb").glob("*.png") if not p.name.startswith("._"))
        if not allow_incomplete and len(self.files) != EXPECTED[split]:
            raise ValueError(f"{split}: expected {EXPECTED[split]} RGB images, found {len(self.files)}. "
                             "Finish copying LIB-HSI before full training.")
        if limit is not None:
            if limit < 1:
                raise ValueError("scene limit must be positive")
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"No RGB images: {base / 'rgb'}")
        self.headers = [base / "reflectance_cubes" / (p.stem + ".hdr") for p in self.files]
        for rgb, hdr in zip(self.files, self.headers):
            cube = cube_view(hdr)
            if cube.shape != (512, 512, 204):
                raise ValueError(f"Unexpected LIB shape {cube.shape}: {hdr}")
            with Image.open(rgb) as im:
                if im.size != (512, 512):
                    raise ValueError(f"Unexpected RGB shape: {rgb}")
        self.tiles_per_side = 512 // patch_size
        self.per_scene = patches_per_scene if split == "train" else self.tiles_per_side ** 2
        self._cache_scene = None
        self._cache_cube = None
        self._stripe_scene = None
        self._stripe_cache = {}
        self.stripe_rows = 64
        self._read_bytes_total = 0

    def _read_patch(self, scene, y, x, p):
        """Read/cache only required contiguous BIL row stripes; output rotated CHW."""
        if self._stripe_scene != scene:
            self._stripe_scene = scene
            self._stripe_cache = {}
            self._stripe_meta = envi_metadata(self.headers[scene])
        meta = self._stripe_meta
        first, last = meta["lines"] - x - p, meta["lines"] - x
        raw_patch = np.empty((p, meta["bands"], p), dtype=np.float32)
        dat = self.headers[scene].with_suffix(".dat")
        for row in range((first // self.stripe_rows) * self.stripe_rows, last, self.stripe_rows):
            rows = min(self.stripe_rows, meta["lines"] - row)
            if row not in self._stripe_cache:
                count = rows * meta["bands"] * meta["samples"]
                offset = meta["header offset"] + row * meta["bands"] * meta["samples"] * 4
                values = np.fromfile(dat, dtype="<f4" if meta["byte order"] == 0 else ">f4",
                                     count=count, offset=offset)
                if values.size != count:
                    raise ValueError(f"Incomplete stripe: {dat}")
                self._read_bytes_total = getattr(self, "_read_bytes_total", 0) + values.nbytes
                self._stripe_cache[row] = values.reshape(rows, meta["bands"], meta["samples"])
            begin, end = max(first, row), min(last, row + rows)
            raw_patch[begin-first:end-first] = self._stripe_cache[row][begin-row:end-row, :, y:y+p]
        return np.ascontiguousarray(raw_patch.transpose(1, 2, 0)[:, :, ::-1])

    def _read_scene(self, scene):
        if self._cache_scene != scene:
            # BIL + rotated crops cause many scattered HDD reads. Read one scene sequentially
            # and keep only that scene (~204 MiB). Never preload the entire dataset.
            self._cache_cube = None
            self._cache_scene = None
            hdr = self.headers[scene]
            meta = envi_metadata(hdr)
            count = meta["lines"] * meta["bands"] * meta["samples"]
            values = np.fromfile(hdr.with_suffix(".dat"),
                                 dtype="<f4" if meta["byte order"] == 0 else ">f4",
                                 count=count, offset=meta["header offset"])
            if values.size != count:
                raise ValueError(f"Incomplete cube: {hdr}")
            self._read_bytes_total = getattr(self, "_read_bytes_total", 0) + values.nbytes
            raw = values.reshape(meta["lines"], meta["bands"], meta["samples"])
            rotated = np.rot90(raw.transpose(0, 2, 1), 3)
            # Contiguous CHW storage makes subsequent rotated crops sequential in RAM.
            self._cache_cube = np.ascontiguousarray(rotated.transpose(2, 0, 1)).transpose(1, 2, 0)
            self._cache_scene = scene
        return self._cache_cube

    def __len__(self):
        return len(self.files) * self.per_scene

    def __getitem__(self, index):
        generator = None
        if isinstance(index, tuple):
            index, epoch = index
            # Epoch/index seeds survive Windows persistent workers and checkpoint resumes.
            generator = torch.Generator().manual_seed(self.crop_seed + epoch * len(self) + index)
        scene, tile = divmod(index, self.per_scene)
        p = self.patch_size
        if self.split == "train":
            # x4-aligned random crops; online joint flips, no precomputed cubes.
            y, x = (int(torch.randint(0, (512 - p) // 4 + 1, (), generator=generator)) * 4 for _ in range(2))
        else:
            y, x = (tile // self.tiles_per_side) * p, (tile % self.tiles_per_side) * p
        before_bytes = self._read_bytes_total
        if self.read_mode == "stripes":
            gt_array = self._read_patch(scene, y, x, p)
        else:
            cube = self._read_scene(scene)
            gt_array = np.array(cube[y:y+p, x:x+p].transpose(2, 0, 1), dtype=np.float32, order="C", copy=True)
        if not np.isfinite(gt_array).all():
            raise ValueError(f"Nonfinite reflectance: {self.headers[scene]}")
        np.clip(gt_array, 0, 1, out=gt_array)
        with Image.open(self.files[scene]) as image:
            rgb_array = np.array(image.convert("RGB").crop((x, y, x+p, y+p)), copy=True)
        rgb_array = rgb_array.transpose(2, 0, 1).astype(np.float32) / 255
        if self.split == "train":
            for dimension in (1, 2):
                if torch.rand((), generator=generator) < 0.5:
                    gt_array = np.flip(gt_array, axis=dimension)
                    rgb_array = np.flip(rgb_array, axis=dimension)
        gt = torch.from_numpy(np.ascontiguousarray(gt_array))
        rgb = torch.from_numpy(np.ascontiguousarray(rgb_array))
        result = {"rgb": rgb, "gt": gt, "scene": self.files[scene].stem,
                  "source_cube_bytes": self._read_bytes_total - before_bytes}
        if self.compute_lr:
            result["lr_hsi"] = downsample(gt)
        return result


class ScenePatchSampler(Sampler):
    """Shuffle scenes per epoch, keep their random patches together for bounded HDD I/O."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        generator = None
        if self.epoch is not None:
            generator = torch.Generator().manual_seed(getattr(self.dataset, "crop_seed", 42) + self.epoch)
        for scene in torch.randperm(len(self.dataset.files), generator=generator).tolist():
            for patch in range(self.dataset.per_scene):
                index = scene * self.dataset.per_scene + patch
                yield index if self.epoch is None else (index, self.epoch)
