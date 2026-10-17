"""Explicit synthetic degradation, provenance checks and resident tensors."""
import hashlib
import json
import math
import time
import warnings
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from .model import resize

DEGRADATION = {
    "id": "qb_proxy_gaussian_decimate_v1", "physical_sensor_model": False,
    "blur": "separable Gaussian sigma=r/2; radius=ceil(3*sigma)",
    "boundary": "reflect on supplied sample BEFORE common top-left crop",
    "sampling": "stride r; phase=(floor(r/2),floor(r/2))",
    "crop": "top-left largest H,W divisible by 12; same target at all scales",
    "lms": "bilinear size=target, align_corners=False, no upsample antialias",
    "seed": "no random degradation", "normalization": "QB DN / 2047",
    "limitations": "H5 patches may lack full-scene context; not audited MTF/Wald multi-ratio data",
}

def sha_file(path, event=None, split=None, progress_seconds=5):
    h = hashlib.sha256()
    total, done, started, last = Path(path).stat().st_size, 0, time.monotonic(), 0
    if event:
        event("validation_hash", split=split, done=0, total=total, unit="bytes", elapsed_seconds=0)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
            done += len(block)
            now = time.monotonic()
            if event and (now-last >= progress_seconds or done == total):
                event("validation_hash", split=split, done=done, total=total, unit="bytes",
                      elapsed_seconds=now-started)
                last = now
    return h.hexdigest()


def sample_rows(f, keys, count, block_rows, event, split, progress_seconds):
    """Read bounded contiguous slabs instead of four HDF5 reads per sample."""
    started, last = time.monotonic(), 0
    event("validation_samples", split=split, done=0, total=count, unit="samples", elapsed_seconds=0)
    for start in range(0, count, block_rows):
        stop = min(start + block_rows, count)
        blocks = {k: np.asarray(f[k][start:stop], dtype=np.float32) for k in keys}
        for i in range(start, stop):
            yield i, {k: blocks[k][i-start] for k in keys}
        now = time.monotonic()
        if now-last >= progress_seconds or stop == count:
            event("validation_samples", split=split, done=stop, total=count, unit="samples",
                  elapsed_seconds=now-started)
            last = now

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def synthetic_pair(gt, pan, scale):
    if not isinstance(scale, int) or isinstance(scale, bool) or scale not in (2, 3, 4) or gt.ndim != 4 or pan.ndim != 4:
        raise ValueError("Expected NCHW and scale 2/3/4")
    if pan.shape != (len(gt), 1, *gt.shape[-2:]):
        raise ValueError("PAN must align with GT")
    h, w = (s - s % 12 for s in gt.shape[-2:])
    if min(h, w) < 12:
        raise ValueError("Synthetic target must be at least 12x12")
    sigma, radius = scale / 2, math.ceil(3 * scale / 2)
    if min(gt.shape[-2:]) <= radius:
        raise ValueError("Sample is too small for reflect blur")
    with torch.autocast(device_type=gt.device.type, enabled=False):
        x = gt.float()
        axis = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
        k = torch.exp(-axis.square() / (2 * sigma * sigma))
        k = k / k.sum()
        c = gt.shape[1]
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"),
                     k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
        x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"),
                     k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
        phase = scale // 2
        ms = x[..., phase:h:scale, phase:w:scale].contiguous()
        target, guide = gt[..., :h, :w], pan[..., :h, :w]
        return {"pan": guide, "gt": target, "ms": ms, "lms": resize(ms, (h, w)),
                "scale": scale, "degradation_id": DEGRADATION["id"]}

def inspect_sources(paths, provenance_path=None, allow_unverified=False, event=None,
                    block_rows=None, progress_seconds=5):
    event = event or (lambda *args, **kwargs: None)
    if block_rows is not None and (type(block_rows) is not int or block_rows < 1):
        raise ValueError("block_rows must be a positive integer")
    paths = {k: Path(v).resolve() for k, v in paths.items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError("Different splits must use different H5 paths")
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8")) if provenance_path else {}
    result, fingerprints, scene_groups = {}, {}, {}
    for split, path in paths.items():
        split_started = time.monotonic()
        event("validation_file", split=split, path=str(path))
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as f:
            keys = ("pan", "ms", "lms") + (() if split == "FR" else ("gt",))
            if any(k not in f for k in keys):
                raise ValueError(f"Missing keys in {split}: {keys}")
            shape = tuple(f["lms"].shape)
            if len(shape) != 4:
                raise ValueError("Expected NCHW QB H5")
            n, c, h, w = shape
            if n < 1 or c != 4 or h % 4 or w % 4 or min(h, w) < 12:
                raise ValueError(f"Invalid QB shape: {shape}")
            expected = {"lms": shape, "pan": (n, 1, h, w), "ms": (n, c, h//4, w//4), "gt": shape}
            if any(tuple(f[k].shape) != expected[k] for k in keys):
                raise ValueError(f"Misaligned shapes in {split}")
            ranges = {k: [float("inf"), float("-inf")] for k in keys}
            hashes = []
            processed_outside = {k: {"samples": 0, "values": 0, "first_sample_indices": []}
                                 for k in keys if k in ("ms", "lms")}
            row_bytes = sum(int(np.prod(f[k].shape[1:])) * 4 for k in keys)
            rows_per_read = block_rows or max(1, min(128, (32 * 1024**2) // max(1, row_bytes)))
            for i, row in sample_rows(f, keys, n, rows_per_read, event, split, progress_seconds):
                digest = hashlib.sha256()
                for k in keys:
                    arr = row[k]
                    finite = np.isfinite(arr)
                    low = float(arr[finite].min()) if finite.any() else None
                    high = float(arr[finite].max()) if finite.any() else None
                    if not finite.all():
                        raise ValueError(f"Nonfinite QB data: {split}/{k}/{i}; "
                                         f"NaN={int(np.isnan(arr).sum())}, "
                                         f"Inf={int(np.isinf(arr).sum())}, finite_min={low}, finite_max={high}, "
                                         f"source_dtype={f[k].dtype}. Original data must be inspected; not replaced/clipped.")
                    outside = int(((arr < 0) | (arr > 2047)).sum())
                    if outside and k not in ("ms", "lms"):
                        raise ValueError(f"Out-of-range QB source DN: {split}/{k}/{i}; "
                                         f"min={low}, max={high}, outside_count={outside}, source_dtype={f[k].dtype}")
                    if outside:
                        entry = processed_outside[k]
                        entry["samples"] += 1
                        entry["values"] += outside
                        if len(entry["first_sample_indices"]) < 10:
                            entry["first_sample_indices"].append(i)
                    ranges[k][0] = min(ranges[k][0], low)
                    ranges[k][1] = max(ranges[k][1], high)
                    if k in ("pan", "gt"):
                        digest.update(arr.tobytes())
                hashes.append(digest.hexdigest())
            entries = provenance.get(split)
            if entries is None and "scene_id" in f:
                entries = [{"scene_id": str(v.decode() if isinstance(v, bytes) else v),
                            "tile": "source H5 sample"} for v in f["scene_id"][:]]
            verified = entries is not None
            if verified and (len(entries) != n or any(not e.get("scene_id") for e in entries)):
                raise ValueError(f"Provenance must provide one nonempty scene_id per {split} sample")
            if not verified:
                entries = [{"scene_id": f"UNVERIFIED:{split}:{i}", "tile": "source H5 sample"}
                           for i in range(n)]
            scene_groups[split] = {e["scene_id"] for e in entries}
            fingerprints[split] = set(hashes)
            result[split] = {"path": str(path), "sha256": sha_file(path, event, split, progress_seconds),
                             "shapes": {k: list(f[k].shape) for k in keys}, "ranges_dn": ranges,
                             "samples": entries, "content_fingerprints": hashes,
                             "scene_provenance_present": verified,
                             "processed_outside_nominal_dn": processed_outside,
                             "range_policy": "pan/gt strict [0,2047]; processed ms/lms finite, "
                                             "outside values recorded and warned; no clipping, DN/2047 unchanged",
                             "processed_range_cause_verified": False}
            for k, counts in processed_outside.items():
                if counts["values"]:
                    warnings.warn(f"{split}/{k}: finite values outside nominal QB [0,2047]; "
                                  f"min={ranges[k][0]}, max={ranges[k][1]}, "
                                  f"values={counts['values']}, samples={counts['samples']}, "
                                  f"first_indices={counts['first_sample_indices']}. "
                                  "Preserved DN/2047 without clipping; preprocessing cause not verified.",
                                  RuntimeWarning)
        event("validation_file_complete", split=split, elapsed_seconds=time.monotonic()-split_started)
    event("validation_overlap_check")
    independent = [s for s in ("train", "val", "RR") if s in result]
    for i, a in enumerate(independent):
        for b in independent[i+1:]:
            if fingerprints[a] & fingerprints[b]:
                raise ValueError(f"Exact duplicated samples across {a}/{b}")
            if scene_groups[a] & scene_groups[b]:
                raise ValueError(f"Original-scene leakage across {a}/{b}")
    # RR/FR can be two representations of the SAME test scenes; that is allowed.
    verified = all(result[s]["scene_provenance_present"] for s in independent)
    if not verified and not allow_unverified:
        raise ValueError("Original scene IDs missing: supply SCENE_MANIFEST or explicitly set "
                         "ALLOW_UNVERIFIED_SPLITS=True for exploratory patch-level runs")
    if len({v["sha256"] for v in result.values()}) != len(result):
        raise ValueError("Duplicate split files")
    event("validation_complete")
    return {"files": result, "scene_split_verified": verified,
            "overlap_check": "scene-disjoint groups when supplied; exact duplicates always checked",
            "degradation": DEGRADATION,
            "provenance_sha256": sha_file(provenance_path) if provenance_path else None}

class ResidentSplit:
    """GPU tensors on CUDA runs. CPU is permitted ONLY by explicit smoke config."""
    def __init__(self, metadata, device):
        self.metadata = metadata
        self.entries = metadata["samples"]
        with h5py.File(metadata["path"], "r") as f:
            self.tensors = {k: torch.as_tensor(np.asarray(f[k], dtype=np.float32) / 2047,
                                             device=device) for k in metadata["shapes"]}
        self.length = len(self.tensors["pan"])
        self.device = torch.device(device)
        self.scene_id_lookup = sorted({entry["scene_id"] for entry in self.entries})
        codes = {name: index for index, name in enumerate(self.scene_id_lookup)}
        self.scene_codes = torch.tensor([codes[e["scene_id"]] for e in self.entries], device=self.device)

    def batch(self, indices, scale, legacy=False):
        t = {k: v.index_select(0, indices) for k, v in self.tensors.items()}
        identifiers = {"scene_id": self.scene_codes.index_select(0, indices),
                       "scene_id_lookup": self.scene_id_lookup, "sample_index": indices}
        if legacy:
            if scale != 4:
                raise ValueError("Reference supports x4 only")
            return dict(t, **identifiers, scale=4, degradation_id="provided_native_x4")
        return dict(synthetic_pair(t["gt"], t["pan"], scale), **identifiers)
