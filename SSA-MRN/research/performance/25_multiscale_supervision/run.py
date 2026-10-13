# Run this entire file as ONE Kaggle notebook cell (T4 x2 + Internet enabled).
# Harness audit SHA256: e34bc16d42eec0e5341ba9c7040d8b091cde3a6af4c8ae383ccdacdb533a7bc4
# The optional path overrides are the only settings normally needing edits.
SENSOR = "QB"
K = 4
EPOCHS = 200
HOLD_LR_EPOCHS = 100  # 1..100 fixed LR; 101..200 cosine decay.
MIN_LEARNING_RATE = 1e-6
STATUS_INTERVAL_SECONDS = 15
CHECKPOINT_INTERVAL_SECONDS = 60
FUSED_ADAM = True  # Same optimizer implementation in both arms; GPU AMP overflow handling.
AUX_WEIGHTS = {"quarter": 0.005, "half": 0.005, "full": 0.001}
RESUME_INPUT = ""  # Saved Output directory from THIS cell; auto-find a unique matching output.
# Default: fresh B0/B0_DS pair. Existing legacy 100-epoch weights are not imported silently.
SEEDS = [42]
VARIANTS = ["B0", "B0_DS"]
EFFECTIVE_BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LEGACY_MICRO_BATCH_SIZE = 32  # One forward/backward per effective batch; equal in B0/B0_DS.
# If either arm fails the preflight with OOM, set this to 16 in BOTH arms.
# Different micro batches change floating-point accumulation, although batch/loss remain equal.
TRAINING_MODES = ["B0", "B0_DS"]

SWAP_GPUS = False  # Swap the two physical GPUs for a repeat in a new output directory.
CONTROLLED_COMPARISON = True  # Same micro batch in both arms; False benchmarks it per model.
AMP = True
USE_ALL_GPUS = True
GPU_CACHE = True
CHANNELS_LAST = True
NUM_WORKERS = 0  # No CPU DataLoader in GPU-resident training.
SAVE_PREDICTIONS = True
DATA_ROOT = "/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset"

TRAIN_H5 = f"{DATA_ROOT}/QuickBird__Training_Dataset__train_qb.h5"
VAL_H5   = f"{DATA_ROOT}/QuickBird__Training_Dataset__valid_qb.h5"
RR_H5    = f"{DATA_ROOT}/QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5"
FR_H5    = f"{DATA_ROOT}/QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5"
OUTPUT_DIR = "/kaggle/working/ssa_kaggle_pyramid_supervision_k4_v1"
REPO_COMMIT = "be02b7cbf03fe00681194ae9fc9419661f24cd56"
UPSTREAM_COMMIT = "a4ca40e407b12bf4c30f804384405ce321d11c51"
PLAN_COMMIT = None  # Architecture is defined directly in this cell.

print("[PERSISTENCE] Enable Files persistence for interactive sessions. Save a Version with Output and verify/download best/latest before leaving. Unsaved files deleted by Kaggle cannot be recovered by this cell.",flush=True)
print("[START] B0/B0_DS k4: original pyramid, GT-only auxiliary supervision", flush=True)
import hashlib
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

for module, package in [("h5py", "h5py"), ("scipy", "scipy"),
                        ("skimage", "scikit-image"), ("PIL", "pillow"),
                        ("matplotlib", "matplotlib"), ("psutil", "psutil")]:
    if importlib.util.find_spec(module) is None:
        print(f"[SETUP] Installing {package}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True, timeout=180)
import h5py
import numpy as np
import torch
from IPython.display import clear_output, display, FileLink

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required. Enable a Kaggle GPU accelerator.")
print(f"[CUDA] {torch.cuda.device_count()} GPUs detected", flush=True)
suite_started = time.monotonic()
if not (0 < HOLD_LR_EPOCHS < EPOCHS and 0 < MIN_LEARNING_RATE <= LEARNING_RATE):
    raise ValueError("Require 0 < HOLD_LR_EPOCHS < EPOCHS, 0 < minimum LR <= initial LR")
if not (STATUS_INTERVAL_SECONDS >= 5 and CHECKPOINT_INTERVAL_SECONDS >= 60):
    raise ValueError("Progress/checkpoint intervals are too small")
CELL_ID = "ssa_pyramid_supervision_k4_v1"
suite_settings = {"cell_id": CELL_ID, "repo_commit": REPO_COMMIT,
    "upstream_commit": UPSTREAM_COMMIT, "sensor": SENSOR, "k": K, "epochs": EPOCHS,
    "seeds": SEEDS, "variants": VARIANTS, "batch": EFFECTIVE_BATCH_SIZE,
    "micro": LEGACY_MICRO_BATCH_SIZE, "lr": LEARNING_RATE, "hold": HOLD_LR_EPOCHS,
    "min_lr": MIN_LEARNING_RATE, "amp": AMP, "channels_last": CHANNELS_LAST, "torch":str(torch.__version__), "cuda":torch.version.cuda,
    "fused_adam": FUSED_ADAM,
    "controlled": CONTROLLED_COMPARISON, "save_predictions": SAVE_PREDICTIONS,
    "status_seconds": STATUS_INTERVAL_SECONDS, "checkpoint_seconds": CHECKPOINT_INTERVAL_SECONDS, "aux_weights": AUX_WEIGHTS}
if SENSOR != "QB":
    raise ValueError("This audited first experiment is QB only.")
if VARIANTS != ["B0", "B0_DS"] or TRAINING_MODES != ["B0", "B0_DS"]:
    raise ValueError("B0 and B0_DS only")
if len(set(SEEDS)) != len(SEEDS) or EPOCHS < 1 or EFFECTIVE_BATCH_SIZE < 1 or LEARNING_RATE <= 0:
    raise ValueError("Invalid run settings")
if not GPU_CACHE or not CONTROLLED_COMPARISON or LEGACY_MICRO_BATCH_SIZE < 1:
    raise ValueError("GPU residency and a shared fixed micro batch are required")
if not USE_ALL_GPUS:
    raise ValueError("USE_ALL_GPUS must remain True for parallel B0/B0_DS")
if torch.cuda.device_count() < 2:
    raise RuntimeError("Select Kaggle T4 x2: B0 and B0_DS require two GPUs for parallel execution")
SCALE = {"QB": 2047., "GF2": 1023., "WV3": 2047.}[SENSOR]
OUT = Path(OUTPUT_DIR)
# Restore only this experiment's exact settings, never legacy or unrelated runs.
if not (OUT / "suite_config.json").exists():
    candidates = ([Path(RESUME_INPUT) / "suite_config.json"] if RESUME_INPUT else
                  sorted(Path("/kaggle/input").rglob("suite_config.json")))
    matches = []
    for candidate in candidates:
        if candidate.is_file():
            try:
                if json.loads(candidate.read_text()) == suite_settings:
                    matches.append(candidate.parent)
            except (OSError, ValueError):
                pass
    if RESUME_INPUT and len(matches) != 1:
        raise ValueError("RESUME_INPUT does not contain this cell's matching suite_config.json")
    if len(matches) > 1:
        raise ValueError(f"Multiple matching previous Outputs; set RESUME_INPUT: {matches}")
    if matches:
        if OUT.exists() and any(OUT.iterdir()):
            raise ValueError("Nonempty output without suite_config.json; choose a new OUTPUT_DIR")
        print(f"[RESTORE] {matches[0]} -> {OUT}", flush=True)
        shutil.copytree(matches[0], OUT, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.tmp", "*.pyc", "__pycache__"))
if OUT.exists() and any(OUT.iterdir()) and not (OUT / "suite_config.json").exists():
    raise ValueError("Nonempty unrelated output; choose a new OUTPUT_DIR")
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "suite_config.json").exists():
    if json.loads((OUT / "suite_config.json").read_text()) != suite_settings:
        raise ValueError("Experiment settings changed; choose a new OUTPUT_DIR")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    path = Path(path)
    size = path.stat().st_size; done = 0; started = last = time.monotonic()
    if size >= 64*1024**2: print(f"[HASH] {path.name}: {size/1024**2:.0f} MB", flush=True)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block); done += len(block)
            if time.monotonic()-last >= 5:
                print(f"[HASH] {path.name}: {100*done/max(size,1):.1f}% | {done/1024**2:.0f}/{size/1024**2:.0f} MB | {time.monotonic()-started:.0f}s", flush=True)
                last = time.monotonic()
    if size >= 64*1024**2: print(f"[HASH DONE] {path.name}: {time.monotonic()-started:.1f}s", flush=True)
    return digest.hexdigest()


def locate(override, filename):
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"Explicit H5 path is missing: {path}")
        if "._" in path.name or not h5py.is_hdf5(path):
            raise ValueError(f"AppleDouble/non-HDF5 input: {path}")
        return path.resolve()
    # Dataset files carry long QuickBird__... prefixes; skip 0 MB AppleDouble files.
    candidates = sorted(p for p in Path(DATA_ROOT).rglob("*.h5")
                        if (p.name == filename or p.name.endswith("__" + filename))
                        and "._" not in p.name and p.stat().st_size > 0 and h5py.is_hdf5(p))
    if len(candidates) != 1:
        raise RuntimeError(f"Set the explicit path for {filename}; valid candidates: {candidates}")
    return candidates[0].resolve()


write_json(OUT / "suite_config.json", suite_settings)


stem = SENSOR.lower()
PATHS = {"train": locate(TRAIN_H5, f"train_{stem}.h5"),
         "val": locate(VAL_H5, f"valid_{stem}.h5"),
         "RR": locate(RR_H5, f"test_{stem}_multiExm1.h5"),
         "FR": locate(FR_H5, f"test_{stem}_OrigScale_multiExm1.h5")}
if len(set(PATHS.values())) != 4:
    raise ValueError("Train, validation, RR and FR must be distinct files.")

# Fetch just the pinned source files, not the repository's large checkpoints.
SOURCE = OUT / "code" / "CtrS"
custom_path = OUT / "code" / "B0_DS_models.py"
custom_hash = None
PASTED_MODEL_SOURCE = r'''
"""GT-only multiscale readouts; the restored SSA-MRN forward stays intact."""
import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet

DEFAULT_AUX_WEIGHTS = {"quarter": 0.005, "half": 0.005, "full": 0.001}


def target_spec():
    return {"source": "normalized GT only; never observed LRMS or Haar LL",
            "ratios": {"quarter": 4, "half": 2, "full": 1},
            "filter": "separable positive unit-sum binomial FIR",
            "binomial_orders": {"quarter": 16, "half": 4},
            "coefficients": {name: [math.comb(order, i) / 2**order for i in range(order+1)]
                             for name, order in (("quarter", 16), ("half", 4))},
            "boundary": "replicate on both axes",
            "coordinates": "j*(N-1)/(M-1), M=N/ratio; bilinear align_corners=True",
            "full": "identity", "precision": "FP32",
            "interpretation": "explicit synthetic GT targets, not a calibrated sensor MTF"}


@lru_cache(maxsize=32)
def _target_geometry(height, width, channels, ratio, device):
    order = ratio**2
    kernel = torch.tensor([math.comb(order, i) / 2**order for i in range(order+1)],
                          device=device, dtype=torch.float32)
    radius = order // 2
    rows = torch.arange(-radius, height+radius, device=device).clamp(0, height-1)
    cols = torch.arange(-radius, width+radius, device=device).clamp(0, width-1)
    horizontal = kernel[None,None,None,:].expand(channels,1,1,-1).contiguous()
    vertical = kernel[None,None,:,None].expand(channels,1,-1,1).contiguous()
    return rows, cols, horizontal, vertical


def gt_target(gt, ratio):
    if gt.ndim != 4 or not gt.is_floating_point() or ratio not in (1, 2, 4):
        raise ValueError("Expected floating NCHW GT and ratio 1, 2 or 4")
    if any(n % ratio for n in gt.shape[-2:]):
        raise ValueError("GT dimensions must be divisible by the target ratio")
    with torch.autocast(gt.device.type, enabled=False):
        x = gt.float()
        if ratio == 1:
            return x
        channels = x.shape[1]
        # index_select reproduces replicate padding and has deterministic CUDA backward.
        rows, cols, horizontal, vertical = _target_geometry(*x.shape[-2:], channels, ratio, x.device)
        x = F.conv2d(x.index_select(-1, cols), horizontal, groups=channels)
        x = F.conv2d(x.index_select(-2, rows), vertical, groups=channels)
        return F.interpolate(x, size=(gt.shape[-2]//ratio, gt.shape[-1]//ratio),
                             mode="bilinear", align_corners=True)


class PyramidSupervision(nn.Module):
    def __init__(self, channels=4, k=4):
        super().__init__()
        self.core = RestoredPansharpeningNet(channels, k)
        self.aux_heads = nn.ModuleDict({"readout_"+name: nn.Conv2d(channels, channels, 1)
                                       for name in ("quarter", "half", "full")})
        with torch.no_grad():
            for head in self.aux_heads.values():
                head.weight.copy_(torch.eye(channels).reshape(channels, channels, 1, 1))
                head.bias.zero_()

    def forward(self, pan, lms, ms, return_aux=False):
        if not return_aux:
            return self.core(pan, lms, ms)
        captures = {}
        handles = []
        try:
            for name, layer in (("quarter", "conv2ctc1"), ("half", "conv2ctc2"), ("full", "conv2ctc3")):
                handles.append(getattr(self.core, layer).register_forward_hook(
                    lambda module, inputs, output, name=name: captures.__setitem__(name, output)))
            final = self.core(pan, lms, ms)
            auxiliary = {name: self.aux_heads["readout_"+name](output) for name, output in captures.items()}
            return final, auxiliary
        finally:
            for handle in handles:
                handle.remove()


def build_model(variant, channels, k):
    if variant not in ("B0", "B0_DS"):
        raise ValueError(variant)
    return PyramidSupervision(channels, k)


def training_loss(model, inputs, gt, weights):
    if set(weights) != {"quarter", "half", "full"} or any(
            not math.isfinite(float(v)) or v < 0 for v in weights.values()):
        raise ValueError("Three finite nonnegative auxiliary weights are required")
    enabled = any(weights.values())
    result = model(*inputs, return_aux=enabled)
    prediction, auxiliary = result if enabled else (result, {})
    mse = F.mse_loss(prediction.float(), gt.float())
    total = mse
    components = {"final_mse": mse}
    for name, ratio in (("quarter", 4), ("half", 2), ("full", 1)):
        term = F.l1_loss(auxiliary[name].float(), gt_target(gt, ratio)) if enabled else mse.new_zeros(())
        components["aux_"+name+"_l1"] = term
        total = total + weights[name] * term
    components["objective"] = total
    return prediction, total, components


def train_config(variant, seed):
    return {"variant": variant, "seed": seed, "core_passes": 1,
            "head_sites": ["conv2ctc1", "conv2ctc2", "conv2ctc3"],
            "loss": "final MSE + weighted mean auxiliary L1",
            "final_output": "original conv2ctc3 output; auxiliary heads training only",
            "target_spec": target_spec()}
'''

custom_hash = hashlib.sha256(PASTED_MODEL_SOURCE.encode()).hexdigest()
custom_path.parent.mkdir(parents=True, exist_ok=True)
if custom_path.exists() and sha(custom_path) != custom_hash:
    raise ValueError("Model code changed; choose a new OUTPUT_DIR")
custom_path.write_text(PASTED_MODEL_SOURCE, encoding="utf-8")
files = ["src/ssamrn/__init__.py", "src/ssamrn/models/__init__.py",
         "src/ssamrn/data/__init__.py", "src/ssamrn/data/pancollection.py",
         "src/ssamrn/models/ssa_mrn.py", "src/ssamrn/models/pan_variants.py",
         "src/ssamrn/models/interp23.py", "src/ssamrn/observation.py",
         "src/ssamrn/metrics.py", "scripts/verify_observation.py"]
source_urls = {f"SSA-MRN/{name}":
               f"https://raw.githubusercontent.com/BIYONGHIYON/CtrS/{REPO_COMMIT}/SSA-MRN/{name}"
               for name in files}
source_urls["SSA-MRN/references/upstream/network.py"] = (
    f"https://raw.githubusercontent.com/zhouchuanxu/SSA-MRN/{UPSTREAM_COMMIT}/network.py")
source_state = OUT / "code" / "source.json"
old_source = json.loads(source_state.read_text()) if source_state.exists() else None
if old_source and (old_source["repo_commit"] != REPO_COMMIT or
                   old_source["upstream_commit"] != UPSTREAM_COMMIT):
    raise ValueError("Source revision changed; choose a new OUTPUT_DIR.")
for relative, url in source_urls.items():
    target = SOURCE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if old_source and sha(target) != old_source["files"][relative]["sha256"]:
            raise ValueError(f"Source was modified: {target}")
    else:
        temporary = target.with_suffix(".download")
        print(f"[DOWNLOAD] {relative}", flush=True)
        # Small pinned Python files only; a stalled connection fails within 30 seconds.
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        print(f"[DOWNLOAD DONE] {relative}", flush=True)
        os.replace(temporary, target)
write_json(source_state, {"repo_commit": REPO_COMMIT, "upstream_commit": UPSTREAM_COMMIT,
                         "files": {name: {"url": url, "sha256": sha(SOURCE / name)}
                                   for name, url in source_urls.items()}})

data_info = {}
for split, path in PATHS.items():
    print(f"Checking {split}: {path}", flush=True)
    with h5py.File(path, "r") as h5:
        keys = ("pan", "lms", "ms", "gt") if split != "FR" else ("pan", "lms", "ms")
        shapes = {key: list(h5[key].shape) for key in keys}
        ranges = {}
        for key in keys:
            low, high = float("inf"), -float("inf")
            for offset in range(0, shapes[key][0], 64):
                chunk = np.asarray(h5[key][offset:offset+64])
                if not np.isfinite(chunk).all(): raise ValueError(f"Nonfinite {split}/{key}")
                low, high = min(low,float(chunk.min())), max(high,float(chunk.max()))
            if high <= 1: raise ValueError(f"Expected DN input, found normalized/empty {split}/{key}")
            if key != "lms" and (low < -1e-4 or high > SCALE+1):
                raise ValueError(f"Unexpected DN range {split}/{key}: {low}, {high}")
            ranges[key] = {"raw_min":low,"raw_max":high,
                           "normalized_min":low/SCALE,"normalized_max":high/SCALE}
        n, c, h, w = shapes["lms"]
        if (n < (5 if split == "RR" else 1) or c != (8 if SENSOR == "WV3" else 4)
                or h % 4 or w % 4 or shapes["pan"] != [n, 1, h, w]
                or shapes["ms"] != [n, c, h // 4, w // 4]
                or (split != "FR" and shapes["gt"] != shapes["lms"])):
            raise ValueError(f"Invalid NCHW arrays in {path}: {shapes}")
        if split in ("RR", "FR") and (h % 32 or w % 32):
            raise ValueError("Repository metrics require 32-pixel aligned full scenes.")
    data_info[split] = {"path": str(path), "bytes": path.stat().st_size,
                        "sha256": sha(path), "shapes": shapes, "ranges":ranges}
if len({info["sha256"] for info in data_info.values()}) != 4:
    raise ValueError("Identical files found in different splits.")
write_json(OUT / "data_manifest.json", data_info)
print(json.dumps(data_info, indent=2), flush=True)
# Record synthetic target geometry, independent of any observation model.
import math
if set(AUX_WEIGHTS) != {"quarter","half","full"} or any(
        not math.isfinite(v) or v < 0 for v in AUX_WEIGHTS.values()) or not any(AUX_WEIGHTS.values()):
    raise ValueError("Require three finite nonnegative auxiliary weights, at least one positive")
audit_path = OUT / "gt_target_spec.json"
write_json(audit_path, {"source":"GT only, never observed MS or wavelet LL",
    "filter":"separable normalized binomial FIR; replicate boundary; FP32",
    "orders":{"quarter":16,"half":4},
    "coefficients":{name:[math.comb(order,i)/2**order for i in range(order+1)]
                    for name,order in (("quarter",16),("half",4))},
    "coordinates":"j*(N-1)/(M-1); M=N/ratio; bilinear align_corners=True",
    "full":"identity", "aux_weights":AUX_WEIGHTS,
    "physical_status":"synthetic GT supervision, not a calibrated sensor model"})
write_json(OUT / "plan_deviations.json", {
    "baseline":"single original restored SSA-MRN forward, final MSE only",
    "candidate":"same core and auxiliary heads; final MSE + weighted auxiliary L1",
    "head_sites":["conv2ctc1","conv2ctc2","conv2ctc3"],
    "heads":"three identity-initialized 1x1 C-to-C readouts, training only",
    "k":K,"backbone_resampling":"original endpoint-aligned bilinear unchanged",
    "feedback":"none", "historical_comparison":"B1/B2/k6 runs not directly comparable"})

selection = {"seed": 42, "protocol": "RR", "tiles": "full scene",
             "zero_based_scene_indices": sorted(np.random.default_rng(42).choice(
                 data_info["RR"]["shapes"]["pan"][0], 5, replace=False).tolist()),
             "order": "ascending scene index", "rgb_assumed_bands_zero_based": [2, 1, 0],
             "rule": "fixed before training and test scoring"}
selection_path = OUT / "selection.json"
if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
    raise ValueError("Existing example selection differs; choose a new OUTPUT_DIR.")
write_json(selection_path, selection)

# No normalized on-disk cache: workers stream original H5 directly into their GPU.
CACHE = OUT / "_unused_cache"  # Retained only for compatibility with config schema.
print("[START WORKERS] Direct H5 -> GPU FP32; no normalized disk copy",flush=True)

# Separate CUDA processes keep seeds and optimizer states independent on T4 x2.
WORKER = r'''

import gc, hashlib, json, math, os, random, shutil, sys, time, traceback
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
from PIL import Image, ImageDraw

C = json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0, str(Path(C["source"]) / "SSA-MRN/src"))
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
from ssamrn.models.pan_variants import HighFrequency
from ssamrn.data.pancollection import PanCollectionH5
from ssamrn.metrics import rr_metrics, fr_metrics

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""): h.update(b)
    return h.hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu(v) for v in value)
    return value

def save(value, path):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp); os.replace(tmp, path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def model_for(config):
    return PASTED.build_model(config["variant"], config["channels"], config["k"])

def load_gpu_cache():
    size = sum(np.prod(v)*4 for split in ("train","val") for v in C["data"][split]["shapes"].values())
    free = torch.cuda.mem_get_info()[0]
    if size > free*.65: raise RuntimeError("Full FP32 dataset exceeds GPU activation reserve")
    result = {}; total = sum(C["data"][split]["shapes"][key][0] for split in ("train","val") for key in KEYS)
    done = 0; last = 0
    try:
        for split in ("train","val"):
            result[split] = {}
            with h5py.File(C["data"][split]["path"],"r") as h5:
                for key in KEYS:
                    shape = C["data"][split]["shapes"][key]
                    tensor = torch.empty(shape,device="cuda",dtype=torch.float32)
                    result[split][key] = tensor
                    for start in range(0,shape[0],128):
                        chunk = np.asarray(h5[key][start:start+128],dtype=np.float32)
                        if not np.isfinite(chunk).all(): raise ValueError(f"Nonfinite {split}/{key}")
                        tensor[start:start+len(chunk)].copy_(torch.from_numpy(chunk))
                        done += len(chunk)
                        if time.monotonic()-last >= 2 or start+len(chunk)==shape[0]:
                            status("loading_cache",cache_array=f"{split}/{key}",cache_percent=100*done/total)
                            print(f"GPU load {split}/{key}: {start+len(chunk)}/{shape[0]} | total {100*done/total:.1f}%",flush=True)
                            last=time.monotonic()
                    tensor.div_(C["scale"])
        torch.cuda.synchronize()
        return result
    except torch.cuda.OutOfMemoryError:
        result.clear()
        if "tensor" in locals(): del tensor
        gc.collect(); torch.cuda.empty_cache()
        raise RuntimeError("GPU dataset allocation failed; CPU fallback disabled")

def batches(split, batch_size, generator=None, shuffle=False, order=None, start_batch=0):
    n = C["data"][split]["shapes"]["gt"][0]
    if GPU_DATA is not None:
        if order is None:
            order = torch.randperm(n, generator=generator, device="cuda") if shuffle else torch.arange(n, device="cuda")
        for start in range(start_batch * batch_size, n, batch_size):
            indices = order[start:start + batch_size]
            batch = {k: v.index_select(0, indices) for k, v in GPU_DATA[split].items()}
            yield batch
    else:
        raise RuntimeError("GPU residency is required")

def layout(b):
    return {k: (v.contiguous(memory_format=FORMAT) if v.ndim == 4 else v) for k, v in b.items()}

def build():
    model = model_for(C).cuda().to(memory_format=FORMAT)
    return model


def make_optimizer(model):
    if C["fused_adam"]:
        return torch.optim.Adam(model.parameters(), lr=C["lr"], fused=True)
    return torch.optim.Adam(model.parameters(), lr=C["lr"], foreach=True)


def epoch_lr(epoch_index):
    if epoch_index < C["hold_lr_epochs"]:
        return C["lr"]
    progress = (epoch_index - C["hold_lr_epochs"] + 1) / (C["epochs"] - C["hold_lr_epochs"])
    return C["min_lr"] + .5 * (C["lr"] - C["min_lr"]) * (1 + math.cos(math.pi * progress))

def tune_micro():
    probe = {k: GPU_DATA["train"][k][:C["batch_size"]] for k in KEYS}
    probe = layout(probe); n = len(probe["gt"])
    records = []
    candidates = ([min(n, C["micro_batch"])] if not C["auto_micro"] else
                  sorted(set(min(n, b) for b in [C["batch_size"], 16, 8, 4, 1]), reverse=True))
    for micro in candidates:
        model = optimizer = scaler = pred = loss = None
        try:
            seed_all(C["seed"]); model = build()
            optimizer = make_optimizer(model)
            scaler = torch.amp.GradScaler("cuda", enabled=C["amp"])
            for repeat in range(3):
                torch.cuda.synchronize(); started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, n, micro):
                    end = min(offset + micro, n)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=C["amp"]):
                        pred, objective, _ = PASTED.training_loss(model,
                            tuple(probe[k][offset:end] for k in INPUTS), probe["gt"][offset:end], C["aux_weights"])
                        loss = objective * ((end - offset) / n)
                    scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                torch.cuda.synchronize()
                if repeat == 2:
                    if not torch.isfinite(loss).item(): raise RuntimeError("Nonfinite benchmark loss")
                    records.append({"micro_batch": micro, "seconds_per_effective_batch": time.monotonic() - started})
            # A full effective batch avoids extra launches; only test smaller
            # successful sizes near it. OOM candidates always continue downward.
            if len(records) >= 2: break
        except torch.cuda.OutOfMemoryError:
            pass
        finally:
            del model, optimizer, scaler, pred, loss
            gc.collect(); torch.cuda.empty_cache()
    del probe
    if not records: raise RuntimeError("Configured micro batch does not fit. Set LEGACY_MICRO_BATCH_SIZE=16 for BOTH arms and choose a new OUTPUT_DIR. GPU residency stays enabled.")
    micro = min(records, key=lambda x: x["seconds_per_effective_batch"])["micro_batch"]
    write(RUN / "benchmark.json", {"candidates": records, "chosen_micro_batch": micro,
          "cache": "GPU" if GPU_DATA is not None else "mmap",
          "memory_format": str(FORMAT)})
    return micro

CURRENT_STATUS = {}
def status(phase, **values):
    global CURRENT_STATUS
    # Carry only this epoch's metrics through validation/save, clear at every new epoch.
    changed_epoch = "epoch" in values and values["epoch"] != CURRENT_STATUS.get("epoch")
    changed_stage = phase.startswith("test_") or phase in ("complete","benchmark","loading_cache")
    if changed_epoch or changed_stage: CURRENT_STATUS = {}
    CURRENT_STATUS.update(values)
    state = {"run_id":RUN.name,"phase":phase,"device":torch.cuda.get_device_name(0),
        "physical_gpu":os.environ.get("CUDA_VISIBLE_DEVICES"),"epoch_target":C["epochs"],
        "allocated_gb":torch.cuda.memory_allocated()/1024**3,
        "reserved_gb":torch.cuda.memory_reserved()/1024**3,
        "peak_gb":torch.cuda.max_memory_allocated()/1024**3,
        "updated_unix":time.time(), **CURRENT_STATUS}
    write(RUN/"status.json",state)
    # Complete evidence remains on disk. The notebook displays only status.json.
    print("[STATUS] "+json.dumps(state,allow_nan=False),flush=True)

@torch.inference_mode()
def validate(model):
    model.eval(); sse = torch.zeros((), device="cuda"); count = 0
    psnr = torch.zeros((), device="cuda"); sam = torch.zeros((), device="cuda")
    samples = 0
    aux_sums = {name:torch.zeros((),device="cuda") for name in ("quarter","half","full")}
    for b in batches("val", C["eval_batch"]):
        b = layout(b)
        # FP32 validation makes best selection independent of AMP inference.
        pred, auxiliary = model(*(b[k] for k in INPUTS),return_aux=True)
        pred = pred.float(); gt = b["gt"]
        for name,ratio in (("quarter",4),("half",2),("full",1)):
            aux_sums[name] += F.l1_loss(auxiliary[name].float(),PASTED.gt_target(gt,ratio))*len(gt)
        diff = (pred - gt).square(); sse += diff.sum(); count += gt.numel()
        band_mse = diff.mean((2, 3)).clamp_min(1e-30)
        psnr += (-10 * band_mse.log10()).mean(1).sum(); samples += len(gt)
        norms = pred.norm(dim=1) * gt.norm(dim=1); valid = norms > 0
        angles = torch.rad2deg(torch.acos(((pred * gt).sum(1) / norms.clamp_min(1e-30)).clamp(-1, 1)))
        valid_counts = valid.flatten(1).sum(1)
        # Zero-valid pixels propagate NaN to the epoch-level finite check.
        sam += ((angles * valid).flatten(1).sum(1) / valid_counts).sum()
    result = {"val_mse": (sse / count).item(), "val_psnr_peak1": (psnr / samples).item(),
              "val_sam_deg": (sam / samples).item()}
    result.update({"val_aux_"+name+"_l1":(value/samples).item() for name,value in aux_sums.items()})
    if not all(math.isfinite(v) for v in result.values()):
        raise RuntimeError("Invalid validation metrics")
    return result

def measure_test_inference(model):
    timings = {}
    model.eval()
    with torch.inference_mode():
        for protocol in ("RR","FR"):
            with h5py.File(C["data"][protocol]["path"],"r") as h5:
                inputs = [torch.from_numpy(np.asarray(h5[k][:1],dtype=np.float32)/C["scale"]).cuda().contiguous(memory_format=FORMAT) for k in INPUTS]
            model(*inputs)  # warmup; no scoring or example selection
            torch.cuda.synchronize(); begin = time.monotonic()
            for _ in range(3): model(*inputs)
            torch.cuda.synchronize()
            timings[protocol] = (time.monotonic()-begin)/3
    write(RUN / "test_timing_probe.json",{"seconds_per_scene":timings,
        "scope":"random initialization, inference only; metrics/I/O/packaging unmeasured", "scene_index":0})
    return sum(timings[p]*C["data"][p]["shapes"]["pan"][0] for p in ("RR","FR"))


# Embedded by build_pyramid_supervision_cell.py; globals come from the cell worker.
LOSS_KEYS = ("final_mse", "aux_quarter_l1", "aux_half_l1", "aux_full_l1", "objective")

def train():
    status("benchmark")
    micro = tune_micro()
    seed_all(C["seed"])
    gen = torch.Generator(device="cuda").manual_seed(C["seed"])
    model = build()
    test_inference_estimate = measure_test_inference(model)
    opt = make_optimizer(model)
    core_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        core_hash.update(name.encode()); core_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    write(RUN/"initialization.json", {"seed":C["seed"],"core_sha256":core_hash.hexdigest(),
          "excluded_candidate_layers":[],"includes_identical_auxiliary_heads":True})
    scaler = torch.amp.GradScaler("cuda",enabled=C["amp"])
    start, best, best_epoch, history, resume_mid = 0, float("inf"), 0, [], None
    best_bytes = None

    def save_best_bytes(blob):
        tmp = RUN/"best.pt.tmp"
        tmp.write_bytes(blob); os.replace(tmp,RUN/"best.pt")

    if (RUN/"latest.pt").exists():
        ck = torch.load(RUN/"latest.pt",map_location="cpu",weights_only=True)
        if ck["config"] != C: raise ValueError("Resume configuration differs; choose a new OUTPUT_DIR")
        model.load_state_dict(ck["model"],strict=True); opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["cpu_rng"]); torch.cuda.set_rng_state_all(ck["cuda_rng"])
        random.setstate(ck["python_rng"])
        st = ck["numpy_rng"]
        np.random.set_state((st[0],np.array(st[1],dtype=np.uint32),st[2],st[3],st[4]))
        gen.set_state(ck["loader_rng"])
        start,best,best_epoch,history = ck["epoch"],ck["best_mse"],ck["best_epoch"],ck["history"]
        resume_mid = ck["mid_epoch"]; best_bytes = ck["best_recovery_bytes"]
        if best_epoch and not best_bytes: raise RuntimeError("Missing original best bytes in latest")
        if best_bytes: save_best_bytes(best_bytes)
        # Recover measured save timings where the atomic history companion exists.
        if (RUN/"history.json").exists():
            companion = json.loads((RUN/"history.json").read_text())
            if len(companion) == len(history) and all(a["sample_order_sha256"] == b["sample_order_sha256"]
                    and a["train_mse"] == b["train_mse"] for a,b in zip(companion,history)):
                history = companion
        del ck
    total_batches = math.ceil(C["data"]["train"]["shapes"]["gt"][0]/C["batch_size"])

    def snapshot(completed_epoch,mid=None):
        st = np.random.get_state()
        return {"config":C,"epoch":completed_epoch,"best_mse":best,"best_epoch":best_epoch,
            "model":cpu(model.state_dict()),"optimizer":cpu(opt.state_dict()),"scaler":scaler.state_dict(),
            "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
            "loader_rng":gen.get_state(),"python_rng":random.getstate(),
            "numpy_rng":(st[0],st[1].tolist(),st[2],st[3],st[4]),
            "history":history,"mid_epoch":mid,"best_recovery_bytes":best_bytes,
            "lr_state":{"completed_epoch":completed_epoch,"hold":C["hold_lr_epochs"],
                        "minimum":C["min_lr"],"initial":C["lr"]}}

    if not (RUN/"latest.pt").exists(): save(snapshot(0),RUN/"latest.pt")

    for epoch in range(start,C["epochs"]):
        lr = epoch_lr(epoch)
        for group in opt.param_groups: group["lr"] = lr
        n_samples = C["data"]["train"]["shapes"]["gt"][0]
        if resume_mid is not None:
            if resume_mid["epoch_index"] != epoch: raise ValueError("Invalid recovered epoch")
            order = resume_mid["order"].cuda(); completed = resume_mid["batches_done"]
            if not 0 <= completed <= total_batches or len(order) != n_samples:
                raise ValueError("Invalid recovered batch cursor")
            expected_hash = hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
            if expected_hash != resume_mid["sample_order_sha256"]:
                raise ValueError("Recovered permutation hash mismatch")
            if not torch.equal(order.sort().values,torch.arange(n_samples,device="cuda")):
                raise ValueError("Recovered order is not a permutation")
            sums = resume_mid["loss_sums"].cuda(); samples = resume_mid["samples_seen"]
            if samples != min(n_samples,completed*C["batch_size"]):
                raise ValueError("Recovered sample accumulator mismatch")
            seconds_before = resume_mid["train_seconds"]; order_hash = expected_hash
            resume_mid = None
        else:
            order = torch.randperm(n_samples,generator=gen,device="cuda")
            order_hash = hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
            completed = 0; sums = torch.zeros(len(LOSS_KEYS),device="cuda",dtype=torch.float64)
            samples = 0; seconds_before = 0.
        model.train(); torch.cuda.synchronize()
        tick = time.monotonic(); last_status = last_checkpoint = tick; last_step = completed

        def emit(phase,step,metrics=None):
            torch.cuda.synchronize()
            elapsed = seconds_before+time.monotonic()-tick
            progress = {"epoch":epoch+1,"batch":step,"batches":total_batches,"lr":lr,
                "epoch_percent":100*step/total_batches,
                "training_percent":100*(epoch*total_batches+step)/(C["epochs"]*total_batches),
                "train_elapsed_seconds":elapsed,"aux_active":any(C["aux_weights"].values()),
                "eta_scope":"measured train+validation+checkpoints after first completed epoch; metrics/I/O/packaging excluded"}
            if samples:
                losses = dict(zip(LOSS_KEYS,(sums/samples).cpu().tolist()))
                progress["losses"] = losses
                progress["weighted_aux_over_final"] = (losses["objective"]-losses["final_mse"])/max(losses["final_mse"],1e-30)
            prior = history[-5:]
            estimate = float(np.mean([r["train_seconds"] for r in prior])) if prior else (elapsed/step*total_batches if step else None)
            if estimate is not None:
                left = max(0.,estimate-elapsed)+(C["epochs"]-epoch-1)*estimate
                progress["train_remaining_seconds"] = left
                if prior:
                    extra = float(np.mean([r["validation_seconds"]+r.get("checkpoint_seconds",0) for r in prior]))
                    end = time.time()+left+(C["epochs"]-epoch)*extra
                    progress["training_finish_unix"] = end
                    progress["exit_finish_unix"] = end+test_inference_estimate
                    progress["exit_eta_scope"] = "provisional train+validation+checkpoints+test inference; test metrics/I/O/packaging unmeasured"
            if metrics: progress["metrics"] = metrics
            status(phase,**progress)

        emit("train",completed)
        for step,b in enumerate(batches("train",C["batch_size"],order=order,start_batch=completed),completed+1):
            b = layout(b); n = len(b["gt"]); opt.zero_grad(set_to_none=True)
            for offset in range(0,n,micro):
                end = min(offset+micro,n)
                with torch.autocast("cuda",dtype=torch.float16,enabled=C["amp"]):
                    pred,loss,parts = PASTED.training_loss(model,tuple(b[k][offset:end] for k in INPUTS),
                                                         b["gt"][offset:end],C["aux_weights"])
                scaler.scale(loss*((end-offset)/n)).backward()
                sums += torch.stack([parts[k].detach() for k in LOSS_KEYS]).double()*(end-offset)
                samples += end-offset
            scaler.step(opt); scaler.update(); last_step = step
            now = time.monotonic()
            if now-last_status >= C["status_seconds"]:
                emit("train",step); last_status = time.monotonic()
            if now-last_checkpoint >= C["checkpoint_seconds"] and step < total_batches:
                emit("checkpoint",step)
                mid = {"epoch_index":epoch,"batches_done":step,"order":order.cpu(),
                    "sample_order_sha256":order_hash,"loss_sums":sums.cpu(),"samples_seen":samples,
                    "train_seconds":seconds_before+time.monotonic()-tick}
                recovery = snapshot(epoch,mid); save(recovery,RUN/"latest.pt"); del recovery
                last_checkpoint = time.monotonic(); emit("train",step)
        losses = dict(zip(LOSS_KEYS,(sums/samples).cpu().tolist()))
        if not all(math.isfinite(v) for v in losses.values()): raise RuntimeError("Nonfinite training objective")
        torch.cuda.synchronize(); training_seconds = seconds_before+time.monotonic()-tick
        emit("validation",last_step)
        val_started = time.monotonic(); validation = validate(model)
        validation_seconds = time.monotonic()-val_started
        improved = validation["val_mse"] < best
        if improved: best,best_epoch = validation["val_mse"],epoch+1
        row = {"epoch":epoch+1,"train_mse":losses["final_mse"],"train_objective":losses["objective"],
            **{k:v for k,v in losses.items() if k.startswith("aux_")}, **validation,"lr":lr,
            "seconds":training_seconds+validation_seconds,"train_seconds":training_seconds,
            "validation_seconds":validation_seconds,"train_samples_per_second":n_samples/max(training_seconds,1e-9),
            "scaler_scale":scaler.get_scale(),"micro_batch":micro,
            "peak_gpu_gb":torch.cuda.max_memory_allocated()/1024**3,"sample_order_sha256":order_hash}
        history.append(row); emit("checkpoint",total_batches,metrics=row)
        checkpoint_started = time.monotonic()
        if improved:
            winner = snapshot(epoch+1)
            winner["best_recovery_bytes"] = None  # Avoid recursively nesting old winners.
            save(winner,RUN/"best.pt"); best_bytes = (RUN/"best.pt").read_bytes(); del winner
        else:
            save_best_bytes(best_bytes)  # Byte-identical winner, atomically refreshed.
        ck = snapshot(epoch+1); save(ck,RUN/"latest.pt")
        row["checkpoint_seconds"] = time.monotonic()-checkpoint_started
        if epoch+1 == C["hold_lr_epochs"]:
            shutil.copy2(RUN/"best.pt",RUN/"best_first100.pt")
        write(RUN/"history.json",history)
        remaining = C["epochs"]-epoch-1
        mean = float(np.mean([r["seconds"]+r.get("checkpoint_seconds",0) for r in history[-5:]]))
        status("epoch_complete",epoch=epoch+1,batch=total_batches,batches=total_batches,lr=lr,
            epoch_percent=100.,training_percent=100*(epoch+1)/C["epochs"],losses=losses,metrics=row,
            training_finish_unix=time.time()+remaining*mean,
            exit_finish_unix=time.time()+remaining*mean+test_inference_estimate,
            exit_eta_scope="recent measured epochs+test inference; metrics/I/O/packaging excluded")
        del ck,order
        print(json.dumps(row),flush=True)
    write(RUN/"training_complete.json",{"config":C,"epoch":C["epochs"],"best_epoch":best_epoch,"best_mse":best})
    del model,opt,scaler; gc.collect(); torch.cuda.empty_cache()
    return history


def panels(sample, prediction, path, index):
    bands = [2, 1, 0]; ref = sample["gt"][bands].transpose(1, 2, 0)
    lo, hi = np.percentile(ref, [1, 99], axis=(0, 1))
    def rgb(a):
        a = a[bands].transpose(1, 2, 0)
        return Image.fromarray((np.clip((a - lo) / np.maximum(hi - lo, 1e-8), 0, 1) * 255).astype("uint8"))
    pan = sample["pan"][0]; p0, p1 = np.percentile(pan, [1, 99])
    gray = Image.fromarray((np.clip((pan - p0) / max(p1 - p0, 1e-8), 0, 1) * 255).astype("uint8")).convert("RGB")
    h, w = pan.shape; canvas = Image.new("RGB", (4 * w, h + 40), "white")
    draw = ImageDraw.Draw(canvas)
    for j, (name, image) in enumerate([("LR MS", rgb(sample["ms"])), ("PAN", gray),
                                      ("Prediction", rgb(prediction)), ("GT", rgb(sample["gt"]))]):
        canvas.paste(image.resize((w, h), Image.Resampling.NEAREST), (j * w, 40))
        draw.text((j * w + 6, 5), name, fill="black")
    draw.text((6, 24), f'{C["sensor"]} RR scene {index} | shared GT 1-99% stretch | full scene', fill="black")
    canvas.save(path)
    return {"lo": lo.tolist(), "hi": hi.tolist(), "pan_lo_hi": [float(p0), float(p1)]}

def evaluate():
    ck = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
    if ck["config"] != C: raise ValueError("Best configuration mismatch")
    model = build(); model.load_state_dict(ck["model"], strict=True); model.eval()
    result = {"config": C, "best_epoch": ck["epoch"], "best_mse": ck["best_mse"],
              "checkpoints": {kind: {"sha256": sha(RUN / f"{kind}.pt"),
                  "bytes": (RUN / f"{kind}.pt").stat().st_size,
                  "epoch": torch.load(RUN / f"{kind}.pt",map_location="cpu",weights_only=True)["epoch"],
                  "original_path":str(RUN / f"{kind}.pt")} for kind in ("best", "latest")},
              "metrics": {}, "images": [], "test_precision": "FP32", "full_scene": True}
    del ck
    for protocol in ("RR", "FR"):
        rows = []; tick = time.monotonic()
        if C["save_predictions"]: (RUN / "predictions" / protocol).mkdir(parents=True, exist_ok=True)
        with h5py.File(C["data"][protocol]["path"], "r") as h5:
            keys = INPUTS + (("gt",) if protocol == "RR" else ())
            for index in range(len(h5["pan"])):
                measured = (time.monotonic()-tick)/index if index else None
                eta = {"protocol_finish_unix":time.time()+measured*(len(h5["pan"])-index),
                       "exit_eta_scope":"measured current-protocol scenes with metrics and I/O; later protocols/packaging excluded"} if measured else {}
                if measured and protocol == "FR": eta["exit_finish_unix"] = eta["protocol_finish_unix"]
                status("test_"+protocol,scene=index+1,scenes=len(h5["pan"]),
                       protocol_percent=100*index/len(h5["pan"]),**eta)
                arrays = {k: np.asarray(h5[k][index], dtype=np.float32) for k in keys}
                if any(not np.isfinite(v).all() for v in arrays.values()): raise ValueError("Nonfinite test input")
                inputs = [torch.from_numpy(arrays[k] / C["scale"]).unsqueeze(0).cuda().contiguous(
                          memory_format=FORMAT) for k in INPUTS]
                begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                with torch.inference_mode():
                    begin.record()
                    prediction = model(*inputs).float()
                    end.record(); end.synchronize()
                    inference_ms = begin.elapsed_time(end)
                    pred = prediction[0].cpu().numpy() * C["scale"]
                del prediction
                if not np.isfinite(pred).all(): raise RuntimeError("Nonfinite prediction")
                if protocol == "RR":
                    metrics = rr_metrics(arrays["gt"], pred)
                    metrics["MSE_peak1"] = float(np.mean(((pred.astype(np.float64) - arrays["gt"]) / C["scale"])**2))
                    metrics["PSNR_sensor_peak"] = metrics["PSNR"] + 20 * math.log10(C["scale"] / 2047.)
                    if index in C["selection"]["zero_based_scene_indices"]:
                        name = f"RR_scene_{index:02}.png"
                        result["images"].append({"scene_index": index, "file": name,
                                                "display": panels(arrays, pred, RUN / name, index)})
                else: metrics = fr_metrics(pred, arrays["lms"], arrays["ms"], arrays["pan"])
                metrics["inference_ms"] = inference_ms
                if not all(math.isfinite(v) for v in metrics.values()): raise RuntimeError("Nonfinite test metric")
                if C["save_predictions"]:
                    np.savez_compressed(RUN / "predictions" / protocol / f"scene_{index:03}.npz",
                                        prediction_dn=pred, scene_index=index, scale=C["scale"])
                rows.append({"scene_index": index, **metrics})
                write(RUN / f"{protocol}_per_scene.json", rows)
                del inputs
        result["metrics"][protocol] = {"count": len(rows), "rows": rows,
             "mean": {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "scene_index"},
             "seconds": time.monotonic() - tick}
    if len(result["images"]) != 5: raise RuntimeError("Incomplete fixed RR examples")
    write(RUN / "results.json", result)
    write(RUN / "evaluation_complete.json", {"best_sha256": result["checkpoints"]["best"]["sha256"],
          "RR": result["metrics"]["RR"]["count"], "FR": result["metrics"]["FR"]["count"]})
    status("complete", epoch=C["epochs"], best_epoch=result["best_epoch"], metrics=result["metrics"]["RR"]["mean"])

if __name__ == "__main__":
    C = json.loads(Path(sys.argv[1]).read_text()); RUN = Path(C["run_dir"])
    RUN.mkdir(parents=True, exist_ok=True)
    if C["mode"] in ("B0", "B0_DS"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("B0_DS_models", C["B0_DS_source"])
        PASTED = importlib.util.module_from_spec(spec)
        # The original setup's constants are available to its function bodies.
        PASTED.__dict__.update({"SENSOR": C["sensor"], "K": C["k"], "EPOCHS": C["epochs"],
            "EFFECTIVE_BATCH_SIZE": C["batch_size"], "MICRO_BATCH_SIZE": C["micro_batch"],
            "LEARNING_RATE": C["lr"], "OUTPUT_DIR": str(RUN.parent),
            "TRAIN_H5": C["data"]["train"]["path"], "VAL_H5": C["data"]["val"]["path"],
            "SOURCE_ROOT": str(Path(C["source"]) / "SSA-MRN/src")})
        spec.loader.exec_module(PASTED)
        C["reference_plan_config"] = {**PASTED.train_config(C["variant"], C["seed"]), "aux_weights":C["aux_weights"]}
    KEYS = ("pan", "lms", "ms", "gt"); INPUTS = KEYS[:3]
    torch.cuda.set_device(0); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(C["controlled"])
    torch.backends.cudnn.benchmark = not C["controlled"]
    torch.backends.cuda.matmul.allow_tf32 = not C["controlled"]
    torch.backends.cudnn.allow_tf32 = not C["controlled"]
    FORMAT = torch.channels_last if C["channels_last"] else torch.contiguous_format
    GPU_DATA = None; LOADERS = {}
    try:
        run_started = time.monotonic()
        status("smoke_checks")
        seed_all(C["seed"])
        first = PASTED.build_model("B0", C["channels"], C["k"])
        seed_all(C["seed"])
        second = PASTED.build_model("B0_DS", C["channels"], C["k"])
        if any(not torch.equal(v, second.state_dict()[k]) for k,v in first.state_dict().items()):
            raise RuntimeError("B0/B0_DS parameter initialization differs")
        del first
        second = second.cuda()
        with h5py.File(C["data"]["train"]["path"],"r") as h5:
            example = {k:torch.from_numpy(np.asarray(h5[k][:1],dtype=np.float32)/C["scale"]).cuda() for k in KEYS}
        pred, objective, components = PASTED.training_loss(second,
            tuple(example[k] for k in INPUTS), example["gt"], C["candidate_aux_weights"])
        if pred.shape != example["gt"].shape: raise RuntimeError("Smoke output shape mismatch")
        objective.backward()
        gradients = [p.grad for p in second.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError("Smoke gradient check failed")
        if any(h.weight.grad is None or not h.weight.grad.abs().sum().item() for h in second.aux_heads.values()):
            raise RuntimeError("Auxiliary head has no gradient")
        for ratio in (2,4):
            constant = torch.ones(1,4,64,64,device="cuda")
            torch.testing.assert_close(PASTED.gt_target(constant,ratio),
                constant[...,::ratio,::ratio],atol=2e-6,rtol=0)
        write(RUN/"smoke_checks.json",{"matching_initialization":True,"shape":list(pred.shape),
            "finite_gradients":True,"three_head_gradients":True,"constant_target":True,"optimizer_steps":0})
        del second, example, pred, objective, components, gradients
        gc.collect(); torch.cuda.empty_cache()
        status("loading_cache")
        GPU_DATA = load_gpu_cache()
        profile = json.loads(Path(C["gt_target_profile"]).read_text())
        if sha(C["gt_target_profile"]) != C["gt_target_profile_sha256"]:
            raise ValueError("GT target manifest changed")
        train()
        training_wall_seconds = time.monotonic() - run_started
        for loader in LOADERS.values():
            if hasattr(loader.dataset, "close"): loader.dataset.close()
        GPU_DATA = None; LOADERS.clear(); gc.collect(); torch.cuda.empty_cache()
        evaluate()
        write(RUN / "session_timing.json", {"training_wall_seconds_this_session": training_wall_seconds,
              "total_wall_seconds_this_session": time.monotonic() - run_started,
              "scope": "cache load + benchmark + completed/replayed training + checkpoint saves + test; resumed session is not whole-experiment time"})
    except BaseException:
        write(RUN / "failure.json", {"traceback": traceback.format_exc(), "time": time.time()})
        traceback.print_exc(); sys.exit(1)
'''

worker_path = OUT / "code" / "kaggle_worker.py"
worker_hash = hashlib.sha256(WORKER.encode()).hexdigest()
if worker_path.exists() and sha(worker_path) != worker_hash:
    raise ValueError("Worker code changed; choose a new OUTPUT_DIR.")
worker_path.write_text(WORKER, encoding="utf-8")
gpu_count = torch.cuda.device_count() if USE_ALL_GPUS else 1
if len(TRAINING_MODES) == 2:
    gpu_count = min(gpu_count, 2)
    if gpu_count == 2:
        a, b = (torch.cuda.get_device_properties(i) for i in (0, 1))
        if a.name != b.name or a.total_memory != b.total_memory:
            raise ValueError("Controlled parallel comparison requires two equivalent GPUs.")
    print("Pair assignment: GPU 0=B0; GPU 1=B0_DS (swap optional)" if gpu_count == 2 else
          "Only one GPU: paired modes will run sequentially, not concurrently.")
configs = []
for mode, seed, variant in [(m, s, v) for m in TRAINING_MODES for s in SEEDS
                           for v in [m]]:
        run_id = f"{mode}_{SENSOR}_{variant}_k{K}_s{seed}"
        config = {"sensor": SENSOR, "k": K, "channels": 8 if SENSOR == "WV3" else 4,
                  "mode": mode, "controlled": CONTROLLED_COMPARISON,
                  "epochs": EPOCHS, "seed": seed, "variant": variant,
                  "hold_lr_epochs": HOLD_LR_EPOCHS, "min_lr": MIN_LEARNING_RATE,
                  "status_seconds": STATUS_INTERVAL_SECONDS, "checkpoint_seconds": CHECKPOINT_INTERVAL_SECONDS,
                  "fused_adam": FUSED_ADAM,
                  "batch_size": EFFECTIVE_BATCH_SIZE, "lr": LEARNING_RATE,
                  "micro_batch": min(LEGACY_MICRO_BATCH_SIZE, EFFECTIVE_BATCH_SIZE),
                  "auto_micro": not CONTROLLED_COMPARISON,
                  "eval_batch": min(LEGACY_MICRO_BATCH_SIZE, EFFECTIVE_BATCH_SIZE),
                  "amp": AMP,"torch":str(torch.__version__),"cuda":torch.version.cuda,
                  "channels_last": CHANNELS_LAST,
                  "gpu_cache": GPU_CACHE, "workers": NUM_WORKERS, "scale": SCALE,
                  "save_predictions": SAVE_PREDICTIONS, "data": data_info,
                  "cache": str(CACHE), "source": str(SOURCE), "run_dir": str(OUT / "runs" / run_id),
                  "selection": selection, "repo_commit": REPO_COMMIT,
                  "plan_commit": PLAN_COMMIT,
                  "B0_DS_source": str(custom_path), "B0_DS_source_sha256": custom_hash,
                  "upstream_commit": UPSTREAM_COMMIT, "worker_sha256": worker_hash,
                  "gt_target_profile": str(audit_path), "gt_target_profile_sha256": sha(audit_path),
                  "model_definition": "single original core; identical three training-only heads; loss-only contrast",
                  "aux_weights": (AUX_WEIGHTS if variant == "B0_DS" else {name:0.0 for name in AUX_WEIGHTS}),
                  "candidate_aux_weights":AUX_WEIGHTS,
                  "numerics": ("Controlled: equal AMP/layout/micro batch, deterministic kernels, no TF32; not legacy bitwise parity" if CONTROLLED_COMPARISON
                               else "Speed profile: both arms use AMP/channels-last and measured micro batch; micro batches may differ")}
        cfg_path = OUT / "code" / f"{run_id}.json"
        if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
            raise ValueError(f"Settings changed for {run_id}; choose a new OUTPUT_DIR.")
        write_json(cfg_path, config)
        configs.append((config, cfg_path))
write_json(OUT / "environment.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
           "gpus": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
           "python": sys.version, "parallel_workers": min(gpu_count, len(configs)),
           "swap_gpus": SWAP_GPUS,
           "old_checkpoints": "legacy not imported; matching suite Output restores full optimizer/scaler/RNG/batch cursor"})

pending = list(configs)
active = {}
finished = []
try:
    while pending or active:
        for device in range(gpu_count):
            if device in active or not pending:
                continue
            if gpu_count == 2 and len(TRAINING_MODES) == 2:
                reference_device = 1 if SWAP_GPUS else 0
                assigned_mode = "B0" if device == reference_device else "B0_DS"
                eligible = next((i for i, (c, p) in enumerate(pending) if c["mode"] == assigned_mode), None)
                if eligible is None: continue
            else: eligible = 0
            config, cfg_path = pending.pop(eligible)
            run = Path(config["run_dir"]); run.mkdir(parents=True, exist_ok=True)
            status_path = run / "status.json"
            complete = run / "evaluation_complete.json"
            if complete.exists() and (run / "results.json").exists():
                prior_result = json.loads((run / "results.json").read_text())
                restored_config = {k: v for k, v in prior_result["config"].items() if k != "reference_plan_config"}
                if restored_config != config:
                    raise ValueError(f"Completed run config differs: {run}")
                essential = ["best.pt","latest.pt","history.json","RR_per_scene.json","FR_per_scene.json"]
                essential += [f"RR_scene_{i:02}.png" for i in selection["zero_based_scene_indices"]]
                if any(not (run/name).is_file() for name in essential):
                    raise ValueError(f"Completed output is incomplete: {run}; restore the missing originals")
                for kind in ("best","latest"):
                    if sha(run/f"{kind}.pt") != prior_result["checkpoints"][kind]["sha256"]:
                        raise ValueError(f"Completed checkpoint hash mismatch: {run}")
                finished.append(run)
                continue
            if (run / "failure.json").exists(): (run / "failure.json").unlink()
            write_json(status_path, {"phase": "starting", "run_id": run.name})
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(device), "OMP_NUM_THREADS": "2",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
            log = (run / "worker.log").open("a", encoding="utf-8")
            process = subprocess.Popen([sys.executable, "-u", str(worker_path), str(cfg_path)],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            active[device] = (process, log, run)
        clear_output(wait=True)
        from datetime import datetime, timezone, timedelta
        kst = timezone(timedelta(hours=9))
        def stamp(value):
            return datetime.fromtimestamp(value,kst).strftime("%Y-%m-%d %H:%M:%S KST") if value else "측정 대기"
        print("SSA-MRN k4 · 현재 에포크 · 화면 갱신 5초 / 상태 기록 15초")
        print(f"완료 작업 {len(finished)}/{len(configs)} | 대기 작업 {len(pending)} | {stamp(time.time())}")
        current_finishes = []; scopes = []
        for device,(process,log,run) in list(active.items()):
            try: state = json.loads((run/"status.json").read_text())
            except (OSError,ValueError): state = {"phase":"starting"}
            print(f"\nGPU {device} · {run.name} · {state.get('phase')}")
            print(f"  현재 epoch {state.get('epoch','-')}/{EPOCHS} | batch {state.get('batch','-')}/{state.get('batches','-')} "
                  f"| epoch {state.get('epoch_percent',0):.1f}% | 전체 학습 배치 {state.get('training_percent',0):.1f}%")
            print(f"  GPU 메모리 allocated/reserved/peak: {state.get('allocated_gb',0):.2f} / "
                  f"{state.get('reserved_gb',0):.2f} / {state.get('peak_gb',0):.2f} GB")
            if "cache_percent" in state: print(f"  데이터 적재 {state['cache_array']}: {state['cache_percent']:.1f}%")
            if "lr" in state: print(f"  lr {state['lr']:.3g} | 현재 epoch 처리 {state.get('train_elapsed_seconds',0):.1f}초")
            if state.get("losses"):
                p = state["losses"]
                print(f"  현재 epoch MSE {p['final_mse']:.7f} | 총 loss {p['objective']:.7f}")
                if state.get('aux_active'):
                    print(f"  보조 L1 1/4={p['aux_quarter_l1']:.5f}, 1/2={p['aux_half_l1']:.5f}, full={p['aux_full_l1']:.5f}")
                else: print("  보조 손실 비활성 (기준 모델)")
            m = state.get("metrics",{})
            if "val_mse" in m:
                print(f"  현재 epoch validation MSE {m['val_mse']:.7f} | PSNR {m['val_psnr_peak1']:.3f} dB | SAM {m['val_sam_deg']:.3f}°")
            print(f"  학습·검증 종료 예상: {stamp(state.get('training_finish_unix'))}")
            print(f"  GPU 작업 종료 예상: {stamp(state.get('exit_finish_unix'))}")
            if "exit_finish_unix" in state:
                current_finishes.append(state["exit_finish_unix"]); scopes.append(state.get('exit_eta_scope','측정 대기'))
            if state.get("exit_eta_scope"): print("  ETA 범위: "+state["exit_eta_scope"])
            if "scene" in state:
                print(f"  현재 평가 {state.get('phase')}: {state['scene']}/{state['scenes']} 장면 "
                      f"({state.get('protocol_percent',0):.1f}% 완료) | 해당 평가 종료 {stamp(state.get('protocol_finish_unix'))}")
            queued = sum(1 for cfg,_ in pending if (cfg['mode']=='B0') == (device==(1 if SWAP_GPUS else 0)))
            if queued: print(f"  GPU 대기 큐 {queued}개: 미측정 작업 시간, 위 ETA에서 제외")
            age = time.time()-state.get("updated_unix",time.time())
            if age > STATUS_INTERVAL_SECONDS*3: print(f"  마지막 상태 기록 {age:.0f}초 전 (진행률은 추정해 올리지 않음)")
            rc = process.poll()
            if rc is not None:
                log.close(); del active[device]
                if rc:
                    failure = run/"failure.json"
                    detail = failure.read_text() if failure.exists() else "worker.log에서 실패 원인을 확인하세요."
                    raise RuntimeError(f"Worker failed: {run}\n{detail}")
                finished.append(run)
        if current_finishes and len(current_finishes)==len(active) and not pending:
            print(f"\nGPU 작업 잠정 종료: {stamp(max(current_finishes))} | 아래 측정 범위 밖 시간은 별도")
            for scope in sorted(set(scopes)): print("  "+scope)
        else: print("\n전체 작업 종료: 측정 대기 (대기 모델·평가·압축의 미측정 시간 포함)")
        print("이전 epoch 로그는 파일에 보존됩니다. 화면에는 현재 epoch만 표시됩니다.")
        if pending or active: time.sleep(5)
finally:
    # Stopping this cell also stops its child training processes.
    for process, log, run in active.values():
        process.terminate()
    for process, log, run in active.values():
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()
    # A stopped cell remains recoverable: preserve only this suite's files.
    if len(finished) != len(configs):
        recovery = OUT.parent / (OUT.name + "_recovery.zip")
        try:
            with zipfile.ZipFile(recovery, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
                for path in sorted(OUT.rglob("*")):
                    if path.is_file() and path.suffix not in (".tmp", ".pyc") and "__pycache__" not in path.parts:
                        bundle.write(path, arcname=str(Path(OUT.name) / path.relative_to(OUT)))
            print(f"[RECOVERY] {recovery}; Save Version with Output to retain checkpoints", flush=True)
        except OSError as error:
            print(f"[RECOVERY] ZIP unavailable: {error}; suite files remain at {OUT}", flush=True)

clear_output(wait=True)
print("학습·전체 RR/FR 평가 종료. 실제 산출물 비교·그래프·해시·ZIP 검증 중입니다.")
print("산출물 보관 단계 ETA: 측정 대기; 이 단계가 끝나야 전체 코드 실행 완료입니다.")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
results = {Path(c["run_dir"]).name: json.loads((Path(c["run_dir"]) / "results.json").read_text())
           for c, _ in configs}
summary_rows = []
for run_id, result in results.items():
    seed = result["config"]["seed"]
    mode = result["config"]["mode"]
    base = results[f"B0_{SENSOR}_B0_k{K}_s{seed}"]["metrics"]
    result["delta_from_same_seed_B0"] = {
        protocol: {key: val - base[protocol]["mean"][key] for key, val in block["mean"].items()}
        for protocol, block in result["metrics"].items()}
    for protocol, block in result["metrics"].items():
        for key, val in block["mean"].items():
            summary_rows.append({"run_id": run_id, "mode": mode, "seed": seed, "best_epoch": result["best_epoch"],
                "protocol": protocol, "metric": key, "mean": val,
                "delta_from_B0": result["delta_from_same_seed_B0"][protocol][key]})
write_json(OUT / "results.json", results)
with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
    writer.writeheader(); writer.writerows(summary_rows)

for seed in SEEDS:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for mode, variant in [("B0", "B0"), ("B0_DS", "B0_DS")]:
        run_id = f"{mode}_{SENSOR}_{variant}_k{K}_s{seed}"
        history = json.loads((OUT / "runs" / run_id / "history.json").read_text())
        for ax, key in zip(axes, ["train_mse", "val_mse", "val_psnr_peak1", "val_sam_deg"]):
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=variant)
            ax.set_title(key); ax.set_xlabel("Epoch"); ax.grid(alpha=.25)
    axes[0].set_yscale("log"); axes[1].set_yscale("log"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"learning_seed{seed}.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for ax,name in zip(axes,("quarter","half","full")):
        for mode in TRAINING_MODES:
            records = json.loads((OUT/"runs"/f"{mode}_{SENSOR}_{mode}_k{K}_s{seed}"/"history.json").read_text())
            ax.plot([r["epoch"] for r in records],[r["val_aux_"+name+"_l1"] for r in records],label=mode+" validation")
            if mode=="B0_DS":
                ax.plot([r["epoch"] for r in records],[r["aux_"+name+"_l1"] for r in records],linestyle="--",label=mode+" train")
        ax.set_title(name+" GT target L1"); ax.set_xlabel("Epoch"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT/f"auxiliary_seed{seed}.png",dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (protocol, key) in zip(axes.flat, [("RR", "PSNR"), ("RR", "SAM"),
            ("RR", "MSE_peak1"), ("RR", "ERGAS"), ("RR", "Q2n"), ("FR", "QNR")]):
        value = results[f"B0_DS_{SENSOR}_B0_DS_k{K}_s{seed}"]["delta_from_same_seed_B0"][protocol][key]
        ax.bar(["B0_DS"], [value], color="#2788b5")
        ax.set_title(f"{protocol} {key}: B0_DS minus B0" + (" (provisional)" if protocol == "FR" else ""))
        ax.axhline(0, color="black", linewidth=.8); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(OUT / f"test_comparison_seed{seed}.png", dpi=150); plt.close(fig)

paired = []
if set(TRAINING_MODES) == {"B0", "B0_DS"}:
    for seed in SEEDS:
        for variant in ["B0_DS"]:
            old_id = f"B0_{SENSOR}_B0_k{K}_s{seed}"
            new_id = f"B0_DS_{SENSOR}_{variant}_k{K}_s{seed}"
            old, new = results[old_id], results[new_id]
            old_h = json.loads((OUT / "runs" / old_id / "history.json").read_text())
            new_h = json.loads((OUT / "runs" / new_id / "history.json").read_text())
            old_sec, new_sec = sum(r["seconds"] for r in old_h), sum(r["seconds"] for r in new_h)
            old_init = json.loads((OUT / "runs" / old_id / "initialization.json").read_text())
            new_init = json.loads((OUT / "runs" / new_id / "initialization.json").read_text())
            if old_init["core_sha256"] != new_init["core_sha256"]:
                raise RuntimeError("Model core initialization differs between comparison arms")
            if [r["sample_order_sha256"] for r in old_h] != [r["sample_order_sha256"] for r in new_h]:
                raise RuntimeError("Epoch sample order differs between comparison arms")
            paired.append({"variant": variant, "seed": seed, "B0_epoch_seconds": old_sec,
                "B0_DS_epoch_seconds": new_sec, "epoch_time_ratio_B0_over_B0_DS": old_sec / new_sec,
                "common_core_initialization_verified": True, "epoch_sample_order_verified": True,
                "timing_scope": "training phase including periodic recovery checkpoints + validation; epoch-end checkpoint/bootstrap/test excluded",
                "metric_delta_B0_DS_minus_B0": {p: {k: v - old["metrics"][p]["mean"][k]
                    for k, v in b["mean"].items()} for p, b in new["metrics"].items()}})
    write_json(OUT / "paired_comparison.json", paired)

report_lines = ["# Kaggle SSA-MRN B0/B0_DS", "",
    "[수치](#results) · [그래프](#graphs) · [이미지](#images) · [가중치](#evidence)", "",
    "| 비교 기준 | 바꾼 점 | 관측 차이 | 판단 |", "|---|---|---|---|",
    "| same-session B0 k4 | GT-only auxiliary supervision | see measured table | no guaranteed improvement |", "",
    "## 1. 목적과 상태",
    "Status: full training and RR/FR evaluation completed.",
    f"Learning rate: fixed {LEARNING_RATE} for {HOLD_LR_EPOCHS} epochs, then cosine to {MIN_LEARNING_RATE}.",
    "Fresh paired experiment; earlier B0/B0_DS scores are context, not a same-session baseline.",
    "## 2. 변경 사항과 평가 조건",
    f"QB; K={K}; epochs={EPOCHS}; seeds={SEEDS}; effective batch={EFFECTIVE_BATCH_SIZE}; micro={LEGACY_MICRO_BATCH_SIZE}.",
    "B0 is retrained in this suite with the original core and zero auxiliary weights.",
    "B0: original single core + final MSE. B0_DS: same core/heads + small quarter/half/full GT L1 supervision.",
    "Same core/head initialization, epoch sample order and final MSE; only auxiliary loss weights differ.",
    "GT-only binomial targets with explicit endpoint coordinates; observed LRMS is not wavelet LL. Backbone resampling stays unchanged.",
    "Inference is FP32 on whole scenes. Best selected only by validation MSE. Auxiliary targets are synthetic, not a calibrated sensor model.",
    "FR has no HR GT; no auxiliary GT target is used in FR evaluation. Repository QNR/Q2n/SCC MATLAB parity limitations remain.",
    "GPU-resident FP32 data, GPU sampling/batching, periodic status synchronization; no CPU DataLoader.",
    "Mid-epoch recovery stores the permutation, batch cursor, optimizer/scaler/RNG and loss accumulator.",
    "Best checkpoint uses validation MSE only. best_first100.pt preserves the actual first-100 validation winner.",
    "Fused Adam/AMP/layout/micro batch are matched; historic B1/B2/k6 are not bitwise reproduced.",
    '<a id="results"></a>', "## 3. 정량 결과",
    "| Run | Best epoch | RR PSNR | RR SAM | FR QNR (provisional) |", "|---|---:|---:|---:|---:|"]
for run_id, r in results.items():
    report_lines.append(f"| {run_id} | {r['best_epoch']} | {r['metrics']['RR']['mean']['PSNR']:.6f} "
                        f"| {r['metrics']['RR']['mean']['SAM']:.6f} | {r['metrics']['FR']['mean']['QNR']:.6f} |")
report_lines.extend(["", "## 4. 직전 연구와 수치 차이",
    "Same-session paired deltas: summary.csv and paired_comparison.json. Historical k6/B2 architectures are not directly comparable.",
    '<a id="graphs"></a>', "## 5. 그래프"])
for seed in SEEDS:
    report_lines.extend(["", f"![Learning](learning_seed{seed}.png)",
                         f"![Test comparison](test_comparison_seed{seed}.png)", f"![Auxiliary objectives](auxiliary_seed{seed}.png)"])
report_lines.extend(['<a id="images"></a>', "## 6. 결과 이미지 예시",
    "Five RR full scenes fixed in selection.json before training/test scoring."])
for run_id, result in results.items():
    report_lines.append(f"\n## {run_id} fixed RR scenes")
    for item in result["images"]:
        report_lines.append(f"![Scene {item['scene_index']}](runs/{run_id}/{item['file']})")
report_lines.extend(['<a id="evidence"></a>', "## 7. 가중치와 검증 근거",
    "Original best/latest epoch, path and SHA256 in each run/results.json; all retained file hashes in artifact_manifest.json.",
    "Checkpoint originals are Saved Output artifacts; no repository commit is implied.",
    "## 8. 한계와 다음 판단", "QB only, one seed, provisional FR/Q2n metrics. Auxiliary loss effectiveness is measured, not guaranteed; no sensor-MTF claim."])
(OUT / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

for cfg, _ in configs:
    run = Path(cfg["run_dir"])
    required = ["best.pt","latest.pt","history.json","initialization.json","results.json",
                "evaluation_complete.json","RR_per_scene.json","FR_per_scene.json"]
    required += [f"RR_scene_{i:02}.png" for i in selection["zero_based_scene_indices"]]
    if any(not (run/name).is_file() for name in required): raise RuntimeError(f"Incomplete artifacts: {run}")
    r = results[run.name]
    for kind in ("best","latest"):
        if sha(run/f"{kind}.pt") != r["checkpoints"][kind]["sha256"]:
            raise RuntimeError("Checkpoint changed after evaluation")
    if len(json.loads((run/"history.json").read_text())) != EPOCHS:
        raise RuntimeError("Incomplete training history")
    for protocol in ("RR","FR"):
        if len(json.loads((run/f"{protocol}_per_scene.json").read_text())) != data_info[protocol]["shapes"]["pan"][0]:
            raise RuntimeError("Incomplete scene scores")
artifacts = {}
for path in sorted(OUT.rglob("*")):
    relative = path.relative_to(OUT)
    if (not path.is_file() or relative.parts[0].startswith("_")
            or path.name in ("artifact_manifest.json", "worker.log", "status.json", "failure.json")
            or path.suffix in (".tmp", ".pyc") or "__pycache__" in path.parts):
        continue
    artifacts[str(relative)] = {"bytes": path.stat().st_size, "sha256": sha(path)}
write_json(OUT / "artifact_manifest.json", {"artifacts": artifacts,
    "suite_wall_seconds_before_packaging": time.monotonic() - suite_started,
    "excluded": "normalized cache and transient status/logs; source H5 data is external and hashed in data_manifest.json",
    "weights": "actual full best/latest with optimizer, scaler, RNG and history; no synthetic weights"})
archive = OUT.parent / (OUT.name + "_results.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
    for relative in [*artifacts, "artifact_manifest.json"]:
        bundle.write(OUT / relative, arcname=relative)
with zipfile.ZipFile(archive) as bundle:
    if bundle.testzip() is not None: raise RuntimeError("Archive CRC verification failed")
write_json(OUT.parent / (archive.name + ".sha256.json"), {"file": archive.name, "sha256": sha(archive)})
print(f"COMPLETE: {len(results)} trained/evaluated models; {len(artifacts)} hashed artifacts")
print(f"Results: {OUT}\nArchive: {archive}")
display(FileLink(str(archive)))
