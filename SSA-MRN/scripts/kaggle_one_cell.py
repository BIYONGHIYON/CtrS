# Run this entire file as ONE Kaggle notebook cell (GPU + Internet enabled).
# The optional path overrides are the only settings normally needing edits.
SENSOR = "QB"
K = 6
EPOCHS = 100
SEEDS = [42]
VARIANTS = ["baseline", "band_gated_hf"]
EFFECTIVE_BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LEGACY_MICRO_BATCH_SIZE = 32  # Replace with MICRO_BATCH_SIZE from the pasted settings.
TRAINING_MODES = ["github", "pasted"]
PASTED_MODEL_SOURCE = None  # Complete imports/classes/build_model/train_config, as a Python source string.
SWAP_GPUS = False  # Swap the two physical GPUs for a repeat in a new output directory.
CONTROLLED_COMPARISON = True  # Same micro batch in both arms; False benchmarks it per model.
AMP = True
USE_ALL_GPUS = True
GPU_CACHE = True
CHANNELS_LAST = True
NUM_WORKERS = 2
SAVE_PREDICTIONS = True
TRAIN_H5 = None  # e.g. "/kaggle/input/pancollection/train_qb.h5"
VAL_H5 = None
RR_H5 = None
FR_H5 = None
OUTPUT_DIR = "/kaggle/working/ssa_kaggle_model_comparison_v1"
REPO_COMMIT = "be02b7cbf03fe00681194ae9fc9419661f24cd56"
UPSTREAM_COMMIT = "a4ca40e407b12bf4c30f804384405ce321d11c51"
PLAN_COMMIT = "1b56efb54e9c200d5b497e719bb0857b25f750bf"

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
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
import h5py
import numpy as np
import torch
from IPython.display import clear_output, display, FileLink

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required. Enable a Kaggle GPU accelerator.")
suite_started = time.monotonic()
if SENSOR not in ("QB", "GF2", "WV3") or "baseline" not in VARIANTS:
    raise ValueError("Use QB/GF2/WV3 and include baseline for comparisons.")
if len(set(VARIANTS)) != len(VARIANTS) or len(set(SEEDS)) != len(SEEDS):
    raise ValueError("Duplicate run specifications.")
if EPOCHS < 1 or EFFECTIVE_BATCH_SIZE < 1 or LEARNING_RATE <= 0:
    raise ValueError("Invalid training settings.")
if LEGACY_MICRO_BATCH_SIZE < 1 or set(TRAINING_MODES) - {"github", "pasted"}:
    raise ValueError("Invalid micro batch or training mode.")
if TRAINING_MODES != ["github", "pasted"]:
    raise ValueError("This comparison cell requires TRAINING_MODES=['github', 'pasted'].")
if set(VARIANTS) != {"baseline", "band_gated_hf"}:
    raise ValueError("This requested comparison uses baseline and band_gated_hf only.")
SCALE = {"QB": 2047., "GF2": 1023., "WV3": 2047.}[SENSOR]
OUT = Path(OUTPUT_DIR)
OUT.mkdir(parents=True, exist_ok=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def locate(override, filename):
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    candidates = sorted(
        path for path in Path("/kaggle/input").rglob("*")
        if path.is_file()
        and path.name.rsplit("__", 1)[-1].lower() == filename.lower()
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Set the path override for {filename}; found {candidates}")
    return candidates[0].resolve()


stem = SENSOR.lower()
PATHS = {"train": locate(TRAIN_H5, f"train_{stem}.h5"),
         "val": locate(VAL_H5, f"valid_{stem}.h5"),
         "RR": locate(RR_H5, f"test_{stem}_multiExm1.h5"),
         "FR": locate(FR_H5, f"test_{stem}_OrigScale_multiExm1.h5")}
if len(set(PATHS.values())) != 4:
    raise ValueError("Train, validation, RR and FR must be distinct files.")

# Fetch just the pinned source files, not the repository's large checkpoints.
SOURCE = OUT / "code" / "CtrS"
custom_path = OUT / "code" / "pasted_models.py"
custom_hash = None
if PASTED_MODEL_SOURCE is None:
    plan_path = OUT / "code" / f"plan_{PLAN_COMMIT}.ipynb"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_url = (f"https://raw.githubusercontent.com/BIYONGHIYON/CtrS/{PLAN_COMMIT}/"
                "SSA-MRN/scripts/kaggle_band_gated_hf.ipynb")
    if not plan_path.exists():
        plan_temporary = plan_path.with_suffix(".download")
        urllib.request.urlretrieve(plan_url, plan_temporary)
        os.replace(plan_temporary, plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    definitions = {}
    wanted = {"FixedHighFrequency", "BandGatedHighFrequency", "build_model", "train_config"}
    for cell in plan["cells"]:
        if cell["cell_type"] != "code": continue
        source = "".join(cell["source"])
        for node in ast.parse(source).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
                definitions[node.name] = ast.get_source_segment(source, node)
    if definitions.keys() != wanted:
        raise RuntimeError("Missing model/config definitions in the pinned plan notebook")
    PASTED_MODEL_SOURCE = ("import torch\nfrom torch import nn\nfrom torch.nn import functional as F\n"
        "from ssamrn.models.ssa_mrn import RestoredPansharpeningNet\n\n" +
        "\n\n".join(definitions[name] for name in
                    ("FixedHighFrequency", "BandGatedHighFrequency", "build_model", "train_config")) + "\n")
if "pasted" in TRAINING_MODES:
    custom_hash = hashlib.sha256(PASTED_MODEL_SOURCE.encode()).hexdigest()
    custom_path.parent.mkdir(parents=True, exist_ok=True)
    if custom_path.exists() and sha(custom_path) != custom_hash:
        raise ValueError("Pasted model code changed; choose a new OUTPUT_DIR.")
    custom_path.write_text(PASTED_MODEL_SOURCE, encoding="utf-8")
files = ["src/ssamrn/__init__.py", "src/ssamrn/models/__init__.py",
         "src/ssamrn/data/__init__.py", "src/ssamrn/data/pancollection.py",
         "src/ssamrn/models/ssa_mrn.py", "src/ssamrn/models/pan_variants.py",
         "src/ssamrn/models/interp23.py", "src/ssamrn/observation.py",
         "src/ssamrn/metrics.py"]
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
        urllib.request.urlretrieve(url, temporary)
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
        n, c, h, w = shapes["lms"]
        if (n < (5 if split == "RR" else 1) or c != (8 if SENSOR == "WV3" else 4)
                or h % 4 or w % 4 or shapes["pan"] != [n, 1, h, w]
                or shapes["ms"] != [n, c, h // 4, w // 4]
                or (split != "FR" and shapes["gt"] != shapes["lms"])):
            raise ValueError(f"Invalid NCHW arrays in {path}: {shapes}")
        if split in ("RR", "FR") and (h % 32 or w % 32):
            raise ValueError("Repository metrics require 32-pixel aligned full scenes.")
    data_info[split] = {"path": str(path), "bytes": path.stat().st_size,
                        "sha256": sha(path), "shapes": shapes}
if len({info["sha256"] for info in data_info.values()}) != 4:
    raise ValueError("Identical files found in different splits.")
write_json(OUT / "data_manifest.json", data_info)
print(json.dumps(data_info, indent=2), flush=True)
write_json(OUT / "plan_deviations.json", {
    "variants": "baseline and band_gated_hf only, per user request",
    "reference_precision": "FP32", "actual_training_precision": "AMP FP16" if AMP else "FP32",
    "reference_micro_batch": 4, "actual_micro_batch": min(LEGACY_MICRO_BATCH_SIZE, EFFECTIVE_BATCH_SIZE),
    "test_scene_counts": {s: {"planned": 20, "actual": data_info[s]["shapes"]["pan"][0]}
                          for s in ("RR", "FR")}})

selection = {"seed": 42, "protocol": "RR", "tiles": "full scene",
             "zero_based_scene_indices": sorted(np.random.default_rng(42).choice(
                 data_info["RR"]["shapes"]["pan"][0], 5, replace=False).tolist()),
             "order": "ascending scene index", "rgb_assumed_bands_zero_based": [2, 1, 0],
             "rule": "fixed before training and test scoring"}
selection_path = OUT / "selection.json"
if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
    raise ValueError("Existing example selection differs; choose a new OUTPUT_DIR.")
write_json(selection_path, selection)

# Normalize once in bounded chunks. Read-only mmap pages are shared by workers.
CACHE = OUT / "_cache"
CACHE.mkdir(exist_ok=True)
cache_spec = {s: {"sha256": data_info[s]["sha256"], "scale": SCALE,
                  "shapes": data_info[s]["shapes"]} for s in ("train", "val")}
cache_marker = CACHE / "ready.json"
if cache_marker.exists() and json.loads(cache_marker.read_text()) != cache_spec:
    raise ValueError("Cache data changed; choose a new OUTPUT_DIR.")
if not cache_marker.exists():
    need = sum(np.prod(shape) * 4 for s in cache_spec for shape in data_info[s]["shapes"].values())
    if shutil.disk_usage(OUT).free < need + 2 * 1024**3:
        raise RuntimeError("Insufficient disk for the normalized cache + output reserve.")
    for split in ("train", "val"):
        with h5py.File(PATHS[split], "r") as h5:
            for key, shape in data_info[split]["shapes"].items():
                print(f"Normalizing {split}/{key} once", flush=True)
                array = np.lib.format.open_memmap(CACHE / f"{split}_{key}.npy", mode="w+",
                                                  dtype="float32", shape=tuple(shape))
                for start in range(0, shape[0], 128):
                    chunk = np.asarray(h5[key][start:start + 128], dtype=np.float32)
                    if not np.isfinite(chunk).all():
                        raise ValueError(f"Nonfinite {split}/{key} at {start}")
                    array[start:start + len(chunk)] = chunk / SCALE
                array.flush()
                del array
    write_json(cache_marker, cache_spec)
for split in ("train", "val"):
    for key, shape in data_info[split]["shapes"].items():
        a = np.load(CACHE / f"{split}_{key}.npy", mmap_mode="r")
        if list(a.shape) != shape or a.dtype != np.float32:
            raise ValueError("Invalid cache array")
        del a

# Separate CUDA processes keep seeds and optimizer states independent on T4 x2.
WORKER = r'''
import gc, hashlib, json, math, os, random, sys, time, traceback
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
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
    if config["mode"] == "github":
        cls = RestoredPansharpeningNet if config["variant"] == "baseline" else HighFrequency
        return cls(config["channels"], config["k"])
    return PASTED.build_model(config["variant"], config["channels"], config["k"])

class BatchIndices(Sampler):
    def __init__(self, n, batch, generator, shuffle):
        self.n, self.batch, self.generator, self.shuffle = n, batch, generator, shuffle
    def __iter__(self):
        order = torch.randperm(self.n, generator=self.generator) if self.shuffle else torch.arange(self.n)
        for start in range(0, self.n, self.batch): yield order[start:start + self.batch].tolist()
    def __len__(self): return math.ceil(self.n / self.batch)

class MmapBatches(Dataset):
    def __init__(self, cache, split): self.cache, self.split, self.arrays = cache, split, None
    def __getitem__(self, indices):
        if self.arrays is None:
            self.arrays = {k: np.load(Path(self.cache) / f"{self.split}_{k}.npy", mmap_mode="r") for k in KEYS}
        return {k: torch.from_numpy(np.array(v[indices], copy=True)) for k, v in self.arrays.items()}
    def __len__(self): return C["data"][self.split]["shapes"]["gt"][0]

def load_gpu_cache():
    size = sum(np.prod(v) * 4 for s in ("train", "val") for v in C["data"][s]["shapes"].values())
    free = torch.cuda.mem_get_info()[0]
    if not C["gpu_cache"] or size > free * .45:
        return None
    result = {}
    try:
        for split in ("train", "val"):
            result[split] = {}
            for key in KEYS:
                a = np.load(Path(C["cache"]) / f"{split}_{key}.npy", mmap_mode="r")
                tensor = torch.empty(a.shape, dtype=torch.float32, device="cuda")
                result[split][key] = tensor
                for start in range(0, len(a), 128):
                    tensor[start:start + 128].copy_(torch.from_numpy(np.array(a[start:start + 128], copy=True)))
                del a
        return result
    except torch.cuda.OutOfMemoryError:
        result.clear()
        if "tensor" in locals(): del tensor
        gc.collect(); torch.cuda.empty_cache()
        return None

def batches(split, batch_size, generator=None, shuffle=False):
    n = C["data"][split]["shapes"]["gt"][0]
    if GPU_DATA is not None:
        order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
        order = order.cuda()
        for start in range(0, n, batch_size):
            indices = order[start:start + batch_size]
            yield {k: v.index_select(0, indices) for k, v in GPU_DATA[split].items()}
    else:
        loader = LOADERS[split]
        loader.sampler.generator = generator
        loader.sampler.shuffle = shuffle
        for b in loader:
            yield {k: v.cuda(non_blocking=True) for k, v in b.items()}

def layout(b):
    return {k: v.contiguous(memory_format=FORMAT) for k, v in b.items()}

def build(): return model_for(C).cuda().to(memory_format=FORMAT)

def tune_micro():
    probe = {k: torch.from_numpy(np.array(np.load(Path(C["cache"]) / f"train_{k}.npy",
            mmap_mode="r")[:C["batch_size"]], copy=True)).cuda() for k in KEYS}
    probe = layout(probe); n = len(probe["gt"])
    records = []
    candidates = ([min(n, C["micro_batch"])] if not C["auto_micro"] else
                  sorted(set(min(n, b) for b in [C["batch_size"], 16, 8, 4, 1]), reverse=True))
    for micro in candidates:
        model = optimizer = scaler = pred = loss = None
        try:
            seed_all(C["seed"]); model = build()
            optimizer = torch.optim.Adam(model.parameters(), lr=C["lr"], foreach=True)
            scaler = torch.amp.GradScaler("cuda", enabled=C["amp"])
            for repeat in range(3):
                torch.cuda.synchronize(); started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                for offset in range(0, n, micro):
                    end = min(offset + micro, n)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=C["amp"]):
                        pred = model(*(probe[k][offset:end] for k in INPUTS))
                        loss = F.mse_loss(pred.float(), probe["gt"][offset:end]) * ((end - offset) / n)
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
    if not records: raise RuntimeError("Configured micro batch does not fit. Disable GPU_CACHE, or use the same smaller micro batch in both modes and a new OUTPUT_DIR.")
    micro = min(records, key=lambda x: x["seconds_per_effective_batch"])["micro_batch"]
    write(RUN / "benchmark.json", {"candidates": records, "chosen_micro_batch": micro,
          "cache": "GPU" if GPU_DATA is not None else "mmap",
          "memory_format": str(FORMAT)})
    return micro

def status(phase, **values):
    write(RUN / "status.json", {"run_id": RUN.name, "phase": phase,
          "device": torch.cuda.get_device_name(0), "epoch_target": C["epochs"],
          "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
          "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
          "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
          "updated_unix": time.time(), **values})

@torch.inference_mode()
def validate(model):
    model.eval(); sse = torch.zeros((), device="cuda"); count = 0
    psnr = torch.zeros((), device="cuda"); sam = torch.zeros((), device="cuda")
    samples = 0
    for b in batches("val", C["eval_batch"]):
        b = layout(b)
        # FP32 validation makes best selection independent of AMP inference.
        pred = model(*(b[k] for k in INPUTS)).float(); gt = b["gt"]
        diff = (pred - gt).square(); sse += diff.sum(); count += gt.numel()
        band_mse = diff.mean((2, 3)).clamp_min(1e-30)
        psnr += (-10 * band_mse.log10()).mean(1).sum(); samples += len(gt)
        norms = pred.norm(dim=1) * gt.norm(dim=1); valid = norms > 0
        angles = torch.rad2deg(torch.acos(((pred * gt).sum(1) / norms.clamp_min(1e-30)).clamp(-1, 1)))
        valid_counts = valid.flatten(1).sum(1)
        if (valid_counts == 0).any().item(): raise RuntimeError("No valid validation SAM pixels")
        sam += ((angles * valid).flatten(1).sum(1) / valid_counts).sum()
    result = {"val_mse": (sse / count).item(), "val_psnr_peak1": (psnr / samples).item(),
              "val_sam_deg": (sam / samples).item()}
    if not all(math.isfinite(v) for v in result.values()):
        raise RuntimeError("Invalid validation metrics")
    return result

def train():
    status("benchmark")
    micro = tune_micro()
    seed_all(C["seed"]); gen = torch.Generator().manual_seed(C["seed"])
    model = build(); opt = torch.optim.Adam(model.parameters(), lr=C["lr"], foreach=True)
    core_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(("detail.", "gate.")): continue
        core_hash.update(name.encode()); core_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    write(RUN / "initialization.json", {"seed": C["seed"], "core_sha256": core_hash.hexdigest(),
          "excluded_candidate_layers": ["detail", "gate"]})
    scaler = torch.amp.GradScaler("cuda", enabled=C["amp"])
    start, best, best_epoch, history = 0, float("inf"), 0, []
    if (RUN / "latest.pt").exists():
        ck = torch.load(RUN / "latest.pt", map_location="cpu", weights_only=True)
        if ck["config"] != C: raise ValueError("Resume configuration differs; use a new output directory")
        model.load_state_dict(ck["model"], strict=True); opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"]); torch.set_rng_state(ck["cpu_rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"]); gen.set_state(ck["loader_rng"])
        start, best, best_epoch, history = ck["epoch"], ck["best_mse"], ck["best_epoch"], ck["history"]
        del ck
        if not (RUN / "best.pt").exists(): raise RuntimeError("Missing best checkpoint; restore it before resuming")
    total_batches = math.ceil(C["data"]["train"]["shapes"]["gt"][0] / C["batch_size"])
    for epoch in range(start, C["epochs"]):
        order_generator = torch.Generator().set_state(gen.get_state())
        order_hash = hashlib.sha256(torch.randperm(C["data"]["train"]["shapes"]["gt"][0],
                     generator=order_generator).numpy().tobytes()).hexdigest()
        model.train(); sse = torch.zeros((), device="cuda"); count = 0
        torch.cuda.synchronize(); tick = time.monotonic(); last = tick
        status("train", epoch=epoch + 1, batch=0, batches=total_batches)
        for step, b in enumerate(batches("train", C["batch_size"], gen, True), 1):
            b = layout(b); n = len(b["gt"]); opt.zero_grad(set_to_none=True)
            for offset in range(0, n, micro):
                end = min(offset + micro, n)
                with torch.autocast("cuda", dtype=torch.float16, enabled=C["amp"]):
                    pred = model(*(b[k][offset:end] for k in INPUTS))
                    loss = F.mse_loss(pred.float(), b["gt"][offset:end])
                scaler.scale(loss * ((end - offset) / n)).backward()
                sse += (pred.detach().float() - b["gt"][offset:end]).square().sum()
                count += b["gt"][offset:end].numel()
            scaler.step(opt); scaler.update()
            if time.monotonic() - last >= 10:
                torch.cuda.synchronize(); elapsed = time.monotonic() - tick
                prior = [r["seconds"] for r in history[-5:]]
                measured = elapsed / step * total_batches
                estimate = np.mean(prior) if prior else measured * 1.15
                status("train", epoch=epoch + 1, batch=step, batches=total_batches,
                       train_remaining_seconds=max(0, estimate - elapsed) + (C["epochs"] - epoch - 1) * estimate)
                last = time.monotonic()
        train_mse = (sse / count).item()
        if not math.isfinite(train_mse): raise RuntimeError("Nonfinite training loss")
        status("validation", epoch=epoch + 1)
        validation = validate(model)
        improved = validation["val_mse"] < best
        if improved: best, best_epoch = validation["val_mse"], epoch + 1
        torch.cuda.synchronize()
        row = {"epoch": epoch + 1, "train_mse": train_mse, **validation,
               "seconds": time.monotonic() - tick, "scaler_scale": scaler.get_scale(),
               "micro_batch": micro, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3}
        row["sample_order_sha256"] = order_hash
        history.append(row)
        status("checkpoint", epoch=epoch + 1, metrics=row)
        ck = {"config": C, "epoch": epoch + 1, "best_mse": best, "best_epoch": best_epoch,
              "model": cpu(model.state_dict()), "optimizer": cpu(opt.state_dict()),
              "scaler": scaler.state_dict(), "cpu_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state_all(), "loader_rng": gen.get_state(), "history": history}
        # Best first: latest is the committed recovery point. History lives in it.
        if improved: save(ck, RUN / "best.pt")
        save(ck, RUN / "latest.pt"); write(RUN / "history.json", history)
        del ck
        print(json.dumps(row), flush=True)
    write(RUN / "history.json", history)
    write(RUN / "training_complete.json", {"config": C, "epoch": C["epochs"],
          "best_epoch": best_epoch, "best_mse": best})
    del model, opt, scaler
    gc.collect(); torch.cuda.empty_cache()
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
                  "bytes": (RUN / f"{kind}.pt").stat().st_size} for kind in ("best", "latest")},
              "metrics": {}, "images": [], "test_precision": "FP32", "full_scene": True}
    del ck
    for protocol in ("RR", "FR"):
        rows = []; tick = time.monotonic()
        if C["save_predictions"]: (RUN / "predictions" / protocol).mkdir(parents=True, exist_ok=True)
        with h5py.File(C["data"][protocol]["path"], "r") as h5:
            keys = INPUTS + (("gt",) if protocol == "RR" else ())
            for index in range(len(h5["pan"])):
                status("test_" + protocol, scene=index + 1, scenes=len(h5["pan"]))
                arrays = {k: np.asarray(h5[k][index], dtype=np.float32) for k in keys}
                if any(not np.isfinite(v).all() for v in arrays.values()): raise ValueError("Nonfinite test input")
                inputs = [torch.from_numpy(arrays[k] / C["scale"]).unsqueeze(0).cuda().contiguous(
                          memory_format=FORMAT) for k in INPUTS]
                with torch.inference_mode(): pred = model(*inputs)[0].float().cpu().numpy() * C["scale"]
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
    if C["mode"] == "pasted":
        import importlib.util
        spec = importlib.util.spec_from_file_location("pasted_models", C["pasted_source"])
        PASTED = importlib.util.module_from_spec(spec)
        # The original setup's constants are available to its function bodies.
        PASTED.__dict__.update({"SENSOR": C["sensor"], "K": C["k"], "EPOCHS": C["epochs"],
            "EFFECTIVE_BATCH_SIZE": C["batch_size"], "MICRO_BATCH_SIZE": C["micro_batch"],
            "LEARNING_RATE": C["lr"], "OUTPUT_DIR": str(RUN.parent),
            "TRAIN_H5": C["data"]["train"]["path"], "VAL_H5": C["data"]["val"]["path"],
            "SOURCE_ROOT": str(Path(C["source"]) / "SSA-MRN/src")})
        spec.loader.exec_module(PASTED)
        C["reference_plan_config"] = PASTED.train_config(C["variant"], C["seed"])
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
        status("loading_cache")
        GPU_DATA = load_gpu_cache()
        if GPU_DATA is None:
            for split in ("train", "val"):
                batch = C["batch_size"] if split == "train" else C["eval_batch"]
                sampler = BatchIndices(C["data"][split]["shapes"]["gt"][0], batch, None, False)
                options = {"prefetch_factor": 2, "persistent_workers": True} if C["workers"] else {}
                LOADERS[split] = DataLoader(MmapBatches(C["cache"], split), batch_size=None,
                     sampler=sampler, num_workers=C["workers"], pin_memory=True,
                     generator=torch.Generator().manual_seed(C["seed"] + 100000), **options)
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
    print("Pair assignment: GPU 0=baseline; GPU 1=band_gated_hf (swap optional)" if gpu_count == 2 else
          "Only one GPU: paired modes will run sequentially, not concurrently.")
configs = []
for mode, seed, variant in [(m, s, v) for m in TRAINING_MODES for s in SEEDS
                           for v in (["baseline"] if m == "github" else ["band_gated_hf"])]:
        run_id = f"{mode}_{SENSOR}_{variant}_k{K}_s{seed}"
        config = {"sensor": SENSOR, "k": K, "channels": 8 if SENSOR == "WV3" else 4,
                  "mode": mode, "controlled": CONTROLLED_COMPARISON,
                  "epochs": EPOCHS, "seed": seed, "variant": variant,
                  "batch_size": EFFECTIVE_BATCH_SIZE, "lr": LEARNING_RATE,
                  "micro_batch": min(LEGACY_MICRO_BATCH_SIZE, EFFECTIVE_BATCH_SIZE),
                  "auto_micro": not CONTROLLED_COMPARISON,
                  "eval_batch": EFFECTIVE_BATCH_SIZE,
                  "amp": AMP,
                  "channels_last": CHANNELS_LAST,
                  "gpu_cache": GPU_CACHE, "workers": NUM_WORKERS, "scale": SCALE,
                  "save_predictions": SAVE_PREDICTIONS, "data": data_info,
                  "cache": str(CACHE), "source": str(SOURCE), "run_dir": str(OUT / "runs" / run_id),
                  "selection": selection, "repo_commit": REPO_COMMIT,
                  "plan_commit": PLAN_COMMIT,
                  "pasted_source": str(custom_path), "pasted_source_sha256": custom_hash,
                  "upstream_commit": UPSTREAM_COMMIT, "worker_sha256": worker_hash,
                  "model_definition": "Pinned GitHub baseline versus exact BandGatedHighFrequency from the linked plan notebook; fixed high_frequency excluded by request",
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
           "old_checkpoints": "not imported; resume only this output directory's matching config"})

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
                assigned_mode = "github" if device == reference_device else "pasted"
                eligible = next((i for i, (c, p) in enumerate(pending) if c["mode"] == assigned_mode), None)
                if eligible is None: continue
            else: eligible = 0
            config, cfg_path = pending.pop(eligible)
            run = Path(config["run_dir"]); run.mkdir(parents=True, exist_ok=True)
            status_path = run / "status.json"
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
        print("=== CtrS Kaggle GPU training + RR/FR evaluation ===")
        print(f"Completed {len(finished)}/{len(configs)} | waiting {len(pending)} | GPUs {gpu_count}")
        try:
            gpu_status = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader"], capture_output=True, text=True, timeout=3)
            if gpu_status.returncode == 0: print(gpu_status.stdout.strip())
        except (OSError, subprocess.TimeoutExpired): pass
        for device, (process, log, run) in list(active.items()):
            state = json.loads((run / "status.json").read_text())
            print(f"GPU {device}: {run.name} | {state.get('phase')} | epoch {state.get('epoch', '-')}/{EPOCHS} "
                  f"| batch {state.get('batch', '-')}/{state.get('batches', '-')} "
                  f"| allocated {state.get('allocated_gb', 0):.2f} GB | peak {state.get('peak_gb', 0):.2f} GB")
            if "train_remaining_seconds" in state:
                print(f"  This run training ETA: {state['train_remaining_seconds'] / 60:.1f} min; evaluation excluded")
            if "scene" in state:
                print(f"  Full test scenes: {state['scene']}/{state['scenes']}")
            rc = process.poll()
            if rc is not None:
                log.close(); del active[device]
                if rc:
                    failure = run / "failure.json"
                    detail = failure.read_text() if failure.exists() else (run / "worker.log").read_text()[-6000:]
                    raise RuntimeError(f"Worker failed: {run}\n{detail}")
                finished.append(run)
        print("A total ETA is not asserted before each variant and evaluation speed are measured.")
        time.sleep(5)
finally:
    # Stopping this cell also stops its child training processes.
    for process, log, run in active.values():
        process.terminate()
    for process, log, run in active.values():
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()

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
    base = results[f"github_{SENSOR}_baseline_k{K}_s{seed}"]["metrics"]
    result["delta_from_same_seed_baseline"] = {
        protocol: {key: val - base[protocol]["mean"][key] for key, val in block["mean"].items()}
        for protocol, block in result["metrics"].items()}
    for protocol, block in result["metrics"].items():
        for key, val in block["mean"].items():
            summary_rows.append({"run_id": run_id, "mode": mode, "seed": seed, "best_epoch": result["best_epoch"],
                "protocol": protocol, "metric": key, "mean": val,
                "delta_from_baseline": result["delta_from_same_seed_baseline"][protocol][key]})
write_json(OUT / "results.json", results)
with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
    writer.writeheader(); writer.writerows(summary_rows)

for seed in SEEDS:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for mode, variant in [("github", "baseline"), ("pasted", "band_gated_hf")]:
        run_id = f"{mode}_{SENSOR}_{variant}_k{K}_s{seed}"
        history = json.loads((OUT / "runs" / run_id / "history.json").read_text())
        for ax, key in zip(axes, ["train_mse", "val_mse", "val_psnr_peak1", "val_sam_deg"]):
            ax.plot([r["epoch"] for r in history], [r[key] for r in history], label=variant)
            ax.set_title(key); ax.set_xlabel("Epoch"); ax.grid(alpha=.25)
    axes[0].set_yscale("log"); axes[1].set_yscale("log"); axes[-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"learning_seed{seed}.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (protocol, key) in zip(axes.flat, [("RR", "PSNR"), ("RR", "SAM"),
            ("RR", "MSE_peak1"), ("RR", "ERGAS"), ("RR", "Q2n"), ("FR", "QNR")]):
        value = results[f"pasted_{SENSOR}_band_gated_hf_k{K}_s{seed}"]["delta_from_same_seed_baseline"][protocol][key]
        ax.bar(["band_gated_hf"], [value], color="#2788b5")
        ax.set_title(f"{protocol} {key}: variant minus baseline" + (" (provisional)" if protocol == "FR" else ""))
        ax.axhline(0, color="black", linewidth=.8); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(OUT / f"test_comparison_seed{seed}.png", dpi=150); plt.close(fig)

paired = []
if set(TRAINING_MODES) == {"github", "pasted"}:
    for seed in SEEDS:
        for variant in ["band_gated_hf"]:
            old_id = f"github_{SENSOR}_baseline_k{K}_s{seed}"
            new_id = f"pasted_{SENSOR}_{variant}_k{K}_s{seed}"
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
            paired.append({"variant": variant, "seed": seed, "github_epoch_seconds": old_sec,
                "pasted_epoch_seconds": new_sec, "epoch_time_ratio_github_over_pasted": old_sec / new_sec,
                "common_core_initialization_verified": True, "epoch_sample_order_verified": True,
                "timing_scope": "training + validation; checkpoint/bootstrap/test excluded",
                "metric_delta_pasted_minus_github": {p: {k: v - old["metrics"][p]["mean"][k]
                    for k, v in b["mean"].items()} for p, b in new["metrics"].items()}})
    write_json(OUT / "paired_comparison.json", paired)

report_lines = ["# Kaggle SSA-MRN experiment", "", "Status: full training and RR/FR evaluation completed.",
    f"Sensor={SENSOR}; K={K}; epochs={EPOCHS}; seeds={SEEDS}; Adam LR={LEARNING_RATE}; effective batch={EFFECTIVE_BATCH_SIZE}.",
    "", "Best selected only by FP32 validation MSE. All test scenes evaluated in FP32 without inference cropping.",
    "GitHub baseline uses pinned RestoredPansharpeningNet; pasted arm uses build_model source recorded under code/.",
    f"Band-gated definition extracted unchanged from plan notebook commit {PLAN_COMMIT}.",
    "Plan model logic is retained; shared training AMP/micro-batch settings differ from the plan's FP32/micro4 protocol.",
    "Fixed high_frequency comparator excluded by user request; this study compares baseline versus band_gated_hf only.",
    f"Controlled comparison={CONTROLLED_COMPARISON}. AMP={AMP}; channels_last={CHANNELS_LAST}; equal micro batches in controlled mode.",
    "Execution/data loading optimizations apply equally to both model arms. Model definitions are not inferred.",
    "reference_plan_config preserves the original source's declared FP32 config; amp/micro_batch fields record actual execution.",
    "FR D_s/QNR, RR Q2n and SCC retain the repository's MATLAB parity limitations.",
    "Repository PSNR uses peak 2047 even for GF2; PSNR_sensor_peak is exported separately.",
    "Example scene indices were fixed before scoring. RGB band order [2,1,0] is assumed.",
    "FR has no GT. No FR ground truth images are generated.", "",
    f"Actual RR scenes={data_info['RR']['shapes']['pan'][0]}, FR scenes={data_info['FR']['shapes']['pan'][0]}; reference plan expects 20 each.",
    "| Run | Best epoch | RR PSNR | RR SAM | FR QNR (provisional) |", "|---|---:|---:|---:|---:|"]
for run_id, r in results.items():
    report_lines.append(f"| {run_id} | {r['best_epoch']} | {r['metrics']['RR']['mean']['PSNR']:.6f} "
                        f"| {r['metrics']['RR']['mean']['SAM']:.6f} | {r['metrics']['FR']['mean']['QNR']:.6f} |")
for seed in SEEDS:
    report_lines.extend(["", f"![Learning](learning_seed{seed}.png)",
                         f"![Test comparison](test_comparison_seed{seed}.png)"])
for run_id, result in results.items():
    report_lines.append(f"\n## {run_id} fixed RR scenes")
    for item in result["images"]:
        report_lines.append(f"![Scene {item['scene_index']}](runs/{run_id}/{item['file']})")
(OUT / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

artifacts = {}
for path in sorted(OUT.rglob("*")):
    relative = path.relative_to(OUT)
    if (not path.is_file() or relative.parts[0].startswith("_")
            or path.name in ("artifact_manifest.json", "worker.log", "status.json", "failure.json")
            or path.suffix == ".tmp"):
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
