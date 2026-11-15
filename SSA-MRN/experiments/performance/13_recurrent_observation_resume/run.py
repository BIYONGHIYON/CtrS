# Run this entire file as ONE Kaggle notebook cell (GPU + Internet enabled).
# The optional path overrides are the only settings normally needing edits.
SENSOR = "QB"
K = 6
EPOCHS = 100  # Hard limit; never train epoch 101.
REQUIRE_RESUME = False  # Set True after interruption: missing checkpoints then STOP, never restart from epoch 1.
STATUS_INTERVAL_SECONDS = 15
CHECKPOINT_INTERVAL_SECONDS = 60  # Recovery point at least once a minute and every completed epoch.
FUSED_ADAM = True  # Same optimizer implementation in both arms; GPU AMP overflow handling.
USE_FFT_OBSERVATION = True  # Same FIR/phase/boundary; GPU checks forward AND backward first.
RESUME_INPUT = ""  # Suite directory in Saved Output; auto-find if exactly one matching suite exists.
# Before the first run: Session options -> Persistence -> Files only.
# After a session loss: attach the Saved Output and set REQUIRE_RESUME=True.
# This cell cannot recover /kaggle/working files that Kaggle has already discarded.
# Compatible checkpoints from the previous 200-epoch cell may be imported up to epoch 100.
SEEDS = [42]
VARIANTS = ["B1", "B2"]
EFFECTIVE_BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LEGACY_MICRO_BATCH_SIZE = 32  # One forward/backward per effective batch; equal in B1/B2.
# If either arm fails the preflight with OOM, set this to 16 in BOTH arms.
# Different micro batches change floating-point accumulation, although batch/loss remain equal.
TRAINING_MODES = ["B1", "B2"]

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
OUTPUT_DIR = "/kaggle/working/ssa_kaggle_B1_B2_100_resume_v1"
REPO_COMMIT = "be02b7cbf03fe00681194ae9fc9419661f24cd56"
UPSTREAM_COMMIT = "a4ca40e407b12bf4c30f804384405ce321d11c51"
PLAN_COMMIT = None  # Architecture is defined directly in this cell.

print("[START] B1/B2: 100 epochs ONLY, fixed LR, GPU-resident pair, batch-level resume", flush=True)
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
if EPOCHS != 100:
    raise ValueError("This cell is fixed to 100 epochs; use a different experiment for other budgets")
if not (STATUS_INTERVAL_SECONDS >= 5 and CHECKPOINT_INTERVAL_SECONDS >= 60):
    raise ValueError("Progress/checkpoint intervals are too small")
CELL_ID = "ssa_b1_b2_100_resume_v1"
suite_settings = {"cell_id": CELL_ID, "repo_commit": REPO_COMMIT,
    "upstream_commit": UPSTREAM_COMMIT, "sensor": SENSOR, "k": K, "epochs": EPOCHS,
    "seeds": SEEDS, "variants": VARIANTS, "batch": EFFECTIVE_BATCH_SIZE,
    "micro": LEGACY_MICRO_BATCH_SIZE, "lr": LEARNING_RATE, "schedule": "constant",
    "amp": AMP, "channels_last": CHANNELS_LAST,
    "fused_adam": FUSED_ADAM, "fft_observation": USE_FFT_OBSERVATION,
    "controlled": CONTROLLED_COMPARISON, "save_predictions": SAVE_PREDICTIONS,
    "status_seconds": STATUS_INTERVAL_SECONDS, "checkpoint_seconds": CHECKPOINT_INTERVAL_SECONDS}
if SENSOR != "QB":
    raise ValueError("This audited first experiment is QB only.")
if VARIANTS != ["B1", "B2"] or TRAINING_MODES != ["B1", "B2"]:
    raise ValueError("B1 and B2 only; B0 is excluded.")
if len(set(SEEDS)) != len(SEEDS) or EPOCHS < 1 or EFFECTIVE_BATCH_SIZE < 1 or LEARNING_RATE <= 0:
    raise ValueError("Invalid run settings")
if not GPU_CACHE or not CONTROLLED_COMPARISON or LEGACY_MICRO_BATCH_SIZE < 1:
    raise ValueError("GPU residency and a shared fixed micro batch are required")
if not USE_ALL_GPUS:
    raise ValueError("USE_ALL_GPUS must remain True for parallel B1/B2")
if torch.cuda.device_count() < 2:
    raise RuntimeError("Select Kaggle T4 x2: B1 and B2 require two GPUs for parallel execution")
SCALE = {"QB": 2047., "GF2": 1023., "WV3": 2047.}[SENSOR]
OUT = Path(OUTPUT_DIR)
import psutil
owned_workers = {str(OUT / "code" / "kaggle_worker.py"),
    "/kaggle/working/ssa_kaggle_B1_B2_200_cosine_v1/code/kaggle_worker.py"}
for process_info in psutil.process_iter(["pid", "cmdline"]):
    try:
        if any(arg in owned_workers for arg in (process_info.info["cmdline"] or [])):
            raise RuntimeError(f"An existing B1/B2 worker is still running (PID {process_info.pid}). Interrupt its original cell and preserve Output before rerunning; no duplicate training was started.")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
# Restore a matching 100-epoch suite, or identify an audited 200-cell import.
LEGACY_BOOTSTRAP = None
legacy_settings = {**suite_settings, "cell_id": "ssa_b1_b2_200_cosine_v1",
                   "epochs": 200, "hold": 100, "min_lr": 1e-6, "checkpoint_seconds": 900}
legacy_settings.pop("schedule", None)
if not (OUT / "suite_config.json").exists():
    candidates = ([Path(RESUME_INPUT) / "suite_config.json"] if RESUME_INPUT else
                  sorted(Path("/kaggle/input").rglob("suite_config.json")) +
                  sorted(Path("/kaggle/working").glob("ssa_kaggle_B1_B2_200_cosine_v1/suite_config.json")))
    exact, legacy = [], []
    for candidate in candidates:
        if candidate.is_file():
            try:
                settings = json.loads(candidate.read_text())
                if settings == suite_settings:
                    exact.append(candidate.parent)
                elif settings == legacy_settings:
                    legacy.append(candidate.parent)
            except (OSError, ValueError):
                pass
    matches = list(dict.fromkeys(exact or legacy))
    if RESUME_INPUT and len(matches) != 1:
        raise ValueError("RESUME_INPUT is not a compatible suite directory")
    if len(matches) > 1:
        raise ValueError(f"Multiple recovery sources; set RESUME_INPUT to the intended suite: {matches}")
    if matches:
        if OUT.exists() and any(OUT.iterdir()):
            raise ValueError("Nonempty output without suite_config.json; do not overwrite it")
        if matches[0] in exact:
            print(f"[RESTORE 100] {matches[0]} -> {OUT}", flush=True)
            shutil.copytree(matches[0], OUT, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.tmp", "*.pyc", "__pycache__"))
        else:
            LEGACY_BOOTSTRAP = matches[0]
            print(f"[IMPORT 200 -> 100] Checking {LEGACY_BOOTSTRAP}; old files remain untouched", flush=True)
if (OUT / "suite_config.json").exists():
    if json.loads((OUT / "suite_config.json").read_text()) != suite_settings:
        raise ValueError("Experiment settings changed; choose a new OUTPUT_DIR")

if LEGACY_BOOTSTRAP is None and (OUT / "bootstrap_source.json").is_file():
    incomplete = any(not any((OUT / "runs" / f"{mode}_{SENSOR}_{mode}_k{K}_s{seed}" / name).is_file()
                            for name in ("latest.pt", "latest.prev.pt"))
                     for seed in SEEDS for mode in TRAINING_MODES)
    if incomplete:
        saved = Path(RESUME_INPUT or json.loads((OUT / "bootstrap_source.json").read_text())["path"])
        if not saved.is_dir():
            raise FileNotFoundError("Previous import incomplete; reattach the original Saved Output. No fresh run was started.")
        LEGACY_BOOTSTRAP = saved

# Read-only preflight checks before hashing/uploading several GB of data.
for seed in SEEDS:
    for mode in TRAINING_MODES:
        run_id = f"{mode}_{SENSOR}_{mode}_k{K}_s{seed}"
        local = OUT / "runs" / run_id
        source = LEGACY_BOOTSTRAP / "runs" / run_id if LEGACY_BOOTSTRAP else local
        has_state = any((source / name).is_file() for name in ("latest.pt", "latest.prev.pt", "best_first100.pt"))
        if REQUIRE_RESUME and not has_state:
            raise FileNotFoundError(f"RESUME REQUIRED: no checkpoint for {run_id}. Attach the Saved Output; training was NOT started.")
        if (source / "initialization.json").exists() and not has_state:
            raise FileNotFoundError(f"Run started previously but checkpoints are missing: {source}; refusing to silently restart")
OUT.mkdir(parents=True, exist_ok=True)
if LEGACY_BOOTSTRAP:
    (OUT / "bootstrap_source.json").write_text(json.dumps({"path": str(LEGACY_BOOTSTRAP)}), encoding="utf-8")
print("[FILES] Use Persistence=Files only and preserve Saved Output before ending a session.", flush=True)


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
custom_path = OUT / "code" / "B2_models.py"
custom_hash = None
PASTED_MODEL_SOURCE = r'''
import torch
from torch import nn
from torch.nn import functional as F
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
from ssamrn.observation import SensorObservation

class FFTObservation(SensorObservation):
    def __init__(self, sensor):
        super().__init__(sensor)
        self.fft_enabled = False
        self._geometry = {}

    def blur(self, prediction):
        if not self.fft_enabled:
            return super().blur(prediction)
        if prediction.ndim != 4 or prediction.shape[1] != self.kernel.shape[0]:
            raise ValueError("Expected sensor-band NCHW")
        h, w = prediction.shape[-2:]
        if h % 4 or w % 4:
            raise ValueError("Observation dimensions must be divisible by four")
        key = (h, w, prediction.device, prediction.dtype)
        if key not in self._geometry:
            from scipy.fft import next_fast_len
            kh, kw = self.kernel.shape[-2:]
            ph, pw = kh // 2, kw // 2
            rows = torch.arange(-ph, h + ph, device=prediction.device).clamp(0, h - 1)
            cols = torch.arange(-pw, w + pw, device=prediction.device).clamp(0, w - 1)
            shape = (next_fast_len(h + 2*ph + kh - 1), next_fast_len(w + 2*pw + kw - 1))
            # conv2d is correlation: reverse the FIR for FFT linear convolution.
            kernel = self.kernel[:, 0].to(dtype=prediction.dtype).flip((-2, -1))
            frequency = torch.fft.rfft2(kernel, s=shape)
            self._geometry[key] = (rows, cols, shape, frequency, kh, kw)
        rows, cols, shape, frequency, kh, kw = self._geometry[key]
        padded = prediction.index_select(-2, rows).index_select(-1, cols)
        full = torch.fft.irfft2(torch.fft.rfft2(padded, s=shape) * frequency, s=shape)
        return full[..., kh-1:kh-1+h, kw-1:kw-1+w]


class SharedThreeStage(nn.Module):
    def __init__(self, channels, k, feedback):
        super().__init__()
        # Instantiate the common core first for identical B1/B2 initialization.
        self.core = RestoredPansharpeningNet(channels, k)
        self.error_encoder = nn.Conv2d(channels, channels, 3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.observation = FFTObservation("QB").float()
        self.feedback = feedback
        self.margin = 5
        self._masks = {}
        self.register_buffer("phase_candidates", torch.tensor([[2,2],[1,2],[2,1]], dtype=torch.long))

    def samples(self, blurred):
        # Static phase slices keep all selection and indexing on the GPU.
        return torch.stack([blurred[...,2::4,2::4], blurred[...,1::4,2::4],
                            blurred[...,2::4,1::4]], dim=1)

    @torch.no_grad()
    def select_phase(self, lms, ms):
        # Inference geometry uses observed LMS/MS only; never test GT.
        with torch.autocast("cuda", enabled=False):
            views = self.samples(self.observation.blur(lms.float()))
        margin = self.margin if min(ms.shape[-2:]) > 2*self.margin else 0
        delta = views - ms.float()[:,None]
        if margin: delta = delta[...,margin:-margin,margin:-margin]
        return delta.square().mean((2,3,4)).argmin(1)

    def forward(self, pan, lms, ms, phase=None):
        current = lms
        if self.feedback and phase is None: phase = self.select_phase(lms, ms)
        for _ in range(3):
            if self.feedback:
                # Fixed FP32 sensor model, with gradients through current reconstruction.
                with torch.autocast("cuda", enabled=False):
                    views = self.samples(self.observation.blur(current.float()))
                    observed = views[torch.arange(len(ms),device=ms.device),phase]
                    error = ms.float() - observed
                    m = self.margin
                    geometry = (ms.shape[-2], ms.shape[-1], ms.device)
                    if geometry not in self._masks:
                        rows = torch.arange(ms.shape[-2],device=ms.device)
                        cols = torch.arange(ms.shape[-1],device=ms.device)
                        self._masks[geometry] = ((rows>=m)&(rows<ms.shape[-2]-m))[:,None] & ((cols>=m)&(cols<ms.shape[-1]-m))[None,:]
                    mask = self._masks[geometry]
                    error = F.interpolate(error*mask, size=current.shape[-2:],mode="bilinear",align_corners=False)
            else:
                error = torch.zeros_like(current)
            conditioned = current + self.error_encoder(error)
            proposal = self.core(pan, conditioned, ms)
            current = current + self.alpha * (proposal - current)
        return current

def build_model(variant, channels, k):
    if variant not in ("B1","B2"): raise ValueError(variant)
    return SharedThreeStage(channels,k,variant=="B2")

def train_config(variant, seed):
    return {"variant":variant,"seed":seed,"stages":3,"shared_weights":True,
            "loss":"final_output_MSE","initial_alpha":0.1,
            "train_phase":"audited original TRAIN GT/MS",
            "inference_phase":"observed LMS/MS, never GT"}
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
# All-pair audit uses GPU FP64 FFT convolution, matching the source FIR protocol.
# This is numerical verification, with no training or test-derived calibration.
audit_path = OUT / "observation_train.json"
sys.path.insert(0, str(SOURCE / "SSA-MRN/src"))
from ssamrn.observation import mtf_kernels, module_hash, SOURCE_COMMIT, GNYQ
audit_identity = {"train_sha256":data_info["train"]["sha256"],
                  "operator_sha256":module_hash(), "audit_version":"gpu_fft64_v1"}
audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
if not audit or audit.get("identity") != audit_identity or audit.get("status") != "validated":
    print("[AUDIT GPU] Verifying every TRAIN GT/MS pair on GPU 0", flush=True)
    kernels = torch.from_numpy(mtf_kernels(SENSOR)[:,0]).to(device="cuda:0",dtype=torch.float64)
    phases = []; errors_all = []; started = time.monotonic()
    with torch.inference_mode(), h5py.File(PATHS["train"],"r") as h5:
        n = len(h5["gt"]); height,width = h5["gt"].shape[-2:]; margin = 5
        if min(h5["ms"].shape[-2:]) <= 2*margin: raise ValueError("Audit interior too small")
        # padded input length H+40 plus kernel length 41 minus one = H+80
        fft_shape = (height+80,width+80)
        frequency_kernel = torch.fft.rfft2(kernels,s=fft_shape)
        for start in range(0,n,32):
            gt = torch.from_numpy(np.asarray(h5["gt"][start:start+32],dtype=np.float64)).to("cuda:0")
            ms = torch.from_numpy(np.asarray(h5["ms"][start:start+32],dtype=np.float64)).to("cuda:0")
            if not torch.isfinite(gt).all().item() or not torch.isfinite(ms).all().item():
                raise ValueError("Nonfinite original TRAIN pair")
            padded = torch.nn.functional.pad(gt,(20,20,20,20),mode="replicate")
            full = torch.fft.irfft2(torch.fft.rfft2(padded,s=fft_shape)*frequency_kernel,s=fft_shape)
            blurred = full[...,40:40+height,40:40+width]
            candidates = [(2,2),(1,2),(2,1)]
            target = ms[...,margin:-margin,margin:-margin]
            scores = torch.stack([(blurred[...,r::4,c::4][...,margin:-margin,margin:-margin]-target).square().mean((1,2,3))
                                  for r,c in candidates],1)
            values, ids = scores.min(1)
            phases.extend([list(candidates[i]) for i in ids.cpu().tolist()])
            errors_all.extend(values.cpu().tolist())
            if start % 256 == 0 or start+len(gt) == n:
                done=start+len(gt)
                print(f"[AUDIT GPU] {done}/{n} ({100*done/n:.1f}%) | {time.monotonic()-started:.0f}s",flush=True)
    rmse = float(np.mean(errors_all)**.5); worst = float(max(errors_all)**.5)
    audit = {"identity":audit_identity,"status":"validated" if rmse<=1.0 and worst<=2.0 else "blocked",
             "sensor":SENSOR,"coverage":n,"sample_count":n,"phases":phases,"lr_margin":5,
             "phase_policy":"fixed_per_training_sample","rmse_dn":rmse,"max_sample_rmse_dn":worst,
             "tolerance":{"rmse_dn":1.0,"max_sample_rmse_dn":2.0},"source_commit":SOURCE_COMMIT,
             "gnyq":GNYQ[SENSOR],"kernel":"DLPan_MATLAB_radial_fwind1",
             "implementation":"FP64 GPU FFT linear convolution; valid slice begins at kernel_size-1",
             "interpretation":"TRAIN interior only; no FR physics claim"}
    write_json(audit_path,audit)
    del kernels,frequency_kernel,gt,ms,padded,full,blurred,target,scores,values,ids
    torch.cuda.empty_cache()
else:
    print("[AUDIT] Reusing complete audit matched to TRAIN SHA256 and operator hash",flush=True)
if audit["status"] != "validated" or audit["coverage"] != data_info["train"]["shapes"]["gt"][0]:
    raise RuntimeError(f"Observation audit failed: RMSE={audit['rmse_dn']}, worst={audit['max_sample_rmse_dn']}")
print(f"[AUDIT PASS] RMSE DN={audit['rmse_dn']:.8g}; launching GPU workers next",flush=True)
write_json(OUT / "plan_deviations.json", {
    "variants": "B1/B2 only; B0 previously measured externally and not rerun",
    "precision": "AMP FP16 training; FP32 sensor/validation/test",
    "stages": 3, "shared_weights": True,
    "micro_batch": min(LEGACY_MICRO_BATCH_SIZE,EFFECTIVE_BATCH_SIZE),
    "schedule": "100 epochs, fixed LR 1e-4",
    "optimizer": "fused Adam" if FUSED_ADAM else "foreach Adam",
    "observation_compute": "FFT with FIR forward/backward checks" if USE_FFT_OBSERVATION else "original conv2d",
    "continuation": "batch-level recovery; compatible 200-cell imports use only states at/before epoch 100",
    "boundary": "feedback excludes 5 LR pixels at each patch boundary",
    "inference_geometry": "LMS/MS selects one phase once before recurrence; no test GT",
    "FR": "input agreement does not validate high-resolution detail or sensor physics"})

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
import gc, hashlib, json, math, os, random, shutil, signal, sys, time, traceback
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
from PIL import Image, ImageDraw

STOP_REQUESTED = False
class PauseRequested(Exception):
    pass

def request_pause(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

signal.signal(signal.SIGTERM, request_pause)
signal.signal(signal.SIGINT, request_pause)

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
    torch.save(value, tmp)
    if path.name == "latest.pt" and path.exists():
        previous = path.with_name("latest.prev.pt")
        previous_tmp = previous.with_name(previous.name + ".tmp")
        shutil.copy2(path, previous_tmp)
        os.replace(previous_tmp, previous)
    os.replace(tmp, path)

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
            if split == "train": batch["phase"] = TRAIN_PHASE.index_select(0, indices)
            yield batch
    else:
        raise RuntimeError("GPU residency is required")

def layout(b):
    return {k: (v.contiguous(memory_format=FORMAT) if v.ndim == 4 else v) for k, v in b.items()}

def build():
    model = model_for(C).cuda().to(memory_format=FORMAT)
    model.observation.fft_enabled = C["fft_observation"]
    return model


def make_optimizer(model):
    if C["fused_adam"]:
        return torch.optim.Adam(model.parameters(), lr=C["lr"], fused=True)
    return torch.optim.Adam(model.parameters(), lr=C["lr"], foreach=True)


def epoch_lr(epoch_index):
    return C["lr"]  # Same 100-epoch fixed-LR protocol in both arms.


def load_recovery():
    failures = []
    for name in ("latest.pt", "latest.prev.pt"):
        path = RUN / name
        if not path.exists():
            continue
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
            if ck["config"] != C:
                raise ValueError("Resume configuration differs; do not mix experiments")
            if not (0 <= ck["epoch"] <= C["epochs"]) or len(ck["history"]) != ck["epoch"]:
                raise ValueError("Invalid completed-epoch cursor")
            mid = ck.get("mid_epoch")
            if mid and (mid["epoch_index"] != ck["epoch"] or ck["epoch"] >= C["epochs"]):
                raise ValueError("Invalid partial-epoch cursor")
            print(f"[RESUME] {RUN.name}: completed {ck['epoch']}/100, "
                  f"next batch {mid['batches_done']+1 if mid else 1}; source={name}", flush=True)
            return ck
        except Exception as error:
            failures.append(f"{name}: {error}")
    if failures or os.environ.get("CTRS_REQUIRE_RESUME") == "1":
        raise RuntimeError("Cannot resume safely; no training started. " + "; ".join(failures))
    print(f"[NEW RUN] {RUN.name}: starting epoch 1/100", flush=True)
    return None


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
    if not records: raise RuntimeError("Configured micro batch does not fit. Set LEGACY_MICRO_BATCH_SIZE=16 for BOTH arms and choose a new OUTPUT_DIR. GPU residency stays enabled.")
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
        # Zero-valid pixels propagate NaN to the epoch-level finite check.
        sam += ((angles * valid).flatten(1).sum(1) / valid_counts).sum()
    result = {"val_mse": (sse / count).item(), "val_psnr_peak1": (psnr / samples).item(),
              "val_sam_deg": (sam / samples).item()}
    if not all(math.isfinite(v) for v in result.values()):
        raise RuntimeError("Invalid validation metrics")
    return result

def train():
    status("benchmark")
    micro = tune_micro()
    seed_all(C["seed"])
    gen = torch.Generator(device="cuda").manual_seed(C["seed"])
    model = build()
    opt = make_optimizer(model)
    core_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        core_hash.update(name.encode())
        core_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    initialization = {"seed": C["seed"], "core_sha256": core_hash.hexdigest(),
                      "excluded_candidate_layers": []}
    if (RUN / "bootstrap_initialization.json").exists():
        original = json.loads((RUN / "bootstrap_initialization.json").read_text())
        if original["core_sha256"] != initialization["core_sha256"]:
            raise ValueError("Original and current model initialization hashes differ")
    write(RUN / "initialization.json", initialization)
    scaler = torch.amp.GradScaler("cuda", enabled=C["amp"])
    start, best, best_epoch, history, resume_mid = 0, float("inf"), 0, [], None
    ck = load_recovery()
    if ck is not None:
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["cpu_rng"])
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
        gen.set_state(ck["loader_rng"])
        start, best, best_epoch, history = ck["epoch"], ck["best_mse"], ck["best_epoch"], ck["history"]
        resume_mid = ck.get("mid_epoch")
        del ck
        if best_epoch and not (RUN / "best.pt").exists():
            raise RuntimeError("Missing best checkpoint; restore it before resuming")
    total_batches = math.ceil(C["data"]["train"]["shapes"]["gt"][0] / C["batch_size"])

    def snapshot(completed_epoch, mid=None):
        return {"config": C, "epoch": completed_epoch, "best_mse": best, "best_epoch": best_epoch,
            "model": cpu(model.state_dict()), "optimizer": cpu(opt.state_dict()),
            "scaler": scaler.state_dict(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "loader_rng": gen.get_state(),
            "history": history, "mid_epoch": mid}

    for epoch in range(start, C["epochs"]):
        lr = epoch_lr(epoch)
        for group in opt.param_groups:
            group["lr"] = lr
        n_samples = C["data"]["train"]["shapes"]["gt"][0]
        if resume_mid is not None:
            if resume_mid["epoch_index"] != epoch:
                raise ValueError("Invalid mid-epoch recovery point")
            order = resume_mid["order"].cuda()
            completed_batches = resume_mid["batches_done"]
            if not (0 <= completed_batches <= total_batches) or len(order) != n_samples:
                raise ValueError("Invalid recovered batch cursor")
            sse = resume_mid["sse"].cuda()
            count = resume_mid["count"]
            seconds_before = resume_mid["train_seconds"]
            order_hash = resume_mid["sample_order_sha256"]
            resume_mid = None
        else:
            order = torch.randperm(n_samples, generator=gen, device="cuda")
            order_hash = hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
            completed_batches = 0
            sse = torch.zeros((), device="cuda")
            count = 0
            seconds_before = 0.
        model.train()
        torch.cuda.synchronize()
        tick = time.monotonic()
        last_status = last_checkpoint = tick
        last_step = completed_batches
        status("train", epoch=epoch + 1, batch=completed_batches, batches=total_batches, lr=lr)
        for step, b in enumerate(batches("train", C["batch_size"], order=order,
                                       start_batch=completed_batches), completed_batches + 1):
            b = layout(b)
            n = len(b["gt"])
            opt.zero_grad(set_to_none=True)
            for offset in range(0, n, micro):
                end = min(offset + micro, n)
                with torch.autocast("cuda", dtype=torch.float16, enabled=C["amp"]):
                    pred = model(*(b[k][offset:end] for k in INPUTS), phase=b["phase"][offset:end])
                    loss = F.mse_loss(pred.float(), b["gt"][offset:end])
                scaler.scale(loss * ((end - offset) / n)).backward()
                # Reuse the detached MSE; no second square/sum or batch .item().
                sse += loss.detach() * b["gt"][offset:end].numel()
                count += b["gt"][offset:end].numel()
            scaler.step(opt)
            scaler.update()
            now = time.monotonic()
            last_step = step
            if now - last_status >= C["status_seconds"]:
                # Periodic boundary makes ETA describe completed GPU work, not queued launches.
                torch.cuda.synchronize()
                elapsed = seconds_before + time.monotonic() - tick
                prior = [r["train_seconds"] for r in history[-5:]]
                measured = elapsed / step * total_batches
                estimate = float(np.mean(prior)) if prior else measured
                status("train", epoch=epoch + 1, batch=step, batches=total_batches, lr=lr,
                    train_remaining_seconds=max(0., estimate - elapsed) + (C["epochs"] - epoch - 1)*estimate,
                    eta_scope="training phase includes recovery checkpoints; validation/final checkpoint/test excluded")
                last_status = time.monotonic()
            if STOP_REQUESTED or (now - last_checkpoint >= C["checkpoint_seconds"] and step < total_batches):
                torch.cuda.synchronize()
                elapsed = seconds_before + time.monotonic() - tick
                status("checkpoint", epoch=epoch + 1, batch=step, batches=total_batches, lr=lr)
                mid = {"epoch_index": epoch, "batches_done": step, "sse": sse.detach().cpu(),
                    "count": count, "order": order.detach().cpu(), "train_seconds": elapsed,
                    "sample_order_sha256": order_hash}
                save(snapshot(epoch, mid), RUN / "latest.pt")
                write(RUN / "resume_point.json", {"completed_epochs": epoch,
                      "partial_epoch": epoch+1, "completed_batches": step, "batches": total_batches,
                      "next_batch": step+1, "updated_unix": time.time()})
                last_checkpoint = time.monotonic()
                if STOP_REQUESTED:
                    status("paused", epoch=epoch + 1, batch=step, batches=total_batches)
                    raise PauseRequested("Saved the last completed batch; rerun this cell to resume")
                status("train", epoch=epoch + 1, batch=step, batches=total_batches, lr=lr)
        train_mse = (sse / count).item()
        if not math.isfinite(train_mse):
            raise RuntimeError("Nonfinite training loss")
        torch.cuda.synchronize()
        training_seconds = seconds_before + time.monotonic() - tick
        status("validation", epoch=epoch + 1, batch=last_step, batches=total_batches, lr=lr)
        val_started = time.monotonic()
        validation = validate(model)
        validation_seconds = time.monotonic() - val_started
        improved = validation["val_mse"] < best
        if improved:
            best, best_epoch = validation["val_mse"], epoch + 1
        row = {"epoch": epoch + 1, "train_mse": train_mse, **validation, "lr": lr,
            "seconds": training_seconds + validation_seconds, "train_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "train_samples_per_second": n_samples / max(training_seconds, 1e-9),
            "scaler_scale": scaler.get_scale(), "micro_batch": micro,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "sample_order_sha256": order_hash}
        history.append(row)
        status("checkpoint", epoch=epoch + 1, batch=total_batches, batches=total_batches, metrics=row, lr=lr)
        ck = snapshot(epoch + 1)
        if improved:
            save(ck, RUN / "best.pt")
        save(ck, RUN / "latest.pt")
        write(RUN / "resume_point.json", {"completed_epochs": epoch+1,
              "partial_epoch": None, "completed_batches": 0, "updated_unix": time.time()})
        if STOP_REQUESTED:
            status("paused", epoch=epoch + 1, batch=total_batches, batches=total_batches)
            raise PauseRequested("Completed epoch checkpoint saved")
        write(RUN / "history.json", history)
        del ck, order
        print(json.dumps(row), flush=True)
    write(RUN / "training_complete.json", {"config": C, "epoch": C["epochs"],
        "best_epoch": best_epoch, "best_mse": best})
    del model, opt, scaler
    gc.collect()
    torch.cuda.empty_cache()
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
                begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                with torch.inference_mode():
                    begin.record()
                    prediction = model(*inputs).float()
                    end.record(); end.synchronize()
                    inference_ms = begin.elapsed_time(end)
                    phase = model.select_phase(inputs[1], inputs[2])
                    observed = model.samples(model.observation.blur(prediction))[
                        torch.arange(len(prediction),device="cuda"),phase]
                    m = model.margin
                    input_mse = F.mse_loss(observed[...,m:-m,m:-m],inputs[2][...,m:-m,m:-m]).item()
                    pred = prediction[0].cpu().numpy() * C["scale"]
                del prediction, observed
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
                metrics["MS_consistency_MSE_peak1_interior"] = input_mse
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
    if C["mode"] in ("B1", "B2"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("B2_models", C["B2_source"])
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
        status("smoke_checks")
        seed_all(C["seed"])
        first = PASTED.build_model("B1", C["channels"], C["k"])
        seed_all(C["seed"])
        second = PASTED.build_model("B2", C["channels"], C["k"])
        if any(not torch.equal(v, second.state_dict()[k]) for k,v in first.state_dict().items()):
            raise RuntimeError("B1/B2 parameter initialization differs")
        del first
        second = second.cuda()
        if C["fft_observation"]:
            # Prove the optimization preserves the FIR including replication boundaries.
            obs = second.observation
            probe = torch.linspace(0, 1, 4*64*64, device="cuda").reshape(1,4,64,64).requires_grad_()
            weights = torch.linspace(.1, 1, probe.numel(), device="cuda").reshape_as(probe)
            obs.fft_enabled = False
            ref = obs.blur(probe)
            ref_grad = torch.autograd.grad((ref * weights).sum(), probe)[0]
            obs.fft_enabled = True
            got = obs.blur(probe)
            got_grad = torch.autograd.grad((got * weights).sum(), probe)[0]
            torch.testing.assert_close(got, ref, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(got_grad, ref_grad, rtol=8e-5, atol=1e-5)
            write(RUN / "fft_parity.json", {"forward_max_abs": (got-ref).abs().max().item(),
                "backward_max_abs": (got_grad-ref_grad).abs().max().item(),
                "tested_geometry": [64,64], "boundary": "replicate", "precision": "FP32",
                "scope": "numerical FIR equivalence; not a training performance result"})
            del probe, weights, ref, got, ref_grad, got_grad
        with h5py.File(C["data"]["train"]["path"], "r") as h5:
            example = {k: torch.from_numpy(np.asarray(h5[k][:1],dtype=np.float32)/C["scale"]).cuda() for k in KEYS}
        pred = second(*(example[k] for k in INPUTS))
        if pred.shape != example["gt"].shape: raise RuntimeError("Smoke output shape mismatch")
        F.mse_loss(pred, example["gt"]).backward()
        gradients = [p.grad for p in second.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError("Smoke gradient check failed")
        if second.error_encoder.weight.grad is None or second.error_encoder.weight.grad.abs().sum().item() == 0:
            raise RuntimeError("Observation feedback has no gradient")
        write(RUN / "smoke_checks.json", {"matching_initialization":True,"shape":list(pred.shape),
              "finite_gradients":True,"feedback_gradient":True,"optimizer_steps":0})
        del second, example, pred, gradients
        gc.collect(); torch.cuda.empty_cache()
        status("loading_cache")
        GPU_DATA = load_gpu_cache()
        profile = json.loads(Path(C["observation_profile"]).read_text())
        if sha(C["observation_profile"]) != C["observation_profile_sha256"]:
            raise ValueError("Observation manifest changed")
        mapping = {(2,2):0, (1,2):1, (2,1):2}
        TRAIN_PHASE = torch.tensor([mapping[tuple(p)] for p in profile["phases"]],device="cuda")
        train()
        training_wall_seconds = time.monotonic() - run_started
        for loader in LOADERS.values():
            if hasattr(loader.dataset, "close"): loader.dataset.close()
        GPU_DATA = None; del TRAIN_PHASE; LOADERS.clear(); gc.collect(); torch.cuda.empty_cache()
        evaluate()
        write(RUN / "session_timing.json", {"training_wall_seconds_this_session": training_wall_seconds,
              "total_wall_seconds_this_session": time.monotonic() - run_started,
              "scope": "cache load + benchmark + completed/replayed training + checkpoint saves + test; resumed session is not whole-experiment time"})
    except PauseRequested as error:
        print(f"[PAUSED] {error}", flush=True)
        sys.exit(130)
    except BaseException:
        write(RUN / "failure.json", {"traceback": traceback.format_exc(), "time": time.time()})
        traceback.print_exc(); sys.exit(1)
'''

worker_path = OUT / "code" / "kaggle_worker.py"
worker_hash = hashlib.sha256(WORKER.encode()).hexdigest()
if worker_path.exists() and sha(worker_path) != worker_hash:
    raise ValueError("Worker code changed; choose a new OUTPUT_DIR.")
worker_path.write_text(WORKER, encoding="utf-8")
LEGACY_WORKER_SHA = '5103d95cd6d2fff4c82827f8e0cbb4b0ab7d806fa4ce545145ad0669d7bf4a4d'

def migrate_200_run(old_run, config):
    new_run = Path(config["run_dir"])
    if not (old_run / "initialization.json").exists():
        raise FileNotFoundError("Missing original initialization evidence")
    new_run.mkdir(parents=True, exist_ok=True)
    candidates = [old_run / "latest.pt", old_run / "latest.prev.pt"]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        raise FileNotFoundError(f"No compatible previous checkpoint: {old_run}")
    ck = None
    for path in candidates:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=True)
            break
        except Exception:
            continue
    if ck is None:
        raise RuntimeError(f"Old recovery checkpoints are unreadable: {old_run}")
    if ck["epoch"] > 100 or (ck.get("mid_epoch") and ck["epoch"] >= 100):
        path = old_run / "best_first100.pt"
        if not path.is_file():
            raise ValueError("Old run already passed epoch 100 but no pre-100 checkpoint was preserved")
        ck = torch.load(path, map_location="cpu", weights_only=True)
    old = ck["config"]
    for key in ("sensor", "k", "channels", "mode", "seed", "variant", "batch_size", "lr",
                "micro_batch", "amp", "channels_last", "fused_adam", "fft_observation",
                "repo_commit", "upstream_commit", "B2_source_sha256", "controlled"):
        if old[key] != config[key]:
            raise ValueError(f"Incompatible checkpoint field {key}: {old_run}")
    if old["worker_sha256"] != LEGACY_WORKER_SHA or old["epochs"] != 200 or old["hold_lr_epochs"] != 100:
        raise ValueError("Only the exact attached 200-epoch cell's checkpoint may migrate")
    for split in ("train", "val", "RR", "FR"):
        if old["data"][split]["sha256"] != config["data"][split]["sha256"] or old["data"][split]["shapes"] != config["data"][split]["shapes"]:
            raise ValueError(f"Dataset changed for {split}; do not mix old weights")
    if not (0 <= ck["epoch"] <= 100) or len(ck["history"]) != ck["epoch"]:
        raise ValueError("Invalid import cursor")
    if any(r.get("lr") != config["lr"] for r in ck["history"]):
        raise ValueError("Old history does not use the same constant learning rate")
    provenance = {"original_path": str(path), "original_sha256": sha(path),
                  "original_config": old, "completed_epochs": ck["epoch"],
                  "partial_epoch": bool(ck.get("mid_epoch")), "history_imported": len(ck["history"])}
    # Read best from the matching original state; never reuse an after-100 best.
    if ck["best_epoch"]:
        best_path = old_run / ("best_first100.pt" if (old_run / "best_first100.pt").exists() else "best.pt")
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        if best["epoch"] != ck["best_epoch"] or best["best_mse"] != ck["best_mse"]:
            raise ValueError("Old best does not match the recovery cursor; need its original best checkpoint")
        best["config"] = {**config, "reference_plan_config": old["reference_plan_config"]}
        temp = new_run / "best.pt.tmp"
        torch.save(best, temp)
        os.replace(temp, new_run / "best.pt")
    ck["config"] = {**config, "reference_plan_config": old["reference_plan_config"]}
    temp = new_run / "latest.pt.tmp"
    torch.save(ck, temp)
    os.replace(temp, new_run / "latest.pt")
    # Preserve the original initialization evidence, not a synthesized hash.
    shutil.copy2(old_run / "initialization.json", new_run / "bootstrap_initialization.json")
    write_json(new_run / "bootstrap_manifest.json", provenance)
    print(f"[IMPORTED] {new_run.name}: completed {ck['epoch']}/100; full optimizer/scaler/RNG retained", flush=True)


gpu_count = torch.cuda.device_count() if USE_ALL_GPUS else 1
if len(TRAINING_MODES) == 2:
    gpu_count = min(gpu_count, 2)
    if gpu_count == 2:
        a, b = (torch.cuda.get_device_properties(i) for i in (0, 1))
        if a.name != b.name or a.total_memory != b.total_memory:
            raise ValueError("Controlled parallel comparison requires two equivalent GPUs.")
    print("Pair assignment: GPU 0=B1; GPU 1=B2 (swap optional)" if gpu_count == 2 else
          "Only one GPU: paired modes will run sequentially, not concurrently.")
configs = []
for mode, seed, variant in [(m, s, v) for m in TRAINING_MODES for s in SEEDS
                           for v in [m]]:
        run_id = f"{mode}_{SENSOR}_{variant}_k{K}_s{seed}"
        config = {"sensor": SENSOR, "k": K, "channels": 8 if SENSOR == "WV3" else 4,
                  "mode": mode, "controlled": CONTROLLED_COMPARISON,
                  "epochs": EPOCHS, "seed": seed, "variant": variant,
                  "schedule": "constant",
                  "status_seconds": STATUS_INTERVAL_SECONDS, "checkpoint_seconds": CHECKPOINT_INTERVAL_SECONDS,
                  "fused_adam": FUSED_ADAM, "fft_observation": USE_FFT_OBSERVATION,
                  "batch_size": EFFECTIVE_BATCH_SIZE, "lr": LEARNING_RATE,
                  "micro_batch": min(LEGACY_MICRO_BATCH_SIZE, EFFECTIVE_BATCH_SIZE),
                  "auto_micro": not CONTROLLED_COMPARISON,
                  "eval_batch": min(LEGACY_MICRO_BATCH_SIZE, EFFECTIVE_BATCH_SIZE),
                  "amp": AMP,
                  "channels_last": CHANNELS_LAST,
                  "gpu_cache": GPU_CACHE, "workers": NUM_WORKERS, "scale": SCALE,
                  "save_predictions": SAVE_PREDICTIONS, "data": data_info,
                  "cache": str(CACHE), "source": str(SOURCE), "run_dir": str(OUT / "runs" / run_id),
                  "selection": selection, "repo_commit": REPO_COMMIT,
                  "plan_commit": PLAN_COMMIT,
                  "B2_source": str(custom_path), "B2_source_sha256": custom_hash,
                  "upstream_commit": UPSTREAM_COMMIT, "worker_sha256": worker_hash,
                  "observation_profile": str(audit_path), "observation_profile_sha256": sha(audit_path),
                  "model_definition": "3 shared SSA-MRN stages; B1 zero error; B2 observed MS residual feedback",
                  "numerics": ("Controlled: equal AMP/layout/micro batch, deterministic kernels, no TF32; not legacy bitwise parity" if CONTROLLED_COMPARISON
                               else "Speed profile: both arms use AMP/channels-last and measured micro batch; micro batches may differ")}
        cfg_path = OUT / "code" / f"{run_id}.json"
        if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
            raise ValueError(f"Settings changed for {run_id}; choose a new OUTPUT_DIR.")
        write_json(cfg_path, config)
        if LEGACY_BOOTSTRAP and not (Path(config["run_dir"]) / "latest.pt").exists():
            migrate_200_run(LEGACY_BOOTSTRAP / "runs" / run_id, config)
        configs.append((config, cfg_path))
write_json(OUT / "environment.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
           "gpus": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
           "python": sys.version, "parallel_workers": min(gpu_count, len(configs)),
           "swap_gpus": SWAP_GPUS,
           "old_checkpoints": "only exact attached 200-cell pre-100 states may migrate; matching 100 suite resumes at saved batch"})

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
                assigned_mode = "B1" if device == reference_device else "B2"
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
                finished.append(run)
                continue
            if (run / "failure.json").exists(): (run / "failure.json").unlink()
            write_json(status_path, {"phase": "starting", "run_id": run.name})
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": str(device), "OMP_NUM_THREADS": "2",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                        "CTRS_REQUIRE_RESUME": "1" if REQUIRE_RESUME else "0"})
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
            try:
                state = json.loads((run / "status.json").read_text())
            except (OSError, ValueError):
                state = {"phase": "status_update"}
            print(f"GPU {device}: {run.name} | {state.get('phase')} | epoch {state.get('epoch', '-')}/{EPOCHS} "
                  f"| batch {state.get('batch', '-')}/{state.get('batches', '-')} "
                  f"| allocated {state.get('allocated_gb', 0):.2f} GB | peak {state.get('peak_gb', 0):.2f} GB")
            if "cache_percent" in state:
                print(f"  H5 -> GPU {state['cache_array']}: {state['cache_percent']:.1f}%",flush=True)
            if "train_remaining_seconds" in state:
                print(f"  This run training ETA: {state['train_remaining_seconds'] / 60:.1f} min; evaluation excluded")
            if "lr" in state:
                print(f"  Learning rate: {state['lr']:.3g}")
            if "scene" in state:
                print(f"  Full test scenes: {state['scene']}/{state['scenes']}")
            rc = process.poll()
            if rc is not None:
                log.close(); del active[device]
                if rc == 130:
                    raise KeyboardInterrupt(f"{run.name} paused after saving; the pair is stopping safely")
                if rc:
                    failure = run / "failure.json"
                    detail = failure.read_text() if failure.exists() else (run / "worker.log").read_text()[-6000:]
                    raise RuntimeError(f"Worker failed: {run}\n{detail}")
                finished.append(run)
        print("A total ETA is not asserted before each variant and evaluation speed are measured.")
        if pending or active:
            time.sleep(5)
finally:
    # Stopping this cell also stops its child training processes.
    for process, log, run in active.values():
        process.terminate()
    for process, log, run in active.values():
        try: process.wait(timeout=120)  # Worker finishes its current batch and saves before exiting.
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
    base = results[f"B1_{SENSOR}_B1_k{K}_s{seed}"]["metrics"]
    result["delta_from_same_seed_B1"] = {
        protocol: {key: val - base[protocol]["mean"][key] for key, val in block["mean"].items()}
        for protocol, block in result["metrics"].items()}
    for protocol, block in result["metrics"].items():
        for key, val in block["mean"].items():
            summary_rows.append({"run_id": run_id, "mode": mode, "seed": seed, "best_epoch": result["best_epoch"],
                "protocol": protocol, "metric": key, "mean": val,
                "delta_from_B1": result["delta_from_same_seed_B1"][protocol][key]})
write_json(OUT / "results.json", results)
with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
    writer.writeheader(); writer.writerows(summary_rows)

for seed in SEEDS:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for mode, variant in [("B1", "B1"), ("B2", "B2")]:
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
        value = results[f"B2_{SENSOR}_B2_k{K}_s{seed}"]["delta_from_same_seed_B1"][protocol][key]
        ax.bar(["B2"], [value], color="#2788b5")
        ax.set_title(f"{protocol} {key}: B2 minus B1" + (" (provisional)" if protocol == "FR" else ""))
        ax.axhline(0, color="black", linewidth=.8); ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(OUT / f"test_comparison_seed{seed}.png", dpi=150); plt.close(fig)

paired = []
if set(TRAINING_MODES) == {"B1", "B2"}:
    for seed in SEEDS:
        for variant in ["B2"]:
            old_id = f"B1_{SENSOR}_B1_k{K}_s{seed}"
            new_id = f"B2_{SENSOR}_{variant}_k{K}_s{seed}"
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
            paired.append({"variant": variant, "seed": seed, "B1_epoch_seconds": old_sec,
                "B2_epoch_seconds": new_sec, "epoch_time_ratio_B1_over_B2": old_sec / new_sec,
                "common_core_initialization_verified": True, "epoch_sample_order_verified": True,
                "timing_scope": "training phase including periodic recovery checkpoints + validation; epoch-end checkpoint/bootstrap/test excluded",
                "metric_delta_B2_minus_B1": {p: {k: v - old["metrics"][p]["mean"][k]
                    for k, v in b["mean"].items()} for p, b in new["metrics"].items()}})
    write_json(OUT / "paired_comparison.json", paired)

report_lines = ["# Kaggle SSA-MRN B1/B2", "",
    "Status: full training and RR/FR evaluation completed.",
    f"Learning rate: fixed {LEARNING_RATE}; hard maximum {EPOCHS} epochs.",
    "Sources: fresh run or explicitly audited pre-100 continuation; see bootstrap_manifest.json when imported.",
    f"QB; K={K}; epochs={EPOCHS}; seeds={SEEDS}; effective batch={EFFECTIVE_BATCH_SIZE}; micro={LEGACY_MICRO_BATCH_SIZE}.",
    "B0 was measured previously and is not executed or numerically reconstructed here.",
    "B1: 3 shared stages with zero observation error; B2: same stages with MS-D(current) feedback.",
    "Same core/error encoder/alpha initialization, sample order and final-output MSE.",
    "Source MTF is fixed; TRAIN GT/MS phases audited before training. Validation/test phase uses observed LMS/MS only.",
    "Feedback masks unknown patch halos; inference is FP32 on whole scenes. Best selected only by validation MSE.",
    "FR input consistency does not establish correct HR detail. Repository QNR/Q2n/SCC MATLAB parity limitations remain.",
    "GPU-resident FP32 data, GPU sampling/batching, periodic status synchronization; no CPU DataLoader.",
    "Mid-epoch recovery stores the permutation, batch cursor, optimizer/scaler/RNG and loss accumulator.",
    "Best checkpoint uses validation MSE only; no epochs beyond 100 are trained.",
    "Fused Adam and optional checked FFT observation are shared runtime settings; old runs are not bitwise reproduced.",
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
