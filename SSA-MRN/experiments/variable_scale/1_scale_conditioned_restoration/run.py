# Paste the ENTIRE file into one Kaggle Python cell. T4 x2; no source download.
# Full QB research suite: six jobs, 100 epochs each. Missing original scene IDs are explicitly recorded as unverified.
MODE = "train"                 # "smoke", "train", "evaluate"
OUTPUT_DIR = f"/kaggle/working/ssa_variable_scale_{MODE}_v4"
RESUME = False                 # Same-session resume: True, same training OUTPUT_DIR.
RESUME_INPUT = ""              # Exact suite directory in a connected Saved Output.
SCENE_MANIFEST = ""            # JSON: split -> [{"scene_id": ..., "tile": ...}, ...].
ALLOW_UNVERIFIED_SPLITS = True # Explicit exploratory opt-in if original scene IDs absent.
PROVISIONAL_FR = False         # Existing x4 QNR is NOT MATLAB parity verified.
EPOCHS = 100
SEED = 42
EFFECTIVE_BATCH = 32
MICRO_BATCH = 32                # Change equally in all jobs, in a fresh experiment.
LEARNING_RATE = 1e-4
AMP = False                    # Same FP32 default in all six jobs; True is a separate config.
JOB_NAMES = ["A_x4", "B_x2", "B_x3", "B_x4", "C_mixed", "D_mixed"]
DATA_ROOT = "/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset"
TRAIN_H5 = f"{DATA_ROOT}/QuickBird__Training_Dataset__train_qb.h5"
VAL_H5 = f"{DATA_ROOT}/QuickBird__Training_Dataset__valid_qb.h5"
RR_H5 = f"{DATA_ROOT}/QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5"
FR_H5 = f"{DATA_ROOT}/QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5"

EMBEDDED_SOURCES = {
'__init__.py': r'''"""Independent, controlled QB variable-scale experiment (K=4)."""
''',
'adapter_snapshot.py': r'''"""Minimal forward-path repair around the untouched official network.py.

The official forward(pan, lms) references an undefined `ms`. Its three
spatial stages require PAN and interpolated LMS at full resolution, plus
the original MS at one-quarter resolution. This adapter supplies that input
explicitly and retains the original modules, parameter names and operations.
"""

import importlib.util
from pathlib import Path

import torch
from torch import nn


UPSTREAM_PATH = Path(__file__).resolve().parents[3] / "references/upstream/network.py"


def _upstream_model():
    if not UPSTREAM_PATH.is_file():
        raise FileNotFoundError(f"Initialize the official SSA-MRN submodule: {UPSTREAM_PATH}")
    spec = importlib.util.spec_from_file_location("ssamrn_official_network", UPSTREAM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PansharpeningNet


class DirectMLPReLU(nn.PReLU):
    """Equivalent PReLU expression avoiding a DirectML batch backward error."""

    def forward(self, input):
        slope = self.weight.view(1, -1, 1, 1)
        return torch.where(input > 0, input, slope * input)


class RestoredPansharpeningNet(_upstream_model()):
    """Official layers with an explicit low-resolution MS input."""

    def __init__(self, channels, ssai_dimension=4, guide_channels=1):
        super().__init__(channels)
        if guide_channels not in (1, 3):
            raise ValueError("guide_channels must be 1 (PAN) or 3 (RGB)")
        self.guide_channels = guide_channels
        if guide_channels == 3:
            self.conv1t1 = nn.Conv2d(3, 3, 3, padding=1)
            self.cov2t64 = nn.Conv2d(channels + 6, 64, 3, padding=1)
            for blocks in (self.SSA_blocks, self.SSA_blocks1, self.SSA_blocks2):
                for block in blocks:
                    block.conv1t6 = nn.Conv2d(3, 4, 3, padding=1)
        if ssai_dimension < 1:
            raise ValueError("ssai_dimension must be positive")
        self.ssai_dimension = ssai_dimension
        if ssai_dimension == 4:
            return  # Preserve the published upstream architecture and existing checkpoints.

        # The upstream code hard-codes K=4, while the paper specifies K=6.
        # Resize only the layers used by the restored forward path; leave upstream untouched.
        self.fixed = ssai_dimension
        for blocks in (self.SSA_blocks, self.SSA_blocks1, self.SSA_blocks2):
            for block in blocks:
                block.fixed = ssai_dimension
                block.conv1t6 = nn.Conv2d(guide_channels, ssai_dimension, 3, padding=1)
                block.conv6t6 = nn.Conv2d(ssai_dimension, ssai_dimension, 3, padding=1)
                block.conv7t6_3 = nn.Conv2d(channels + 1, ssai_dimension, 3, padding=1)
        for name in ("conv48tnum1", "conv48tnum2", "conv48tnum3"):
            setattr(self, name, nn.Conv2d(ssai_dimension * channels, channels, 3, padding=1))

    def use_directml_prelu(self):
        """Keep PReLU weights/checkpoints while using a DirectML-safe backward."""
        for name, module in self.named_modules():
            if not name or not isinstance(module, nn.PReLU):
                continue
            parent = self.get_submodule(name.rpartition(".")[0]) if "." in name else self
            replacement = DirectMLPReLU(module.num_parameters)
            with torch.no_grad():
                replacement.weight.copy_(module.weight)
            setattr(parent, name.rpartition(".")[2], replacement)

    def _attend(self, pan, features, ms, blocks, fusion, activation):
        outputs = []
        for band in range(self.channels):
            band_input = torch.cat([features, ms[:, band : band + 1]], dim=1)
            outputs.append(blocks[band](pan, band_input))
        return activation(fusion(torch.cat(outputs, dim=1)))

    def forward(self, pan, lms, ms):
        if pan.ndim != 4 or lms.ndim != 4 or ms.ndim != 4:
            raise ValueError("Expected NCHW tensors for pan, lms, and ms")
        n, channels, h, w = lms.shape
        if channels != self.channels or h % 4 or w % 4 or min(h, w) < 4:
            raise ValueError(f"Invalid LMS shape: {tuple(lms.shape)}")
        if pan.shape != (n, self.guide_channels, h, w) or ms.shape != (n, channels, h // 4, w // 4):
            raise ValueError("Expected guide at LMS resolution and MS at 1/4 resolution")

        pan_down_up = self.upsample1(self.downsample1(pan))
        pan_updown = self.prelu(self.conv1t1(pan_down_up))
        features = torch.cat([lms, pan, pan_updown], dim=1)
        features = self.backbone(self.cov2t64(features))
        res = self.conv64t48(features) + lms

        pan_ms = self._attend(pan, res, lms, self.SSA_blocks, self.conv48tnum1, self.prelu1)
        res = res + pan_ms

        res_half = self.downsample100(res)
        ms_half = self.upsample100(ms)
        pan_half = self.downsample100(pan)
        pan_ms = self._attend(pan_half, res_half, ms_half, self.SSA_blocks1, self.conv48tnum2, self.prelu2)
        res_half = res_half + pan_ms

        res_quarter = self.downsample200(res_half)
        pan_quarter = self.downsample2(pan)
        pan_ms = self._attend(pan_quarter, res_quarter, ms, self.SSA_blocks2, self.conv48tnum3, self.prelu3)

        output = self.conv2ctc1(torch.cat([ms, pan_ms], dim=1))
        output = self.conv2ctc2(torch.cat([self.upsample101(output), res_half], dim=1))
        output = self.conv2ctc3(torch.cat([self.upsample102(output), res], dim=1))
        return output
''',
'data.py': r'''"""Explicit synthetic degradation, provenance checks and resident tensors."""
import hashlib
import json
import math
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

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

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

def inspect_sources(paths, provenance_path=None, allow_unverified=False):
    paths = {k: Path(v).resolve() for k, v in paths.items()}
    if len(set(paths.values())) != len(paths):
        raise ValueError("Different splits must use different H5 paths")
    provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8")) if provenance_path else {}
    result, fingerprints, scene_groups = {}, {}, {}
    for split, path in paths.items():
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
            for i in range(n):
                digest = hashlib.sha256()
                for k in keys:
                    arr = np.asarray(f[k][i], dtype=np.float32)
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
            result[split] = {"path": str(path), "sha256": sha_file(path),
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
''',
'engine.py': r'''"""Training, atomic batch-boundary recovery, current-epoch status and ETA."""
import contextlib
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from .data import ResidentSplit, canonical_hash, sha_file
from .model import VariableScaleNet, DeterministicUpsample
from .reference import ReferenceX4
from .scoring import rr_score, scene_means

KST = dt.timezone(dt.timedelta(hours=9))

def replace_atomic(source, destination):
    # Windows can briefly deny replace while a dashboard reader has the old file open.
    for attempt in range(100):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 99:
                raise
            time.sleep(.01)

@contextlib.contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Run is already active; exclusive lock unavailable: {path}") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)

def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    replace_atomic(tmp, path)

def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)

def atomic_save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    replace_atomic(tmp, path)

def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_model(job, seed, device):
    seed_all(seed)
    model = ReferenceX4(4, ssai_dimension=4) if job["legacy"] else VariableScaleNet(4, job["conditioned"])
    if job["legacy"]:
        for name, layer in list(model.named_children()):
            if isinstance(layer, torch.nn.Upsample) and layer.mode == "bilinear":
                setattr(model, name, DeterministicUpsample(size=layer.size, scale_factor=layer.scale_factor,
                                                          mode=layer.mode, align_corners=layer.align_corners))
    return model.to(device)

def predict(model, batch, legacy):
    args = [batch[k] for k in ("pan", "lms", "ms")]
    return model(*args) if legacy else model(*args, batch["scale"])

def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)

def evaluate(model, split, scales, legacy, batch_size, amp=False):
    model.eval()
    result = {}
    with torch.inference_mode():
        for scale in scales:
            rows = []
            for start in range(0, split.length, batch_size):
                ids = torch.arange(start, min(start+batch_size, split.length), device=split.device)
                batch = split.batch(ids, scale, legacy)
                with torch.autocast(split.device.type, enabled=amp):
                    output = predict(model, batch, legacy)
                targets, predictions = batch["gt"].float().cpu().numpy(), output.float().cpu().numpy()
                for i, (gt, pred) in enumerate(zip(targets, predictions), start):
                    rows.append({"sample_index": i, "scene_id": split.entries[i]["scene_id"],
                                 "scale": scale, "metrics": rr_score(gt, pred, scale)})
            result[str(scale)] = scene_means(rows)
    return result

class Trainer:
    def __init__(self, config, job, manifest, run_dir, device, source_hash, resume=False):
        self.config, self.job, self.manifest = config, job, manifest
        self.run_dir, self.device = Path(run_dir), torch.device(device)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(config.get("cpu_threads", 2))
        torch.use_deterministic_algorithms(config["deterministic"])
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if config["amp"] and self.device.type != "cuda":
            raise ValueError("AMP requires CUDA; smoke must explicitly disable AMP")
        self.model = make_model(job, config["seed"], self.device)
        self.initial_hash = tensor_hash({k: v for k, v in self.model.state_dict().items()
                                         if not k.startswith("film.")})
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], foreach=True)
        self.step_applied = False
        self.optimizer.register_step_post_hook(lambda *_: setattr(self, "step_applied", True))
        self.scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
        self.generator = torch.Generator(device=self.device).manual_seed(config["seed"] + 1000)
        # Paths/output/resume mode do not identify data; bytes/provenance do.
        identity_data = copy.deepcopy(manifest)
        for item in identity_data["files"].values():
            item.pop("path", None)
        self.identity = {"config": config, "job": job, "data": identity_data,
                         "source_hash": source_hash, "torch": torch.__version__}
        self.signature = canonical_hash(self.identity)
        self.epoch, self.next_batch, self.order, self.scale_order = 0, 0, None, None
        self.loss_sum, self.sample_count = torch.zeros((), device=self.device), 0
        self.history, self.order_history = [], []
        self.exposures = {str(r): {"samples": 0, "target_pixels": 0, "updates": 0} for r in job["scales"]}
        self.best_value, self.best_epoch, self.best_model = float("inf"), None, None
        self.attempted_updates, self.applied_updates = 0, 0
        self.timing = {"train_seconds": 0., "measured_steps": 0, "val_seconds": [], "save_seconds": []}
        self.stop_requested, self.pending_events = False, []
        self.last_status = self.last_save = time.monotonic()
        if resume:
            if not (self.run_dir / "latest.pt").is_file():
                raise FileNotFoundError("Resume requires this job's latest.pt")
            self.load(self.run_dir / "latest.pt")
        elif (self.run_dir / "latest.pt").exists():
            raise FileExistsError("Existing run: select explicit resume or a fresh output directory")
        json_write(self.run_dir / "identity.json", self.identity)
        json_write(self.run_dir / "initialization.json", {"common_parameter_sha256": self.initial_hash,
                   "parameters": sum(p.numel() for p in self.model.parameters()), "k": 4})
        try:
            self.train = ResidentSplit(manifest["files"]["train"], self.device)
            self.val = ResidentSplit(manifest["files"]["val"], self.device)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError("GPU residency failed. No CPU fallback. Use a matched new configuration "
                               "or a larger GPU; changing micro batch alone may not fit the data.") from error
        if not resume:
            self.save()
        # Reconstruct best from latest, even when its separate file was lost.
        if self.best_model is not None:
            self.save_best()

    @property
    def batches(self):
        return math.ceil(self.train.length / self.config["batch_size"])

    def load(self, path):
        # Only load checkpoints produced by this code from trusted local/Saved Output.
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("signature") != self.signature:
            raise ValueError("Checkpoint code/config/data identity mismatch; use a new run")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        for key in ("epoch", "next_batch", "scale_order", "sample_count", "history", "order_history",
                    "exposures", "best_value", "best_epoch", "best_model", "attempted_updates",
                    "applied_updates", "timing"):
            setattr(self, key, state[key])
        self.order = state["order"].to(self.device) if state["order"] is not None else None
        self.loss_sum = state["loss_sum"].to(self.device)
        self.generator.set_state(state["generator"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], self.device)

    def measure_pending(self):
        synchronize(self.device)
        for start, end in self.pending_events:
            self.timing["train_seconds"] += start.elapsed_time(end) / 1000
            self.timing["measured_steps"] += 1
        self.pending_events.clear()

    def checkpoint(self):
        return {"signature": self.signature, "model": cpu_tree(self.model.state_dict()),
                "optimizer": cpu_tree(self.optimizer.state_dict()), "scaler": self.scaler.state_dict(),
                "scheduler": {"kind": "constant", "lr": self.config["lr"]},
                "epoch": self.epoch, "next_batch": self.next_batch,
                "order": cpu_tree(self.order), "scale_order": self.scale_order,
                "loss_sum": cpu_tree(self.loss_sum), "sample_count": self.sample_count,
                "generator": self.generator.get_state().cpu(), "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(self.device).cpu() if self.device.type == "cuda" else None,
                "best_value": self.best_value, "best_epoch": self.best_epoch, "best_model": self.best_model,
                "history": self.history, "order_history": self.order_history,
                "exposures": self.exposures, "attempted_updates": self.attempted_updates,
                "applied_updates": self.applied_updates, "timing": self.timing,
                "complete_epochs": self.epoch, "lr": self.config["lr"]}

    def save_best(self):
        atomic_save(self.run_dir / "best.pt", {"signature": self.signature,
                    "model": self.best_model, "epoch": self.best_epoch, "val_mse": self.best_value})

    def save(self):
        self.measure_pending()
        start = time.monotonic()
        atomic_save(self.run_dir / "latest.pt", self.checkpoint())
        self.timing["save_seconds"].append(time.monotonic()-start)
        self.last_save = time.monotonic()

    def status(self, stage="train", validation=None):
        self.measure_pending()
        done = self.epoch * self.batches + self.next_batch
        total = self.config["epochs"] * self.batches
        remaining = total - done
        eta = None
        exclusions = ["test scoring", "plots", "ZIP/I/O", "queued models without measured timing"]
        if self.timing["measured_steps"]:
            eta = remaining * self.timing["train_seconds"] / self.timing["measured_steps"]
            if self.timing["val_seconds"]:
                eta += (self.config["epochs"]-self.epoch) * np.mean(self.timing["val_seconds"])
            else:
                exclusions.append("validation not yet measured")
            if self.timing["save_seconds"]:
                # Lower bound: epoch saves, plus measured periodic save rate.
                eta += (self.config["epochs"]-self.epoch) * np.mean(self.timing["save_seconds"])
            exclusions.append("additional periodic checkpoint saves")
        mem = {"allocated": 0, "reserved": 0, "peak": 0}
        if self.device.type == "cuda":
            mem = {"allocated": torch.cuda.memory_allocated(self.device),
                   "reserved": torch.cuda.memory_reserved(self.device),
                   "peak": torch.cuda.max_memory_allocated(self.device)}
        loss = float(self.loss_sum.cpu())/self.sample_count if self.sample_count else None
        if loss is not None and not math.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        value = {"job": self.job["name"], "device": str(self.device), "stage": stage,
                 "epoch": min(self.epoch+1, self.config["epochs"]), "epochs": self.config["epochs"],
                 "batch": self.next_batch, "batches": self.batches,
                 "epoch_percent": 100*self.next_batch/self.batches,
                 "total_percent": 100*done/total, "lr": self.config["lr"], "train_mse": loss,
                 "current_validation": validation, "memory_bytes": mem, "timing": self.timing,
                 "eta_training_seconds": float(eta) if eta is not None else None,
                 "eta_training_kst": (dt.datetime.now(KST)+dt.timedelta(seconds=float(eta))).isoformat()
                                      if eta is not None else "측정 대기",
                 "eta_overall": "측정 대기: evaluation/plots/archive duration not yet measured",
                 "eta_exclusions": exclusions}
        json_write(self.run_dir / "status.json", value)
        with (self.run_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(value, timestamp_kst=dt.datetime.now(KST).isoformat()),
                                    allow_nan=False)+"\n")
        self.last_status = time.monotonic()

    def new_epoch(self):
        self.order = torch.randperm(self.train.length, generator=self.generator, device=self.device)
        # Each consecutive group of R batches visits every scale once, balanced to within one.
        scales = self.job["scales"]
        offset = self.epoch % len(scales)
        self.scale_order = [scales[(i+offset) % len(scales)] for i in range(self.batches)]
        self.next_batch = 0
        self.loss_sum.zero_()
        self.sample_count = 0
        self.order_history.append({"epoch": self.epoch+1,
                                   "order_sha256": tensor_hash({"indices": self.order}),
                                   "scales": self.scale_order})
        self.status("train")

    def run(self, stop_after_updates=None):
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: setattr(self, "stop_requested", True))
        local_updates = 0
        while self.epoch < self.config["epochs"]:
            if self.order is None:
                self.new_epoch()
            self.model.train()
            while self.next_batch < self.batches:
                if self.stop_requested or (self.run_dir/"STOP").exists():
                    self.save()
                    self.status("interrupted")
                    return False
                step_start = time.monotonic()
                events = None
                if self.device.type == "cuda":
                    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                    events[0].record()
                start = self.next_batch * self.config["batch_size"]
                ids = self.order[start:start+self.config["batch_size"]]
                scale = self.scale_order[self.next_batch]
                self.optimizer.zero_grad(set_to_none=True)
                effective_loss = torch.zeros((), device=self.device)
                for micro_ids in ids.split(self.config["micro_batch"]):
                    batch = self.train.batch(micro_ids, scale, self.job["legacy"])
                    with torch.autocast(self.device.type, enabled=self.config["amp"]):
                        prediction = predict(self.model, batch, self.job["legacy"])
                        loss = F.mse_loss(prediction.float(), batch["gt"].float())
                    fraction = len(micro_ids)/len(ids)
                    self.scaler.scale(loss * fraction).backward()
                    effective_loss.add_(loss.detach()*fraction)
                self.step_applied = False
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Optimizer hook detects skipped AMP updates without extra scale .item() calls.
                applied = self.step_applied
                self.attempted_updates += 1
                self.applied_updates += int(applied)
                self.loss_sum.add_(effective_loss*len(ids))
                self.sample_count += len(ids)
                self.exposures[str(scale)]["samples"] += len(ids)
                self.exposures[str(scale)]["target_pixels"] += len(ids)*batch["gt"].shape[-2]*batch["gt"].shape[-1]
                self.exposures[str(scale)]["updates"] += int(applied)
                self.next_batch += 1
                local_updates += 1
                if events:
                    events[1].record()
                    self.pending_events.append(events)
                else:
                    self.timing["train_seconds"] += time.monotonic()-step_start
                    self.timing["measured_steps"] += 1
                now = time.monotonic()
                if now-self.last_save >= self.config["checkpoint_seconds"]:
                    self.save()
                if now-self.last_status >= self.config["status_seconds"]:
                    self.status()
                if self.stop_requested or (self.run_dir/"STOP").exists() or (stop_after_updates and local_updates >= stop_after_updates):
                    self.save()
                    self.status("interrupted")
                    return False
            self.measure_pending()
            self.status("validation")
            begin = time.monotonic()
            metrics = evaluate(self.model, self.val, self.job["scales"], self.job["legacy"],
                               self.config["micro_batch"], self.config["amp"])
            synchronize(self.device)
            self.timing["val_seconds"].append(time.monotonic()-begin)
            val_mse = float(np.mean([v["scene_mean"]["MSE"] for v in metrics.values()]))
            train_mse = float(self.loss_sum.cpu())/self.sample_count
            if not math.isfinite(train_mse+val_mse):
                raise FloatingPointError("Nonfinite epoch loss")
            self.history.append({"epoch": self.epoch+1, "train_mse": train_mse,
                                 "validation": metrics, "selection_mse": val_mse})
            if val_mse < self.best_value:
                self.best_value, self.best_epoch = val_mse, self.epoch+1
                self.best_model = cpu_tree(self.model.state_dict())
            self.status("epoch_complete", {r: v["scene_mean"] for r, v in metrics.items()})
            self.epoch += 1
            self.next_batch, self.order, self.scale_order = 0, None, None
            self.loss_sum.zero_()
            self.sample_count = 0
            self.save_best()  # Rewrites the actual best; never replaces it with last weights.
            self.save()
            json_write(self.run_dir / "history.json", self.history)
            json_write(self.run_dir / "batch_orders.json", self.order_history)
            json_write(self.run_dir / "exposures.json", self.exposures)
        self.status("training_complete")
        return True
''',
'metrics_snapshot.py': r'''"""PanCollection RR/FR metrics, following DLPan-Toolbox definitions.

FR PAN resizing uses scikit-image's antialiased bicubic approximation to
MATLAB imresize; D_s/QNR must be parity-checked before a strict paper claim.
"""

import numpy as np
from scipy.ndimage import convolve, convolve1d
from skimage.transform import resize


def _chw(image):
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError("Expected finite CHW image")
    return array


def _onion(a, b):
    """Cayley-Dickson multiplication used by MATLAB onions_quality."""
    n = a.shape[-1]
    if n == 1:
        return a * b
    half = n // 2
    x, y = a[..., :half], a[..., half:]
    z, w = b[..., :half], b[..., half:]
    cy = np.concatenate((y[..., :1], -y[..., 1:]), axis=-1)
    cw = np.concatenate((w[..., :1], -w[..., 1:]), axis=-1)
    if n == 2:
        return np.concatenate((x * z - w * y, x * w + z * y), axis=-1)
    cx = np.concatenate((x[..., :1], -x[..., 1:]), axis=-1)
    return np.concatenate((_onion(x, z) - _onion(w, cy), _onion(cx, w) + _onion(z, y)), axis=-1)


def q2n(reference, fused, block=32):
    """Q4/Q8 with nonoverlapping 32-pixel blocks, matching q2n.m here."""
    reference, fused = _chw(reference), _chw(fused)
    if reference.shape != fused.shape:
        raise ValueError("Q2n shapes differ")
    channels, height, width = reference.shape
    if height % block or width % block or channels & (channels - 1):
        raise ValueError("Q2n requires block-aligned images and power-of-2 bands")
    def blocks(image):
        # MATLAB uint16 saturates/rounds values before the hypercomplex index.
        image = np.rint(np.clip(image, 0, 65535)).transpose(1, 2, 0)
        return image.reshape(height // block, block, width // block, block, channels).transpose(0, 2, 1, 3, 4).reshape(-1, block, block, channels)
    a, b = blocks(reference), blocks(fused)
    mean_a = a.mean(axis=(1, 2), keepdims=True)
    std_a = a.std(axis=(1, 2), ddof=1, keepdims=True)
    std_a = np.where(std_a == 0, np.finfo(float).eps, std_a)
    a = (a - mean_a) / std_a + 1
    b = (b - mean_a) / std_a + 1
    b[..., 1:] *= -1
    m1, m2 = a.mean(axis=(1, 2)), b.mean(axis=(1, 2))
    m1_sq, m2_sq = np.sum(m1 * m1, axis=-1), np.sum(m2 * m2, axis=-1)
    bias = np.divide(2 * np.sqrt(m1_sq * m2_sq), m1_sq + m2_sq,
                     out=np.zeros_like(m1_sq), where=(m1_sq + m2_sq) != 0)
    n = block * block
    variance_term = n / (n - 1) * (
        np.mean(np.sum(a * a + b * b, axis=-1), axis=(1, 2)) - m1_sq - m2_sq
    )
    quality = np.empty((len(a), channels), dtype=np.float64)
    nonzero = np.abs(variance_term) > 1e-12
    quality[~nonzero] = 0
    quality[~nonzero, -1] = bias[~nonzero]
    if np.any(nonzero):
        product_mean = _onion(a[nonzero], b[nonzero]).mean(axis=(1, 2))
        mean_product = _onion(m1[nonzero], m2[nonzero])
        quality[nonzero] = (product_mean - mean_product) * (n / (n - 1) * bias[nonzero] * 2 / variance_term[nonzero])[:, None]
    return float(np.linalg.norm(quality, axis=-1).mean())


def rr_metrics(reference, fused, *, ratio=4):
    """Return per-image RR metrics, not an image-concatenated score."""
    reference, fused = _chw(reference), _chw(fused)
    if reference.shape != fused.shape:
        raise ValueError("RR shapes differ")
    diff = fused - reference
    dot = np.sum(reference * fused, axis=0)
    norms = np.linalg.norm(reference, axis=0) * np.linalg.norm(fused, axis=0)
    valid = norms > 0
    sam = float(np.degrees(np.arccos(np.clip(dot[valid] / norms[valid], -1, 1))).mean())
    means = reference.mean(axis=(1, 2))
    band_mse = np.mean(diff * diff, axis=(1, 2))
    ergas = float(100 / ratio * np.sqrt(np.mean(band_mse / np.maximum(means * means, 1e-30))))
    # MATLAB SCC: Sobel gradient magnitudes on a 1-pixel cropped image.
    sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    def gradients(image):
        image = image[:, 1:-1, 1:-1]
        gy = np.stack([convolve(band, sobel, mode="constant") for band in image])
        gx = np.stack([convolve(band, sobel.T, mode="constant") for band in image])
        return np.hypot(gx, gy)
    a, b = gradients(reference), gradients(fused)
    scc = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    # The paper does not specify its PSNR convention. Use the empirically
    # closest documented choice: per-band PSNR at peak 2047, then average.
    psnr = float(np.mean(10 * np.log10(2047 * 2047 / np.maximum(band_mse, 1e-30))))
    return {"SAM": sam, "ERGAS": ergas, "PSNR": psnr,
            "SCC": scc, "Q2n": q2n(reference, fused)}


def _block_uqi(a, b, block=32):
    """Average MATLAB uqi over nonoverlapping blocks."""
    height, width = a.shape
    if a.shape != b.shape or height % block or width % block:
        raise ValueError("UQI shapes are not block aligned")
    def group(image):
        return image.reshape(height // block, block, width // block, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    x, y = group(a), group(b)
    mx, my = x.mean(axis=1), y.mean(axis=1)
    vx, vy = x.var(axis=1, ddof=1), y.var(axis=1, ddof=1)
    cov = np.sum((x - mx[:, None]) * (y - my[:, None]), axis=1) / (x.shape[1] - 1)
    denominator = (vx + vy) * (mx * mx + my * my)
    quality = np.divide(4 * cov * mx * my, denominator,
                        out=np.zeros_like(mx), where=np.abs(denominator) > 1e-30)
    return float(quality.mean())


def _interp23tap(image, ratio=4):
    coeff = 2 * np.array([.5, .305334091185, 0, -.072698593239, 0, .021809577942,
                          0, -.005192756653, 0, .000807762146, 0, -.000060081482])
    coeff = np.r_[coeff[:0:-1], coeff]
    for stage in range(int(np.log2(ratio))):
        up = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
        offset = 1 if stage == 0 else 0
        up[offset::2, offset::2] = image
        image = convolve1d(convolve1d(up, coeff, axis=0, mode="wrap"), coeff, axis=1, mode="wrap")
    return image


def fr_metrics(fused, lms, ms, pan, *, ratio=4, block=32):
    """Toolbox-style QNR; PAN antialias resize is not MATLAB-bit-exact."""
    fused, lms, ms = _chw(fused), _chw(lms), _chw(ms)
    pan = np.asarray(pan, dtype=np.float64).squeeze()
    if fused.shape != lms.shape or pan.shape != fused.shape[1:] or ms.shape[1:] != (pan.shape[0] // ratio, pan.shape[1] // ratio):
        raise ValueError("FR input shapes differ")
    channels = fused.shape[0]
    spectral = [abs(_block_uqi(fused[i], fused[j], block) - _block_uqi(lms[i], lms[j], block))
                for i in range(channels) for j in range(i + 1, channels)]
    down = resize(pan, ms.shape[1:], order=3, mode="reflect", anti_aliasing=True, preserve_range=True)
    pan_filtered = _interp23tap(down, ratio)
    spatial = [abs(_block_uqi(fused[i], pan, block) - _block_uqi(lms[i], pan_filtered, block))
               for i in range(channels)]
    d_lambda, d_s = float(np.mean(spectral)), float(np.mean(spatial))
    return {"D_lambda": d_lambda, "D_s": d_s, "QNR": (1 - d_lambda) * (1 - d_s)}
''',
'model.py': r'''"""Same official SSA blocks; independent input ratio and internal pyramid."""
import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from .reference import ReferenceX4

SCALES = (2, 3, 4)

@lru_cache(maxsize=64)
def _axis_weights(length, target, align_corners, antialias, dtype, device):
    # CPU native interpolation of basis vectors gives the exact linear operator,
    # including the native edge normalization. Never approximate AA with another blur.
    basis = torch.eye(length, dtype=torch.float64).reshape(1, length, 1, length)
    weights = F.interpolate(basis, size=(1, target), mode="bilinear",
                            align_corners=align_corners, antialias=antialias)
    return weights[0, :, 0, :].t().contiguous().to(device=device, dtype=dtype)

class _DeterministicResize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, height, width, align_corners, antialias):
        ctx.input_size = tuple(x.shape[-2:])
        ctx.align_corners, ctx.antialias = align_corners, antialias
        # Native forward is retained; only its CUDA scatter-based backward is replaced.
        return F.interpolate(x, size=(height, width), mode="bilinear",
                             align_corners=align_corners, antialias=antialias)

    @staticmethod
    def backward(ctx, gradient):
        h, w = ctx.input_size
        oh, ow = gradient.shape[-2:]
        wy = _axis_weights(h, oh, ctx.align_corners, ctx.antialias, gradient.dtype, gradient.device)
        wx = _axis_weights(w, ow, ctx.align_corners, ctx.antialias, gradient.dtype, gradient.device)
        # Dense transpose products avoid CUDA scatter/atomic accumulation. CUBLAS
        # workspace configuration is set before worker CUDA initialization by runner.
        with torch.autocast(device_type=gradient.device.type, enabled=False):
            result = torch.matmul(torch.matmul(wy.t(), gradient), wx)
        return result, None, None, None, None

def deterministic_bilinear(x, size, align_corners=False, antialias=False):
    return _DeterministicResize.apply(x, *tuple(size), align_corners, antialias)

class DeterministicUpsample(nn.Upsample):
    """Reference-path resize with identical native forward and strict backward."""
    def forward(self, x):
        if x.requires_grad and torch.are_deterministic_algorithms_enabled():
            if self.size is not None:
                size = self.size
            else:
                factor = self.scale_factor
                factors = (factor, factor) if isinstance(factor, (int,float)) else factor
                size = tuple(int(s*f) for s,f in zip(x.shape[-2:], factors))
            return deterministic_bilinear(x, size, align_corners=self.align_corners)
        return super().forward(x)

def resize(x, size):
    size = tuple(size)
    if tuple(x.shape[-2:]) == size:
        return x
    antialias = any(a > b for a, b in zip(x.shape[-2:], size))
    if x.requires_grad and torch.are_deterministic_algorithms_enabled():
        return deterministic_bilinear(x, size, antialias=antialias)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=antialias)

class ScaleFiLM(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 2 * channels))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, scale):
        condition = x.new_full((x.shape[0], 1), math.log2(scale))
        gamma, beta = self.net(condition).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

class VariableScaleNet(ReferenceX4):
    def __init__(self, channels=4, conditioned=False):
        super().__init__(channels=channels, ssai_dimension=4)
        self.conditioned = conditioned
        # Created AFTER the reference layers: common initialization is identical.
        self.film = ScaleFiLM(64) if conditioned else nn.Identity()

    def forward(self, pan, lms, ms, scale):
        if not isinstance(scale, int) or isinstance(scale, bool) or scale not in SCALES:
            raise ValueError("Supported integer scales are 2, 3, 4")
        if any(x.ndim != 4 for x in (pan, lms, ms)):
            raise ValueError("Expected NCHW tensors")
        n, c, h, w = lms.shape
        if c != self.channels or min(h, w) < 12 or h % scale or w % scale:
            raise ValueError("Target dimensions must be divisible by input scale and >=12")
        if pan.shape != (n, 1, h, w) or ms.shape != (n, c, h // scale, w // scale):
            raise ValueError("PAN/LMS/MS shapes disagree with scale")
        half, quarter = (h // 2, w // 2), (h // 4, w // 4)
        # Original PAN low-pass context is at internal quarter scale, not input scale.
        context = self.prelu(self.conv1t1(resize(resize(pan, quarter), (h, w))))
        features = self.backbone(self.cov2t64(torch.cat((lms, pan, context), dim=1)))
        if self.conditioned:
            features = self.film(features, scale)
        res = self.conv64t48(features) + lms
        res = res + self._attend(pan, res, lms, self.SSA_blocks, self.conv48tnum1, self.prelu1)
        res_half, ms_half, pan_half = resize(res, half), resize(ms, half), resize(pan, half)
        res_half = res_half + self._attend(pan_half, res_half, ms_half,
                                         self.SSA_blocks1, self.conv48tnum2, self.prelu2)
        ms_small = resize(ms, quarter)
        attended = self._attend(resize(pan, quarter), resize(res_half, quarter), ms_small,
                                self.SSA_blocks2, self.conv48tnum3, self.prelu3)
        out = self.conv2ctc1(torch.cat((ms_small, attended), dim=1))
        out = self.conv2ctc2(torch.cat((resize(out, half), res_half), dim=1))
        return self.conv2ctc3(torch.cat((resize(out, (h, w)), res), dim=1))
''',
'reference.py': r'''"""Execute preserved source snapshots; never initialize or edit the submodule."""
import ast
import importlib.util
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
provenance = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
for filename, key in (("upstream_snapshot.py", "upstream_sha256"),
                      ("adapter_snapshot.py", "adapter_sha256"),
                      ("metrics_snapshot.py", "metrics_sha256")):
    if hashlib.sha256((HERE / filename).read_bytes()).hexdigest() != provenance[key]:
        raise ValueError(f"Pinned source snapshot checksum mismatch: {filename}")
spec = importlib.util.spec_from_file_location("variable_scale_upstream", HERE / "upstream_snapshot.py")
upstream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)
tree = ast.parse((HERE / "adapter_snapshot.py").read_text(encoding="utf-8"))
# Only substitute source discovery. The original restored class is unchanged.
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.FunctionDef) and node.name == "_upstream_model")]
namespace = {"__file__": str(HERE / "adapter_snapshot.py"), "__name__": "variable_scale.reference_adapter",
             "_upstream_model": lambda: upstream.PansharpeningNet}
exec(compile(tree, str(HERE / "adapter_snapshot.py"), "exec"), namespace)
ReferenceX4 = namespace["RestoredPansharpeningNet"]
''',
'reporting.py': r'''"""Actual curves, fixed panels, timing measurements, manifests and archive."""
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import ResidentSplit, sha_file
from .engine import evaluate, json_write, predict, synchronize
from .metrics_snapshot import fr_metrics

def select_examples(manifest, count=5, seed=42):
    entries = manifest["files"]["RR"]["samples"]
    representatives = {}
    for i, entry in enumerate(entries):
        representatives.setdefault(entry["scene_id"], i)
    if len(representatives) < count:
        raise ValueError(f"RR needs {count} distinct scenes (or explicitly unverified samples)")
    candidates = sorted(representatives.values())
    ids = sorted(np.random.default_rng(seed).choice(candidates, count, replace=False).tolist())
    return {"seed": seed, "indices": ids, "samples": [dict(entries[i], index=i) for i in ids],
            "rule": "one first sample per scene, seeded choice then ascending index",
            "scene_identity_verified": manifest["files"]["RR"]["scene_provenance_present"],
            "tile": "whole supplied H5 sample; synthetic target uses documented common crop",
            "display_order": ["LR MS", "PAN input", "prediction", "GT"],
            "rgb_bands": [2, 1, 0], "fixed_before_training_and_test_scoring": True}

def panel(batch, prediction, path, title):
    gt = batch["gt"][0].float().cpu().numpy()
    ms = batch["ms"][0:1]
    nearest = torch.nn.functional.interpolate(ms, size=gt.shape[-2:], mode="nearest")[0].cpu().numpy()
    pred = prediction[0].float().cpu().numpy()
    guide = batch["pan"][0, 0].float().cpu().numpy()
    low, high = np.percentile(gt[[2, 1, 0]], [2, 98])
    def rgb(x):
        return np.clip((x[[2, 1, 0]].transpose(1, 2, 0)-low)/max(high-low, 1e-8), 0, 1)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, value, label in zip(axes, [rgb(nearest), guide, rgb(pred), rgb(gt)],
                               ["LR MS (nearest)", "PAN input", "prediction", "GT"]):
        ax.imshow(value, cmap="gray" if label == "PAN input" else None)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    np.save(Path(path).with_suffix(".prediction.npy"), pred)

def history_plot(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    epochs = [r["epoch"] for r in history]
    axes[0].plot(epochs, [r["train_mse"] for r in history], label="train")
    for scale in history[0]["validation"]:
        for ax, metric in zip(axes, ["MSE", "PSNR", "SAM"]):
            ax.plot(epochs, [r["validation"][scale]["scene_mean"][metric] for r in history], label=f"val x{scale}")
    for ax, metric in zip(axes, ["MSE", "PSNR", "SAM"]):
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def benchmark(model, batch, legacy, device, amp, repetitions=5):
    model.eval()
    with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=amp):
        for _ in range(3):
            predict(model, batch, legacy)
        synchronize(device)
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(repetitions):
            predict(model, batch, legacy)
        synchronize(device)
        elapsed = (time.perf_counter()-start)/repetitions
    return {"seconds_per_forward": elapsed, "batch_size": len(batch["pan"]),
            "target_shape": list(batch["pan"].shape), "ms_shape": list(batch["ms"].shape),
            "warmup": 3, "repetitions": repetitions, "cuda_synchronized": torch.device(device).type == "cuda",
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None,
            "memory_includes_resident_training_data": True}

def finish_job(trainer, selection, provisional_fr=False):
    output, model, job = trainer.run_dir, trainer.model, trainer.job
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint["signature"] != trainer.signature:
        raise ValueError("Best checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    trainer.status("RR scoring")
    begin = time.monotonic()
    rr = ResidentSplit(trainer.manifest["files"]["RR"], trainer.device)
    result = evaluate(model, rr, job["scales"], job["legacy"], trainer.config["micro_batch"], trainer.config["amp"])
    rr_seconds = time.monotonic()-begin
    json_write(output / "rr_metrics.json", {"protocol": "native_x4" if job["legacy"] else "synthetic_proxy",
               "scene_split_verified": trainer.manifest["scene_split_verified"],
               "prediction_clipping": "none", "border_exclusion": 0, "psnr_peak": 1,
               "aggregation": "sample metrics -> scene mean -> equally weighted scene mean", "by_scale": result})
    measurements = {}
    with torch.inference_mode():
        for scale in job["scales"]:
            batch = rr.batch(torch.arange(1, device=rr.device), scale, job["legacy"])
            measurements[str(scale)] = benchmark(model, batch, job["legacy"], rr.device, trainer.config["amp"])
            for idx in selection["indices"]:
                sample = rr.batch(torch.tensor([idx], device=rr.device), scale, job["legacy"])
                with torch.autocast(rr.device.type, enabled=trainer.config["amp"]):
                    pred = predict(model, sample, job["legacy"])
                panel(sample, pred, output / f"rr_x{scale}_sample_{idx:03d}.png",
                      f"{job['name']} x{scale}, sample {idx}; display RGB bands assumed [2,1,0]")
    trainer.status("FR assessment")
    fr_result = {"status": "not_evaluated", "reason": "QNR/D_s MATLAB parity not verified; "
                 "only actual native x4 is eligible. No GT/PSNR is fabricated.", "ratio": 4,
                 "unseen_training_ratio": 4 not in job["scales"]}
    if provisional_fr:
        fr = ResidentSplit(trainer.manifest["files"]["FR"], trainer.device)
        rows = []
        with torch.inference_mode():
            for i in range(fr.length):
                ids = torch.tensor([i], device=fr.device)
                # Actual native sensor observations, never synthetic GT-based FR inputs.
                batch = {k: v.index_select(0, ids) for k, v in fr.tensors.items()}
                batch["scale"] = 4
                with torch.autocast(fr.device.type, enabled=trainer.config["amp"]):
                    prediction = predict(model, batch, job["legacy"])
                arrays = {k: v[0].float().cpu().numpy()*2047 for k, v in batch.items() if isinstance(v, torch.Tensor)}
                metrics = fr_metrics(prediction[0].float().cpu().numpy()*2047,
                                     arrays["lms"], arrays["ms"], arrays["pan"], ratio=4)
                rows.append({"sample_index": i, "scene_id": fr.entries[i]["scene_id"], "metrics": metrics})
        fr_result.update(status="provisional_full_x4", metrics=scene_means(rows),
                         reason="Existing QNR implementation; MATLAB parity UNVERIFIED")
    json_write(output / "fr_metrics.json", fr_result)
    trainer.status("plots and manifest")
    history_plot(trainer.history, output / "learning.png")
    json_write(output / "costs.json", {"parameters": sum(p.numel() for p in model.parameters()),
               "device": str(trainer.device), "device_name": torch.cuda.get_device_name(trainer.device)
                          if trainer.device.type == "cuda" else "CPU explicit smoke",
               "torch": torch.__version__, "amp": trainer.config["amp"], "timing": trainer.timing,
               "rr_evaluation_seconds": rr_seconds, "inference": measurements,
               "optimizer_attempts": trainer.attempted_updates, "optimizer_applied": trainer.applied_updates,
               "effective_batch": trainer.config["batch_size"], "micro_batch": trainer.config["micro_batch"],
               "exposures": trainer.exposures,
               "checkpoint_bytes": {name: (output/name).stat().st_size for name in ("best.pt", "latest.pt")}})
    hashes = {p.name: sha_file(p) for p in output.iterdir() if p.is_file() and p.name not in
              ("artifact_manifest.json", "status.json", "progress.jsonl")}
    json_write(output / "artifact_manifest.json", {"hashes": hashes, "best_epoch": trainer.best_epoch,
               "latest_complete_epoch": trainer.epoch, "code_config_data_signature": trainer.signature,
               "training_complete": True, "rr_complete": True,
               "fr_validated_complete": False, "remote_storage_verified": False,
               "smoke_only": trainer.config.get("smoke", False)})
    trainer.status("complete_for_requested_scope")

def suite_summary(output, jobs):
    output = Path(output)
    compare, init_hashes, order_hashes, costs = [], {}, {}, {}
    for job in jobs:
        run = output / "runs" / job["name"]
        init_hashes[job["name"]] = json.loads((run/"initialization.json").read_text())["common_parameter_sha256"]
        order_hashes[job["name"]] = [r["order_sha256"] for r in json.loads((run/"batch_orders.json").read_text())]
        costs[job["name"]] = json.loads((run/"costs.json").read_text())
        for scale, data in json.loads((run/"rr_metrics.json").read_text())["by_scale"].items():
            if not job["legacy"]:
                compare.append({"job": job["name"], "scale": int(scale), "metrics": data["scene_mean"]})
    rows = []
    for row in compare:
        baseline = next((r for r in compare if r["job"] == f"B_x{row['scale']}" and r["scale"] == row["scale"]), None)
        rows.append(dict(row, baseline=baseline["job"] if baseline else None,
                         delta={k: row["metrics"][k]-baseline["metrics"][k] for k in row["metrics"]} if baseline else None))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, metric in zip(axes, ["PSNR", "SAM", "MSE"]):
        for name in sorted({r["job"] for r in compare}):
            points = sorted([r for r in compare if r["job"] == name], key=lambda r: r["scale"])
            ax.plot([r["scale"] for r in points], [r["metrics"][metric] for r in points], "o-", label=name)
        ax.set_title(f"Synthetic RR {metric}")
        ax.set_xlabel("ratio (compare models WITHIN each ratio)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "test_comparison.png", dpi=130)
    plt.close(fig)
    applied = {name: item["optimizer_applied"] for name, item in costs.items()}
    json_write(output / "comparison.json", {"rows": rows,
               "common_initialization_matches": len(set(init_hashes.values())) == 1,
               "epoch_index_order_matches": len({tuple(v) for v in order_hashes.values()}) == 1,
               "initial_hashes": init_hashes, "order_hashes": order_hashes,
               "applied_updates": applied, "equal_applied_updates": len(set(applied.values())) == 1,
               "scope": "A native x4 excluded from synthetic B/C/D deltas",
               "dedicated_total_train_seconds": sum(v["timing"]["train_seconds"] for k,v in costs.items() if k.startswith("B_")),
               "mixed_train_seconds": {k:v["timing"]["train_seconds"] for k,v in costs.items() if k.startswith(("C_", "D_"))}})

def archive_suite(output):
    output = Path(output)
    files = [p for p in output.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and not p.name.endswith((".tmp", ".lock")) and p.name != "suite_manifest.json"]
    manifest = {p.relative_to(output).as_posix(): sha_file(p) for p in files}
    json_write(output / "suite_manifest.json", {"hashes": manifest, "remote_storage_verified": False,
               "status": "local_package_created; FR validation and remote recovery not verified"})
    files.append(output / "suite_manifest.json")
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(output.parent))
    # Verify every uncompressed ZIP member against the local manifest.
    import hashlib
    with zipfile.ZipFile(archive) as z:
        for rel, digest in manifest.items():
            if hashlib.sha256(z.read(output.name+"/"+rel)).hexdigest() != digest:
                raise ValueError("Archive member checksum mismatch")
    json_write(archive.with_suffix(".zip.sha256.json"), {"path": str(archive), "sha256": sha_file(archive),
               "members_verified": True, "remote_storage_verified": False})
    return archive
''',
'runner.py': r'''"""Explicit fresh/resume suites and two independent GPU work queues."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import h5py
import numpy as np
import torch
from .data import canonical_hash, inspect_sources, sha_file
from .engine import KST, Trainer, file_lock, json_write
from .reporting import archive_suite, finish_job, select_examples, suite_summary

DATA_ROOT = "/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset"
DEFAULT_PATHS = {"train": f"{DATA_ROOT}/QuickBird__Training_Dataset__train_qb.h5",
                 "val": f"{DATA_ROOT}/QuickBird__Training_Dataset__valid_qb.h5",
                 "RR": f"{DATA_ROOT}/QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5",
                 "FR": f"{DATA_ROOT}/QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5"}
DEFAULT_TRAIN = {"epochs": 100, "batch_size": 32, "micro_batch": 32, "lr": 1e-4,
                 "seed": 42, "amp": False, "deterministic": True,
                 "checkpoint_seconds": 60, "status_seconds": 15, "cpu_threads": 2, "smoke": False}
JOBS = [{"name": "A_x4", "legacy": True, "conditioned": False, "scales": [4]},
        {"name": "B_x2", "legacy": False, "conditioned": False, "scales": [2]},
        {"name": "B_x3", "legacy": False, "conditioned": False, "scales": [3]},
        {"name": "B_x4", "legacy": False, "conditioned": False, "scales": [4]},
        {"name": "C_mixed", "legacy": False, "conditioned": False, "scales": [2, 3, 4]},
        {"name": "D_mixed", "legacy": False, "conditioned": True, "scales": [2, 3, 4]}]

def source_files():
    root = Path(__file__).resolve().parent
    return {p.name: p.read_bytes() for p in sorted(root.iterdir()) if p.suffix in (".py", ".json")}

def source_identity(files):
    import hashlib
    return canonical_hash({name: hashlib.sha256(value).hexdigest() for name, value in files.items()})

def smoke_data(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(9214)
    provenance, paths = {}, {}
    for split, count in [("train", 6), ("val", 3), ("RR", 5), ("FR", 5)]:
        gt = rng.uniform(10, 2000, (count, 4, 24, 24)).astype(np.float32)
        pan = gt.mean(axis=1, keepdims=True)
        ms = gt.reshape(count, 4, 6, 4, 6, 4).mean((3, 5))
        lms = np.repeat(np.repeat(ms, 4, axis=2), 4, axis=3)
        path = root / f"{split}.h5"
        with h5py.File(path, "w") as f:
            for key, value in {"pan": pan, "ms": ms, "lms": lms}.items():
                f.create_dataset(key, data=value)
            if split != "FR":
                f.create_dataset("gt", data=gt)
        paths[split] = str(path.resolve())
        provenance[split] = [{"scene_id": f"fixture-{split}-{i}", "tile": "artificial 24x24 fixture"}
                             for i in range(count)]
    scene_path = root / "scene_manifest.json"
    json_write(scene_path, provenance)
    return paths, str(scene_path)

def validate_training(config):
    if (config["epochs"] < 1 or config["batch_size"] < 1 or config["micro_batch"] < 1
        or config["micro_batch"] > config["batch_size"] or config["lr"] <= 0
        or config["checkpoint_seconds"] <= 0 or config["status_seconds"] <= 0):
        raise ValueError("Invalid training configuration")

def prepare_output(output, files, resume=False, resume_input=None):
    output = Path(output).resolve()
    source = Path(resume_input).resolve() if resume_input else None
    if source:
        if not resume or not (source/"suite_identity.json").is_file():
            raise ValueError("Saved Output resume needs explicit resume and a suite_identity.json")
        if output.exists():
            raise FileExistsError("Import Saved Output into a fresh OUTPUT_DIR")
        shutil.copytree(source, output, ignore=shutil.ignore_patterns("__pycache__", "*.tmp"))
    elif not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not (output/"suite_identity.json").is_file():
        raise FileNotFoundError("Explicit resume requested but existing suite not found")
    if resume:
        expected = json.loads((output/"suite_identity.json").read_text(encoding="utf-8"))
        if expected["source_hash"] != source_identity(files):
            raise ValueError("Code changed; a new experiment directory is required")
    code = output / "code" / "variable_scale"
    code.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = code/name
        if target.exists() and target.read_bytes() != content:
            raise ValueError("Source snapshot mismatch")
        if not target.exists():
            target.write_bytes(content)
    return output

def worker(spec):
    run = Path(spec["run_dir"])
    with file_lock(run.with_suffix(".lock")):
        return _worker(spec)

def _worker(spec):
    run = Path(spec["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    try:
        trainer = Trainer(spec["config"], spec["job"], spec["manifest"], run,
                          spec["device"], spec["source_hash"], spec["resume"])
        if spec.get("evaluate_only"):
            if trainer.epoch != spec["config"]["epochs"] or trainer.best_model is None:
                raise ValueError("Evaluation requires a fully trained matching checkpoint")
        elif not trainer.run():
            return 2
        finish_job(trainer, spec["selection"], spec["provisional_fr"])
        return 0
    except BaseException as error:
        import traceback
        json_write(run / "failure.json", {"type": type(error).__name__, "message": str(error),
                   "traceback": traceback.format_exc()})
        raise

def dashboard(queues, active, completed, output, suite_started):
    lines = ["SSA-MRN variable scale | 현재 에포크만 표시 | " + dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")]
    for device, queue in queues.items():
        state = active.get(device)
        if state:
            job, process, _ = state
            status_path = output / "runs" / job["name"] / "status.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                status = None
            if status:
                mem = status["memory_bytes"]
                lines.append(f"{device} {job['name']} | {status['stage']} | "
                             f"epoch {status['epoch']}/{status['epochs']} | batch {status['batch']}/{status['batches']} | "
                             f"에포크 {status['epoch_percent']:.1f}% / 학습 {status['total_percent']:.1f}%")
                loss = status["train_mse"]
                lines.append(f"  lr={status['lr']:.3g} loss={loss if loss is not None else '측정 대기'} | "
                             f"val(current epoch)={status['current_validation']}")
                lines.append("  GPU allocated/reserved/peak=" + "/".join(f"{mem[k]/1024**3:.2f}GiB" for k in ("allocated","reserved","peak")))
                remaining = status["eta_training_seconds"]
                lines.append(f"  남은 학습 시간: {str(dt.timedelta(seconds=round(remaining))) if remaining is not None else '측정 대기'} | "
                             f"학습 종료 예상: {status['eta_training_kst']}")
                lines.append("  전체 종료 예상: " + status["eta_overall"])
                lines.append("  ETA 제외 범위: " + ", ".join(status["eta_exclusions"]))
            else:
                lines.append(f"{device} {job['name']} | 시작/데이터 GPU 상주 준비 | ETA 측정 대기")
        else:
            lines.append(f"{device} | idle")
        # Pending models have not been measured: do not assign invented durations.
        lines.append("  GPU 대기 큐: " + (", ".join(j["name"] for j in queue) or "없음") + " | 큐 포함 종료 예상: 미측정 작업이 있으면 측정 대기")
    lines.append(f"suite 완료 작업 {len(completed)} | elapsed {time.monotonic()-suite_started:.1f}s")
    return "\n".join(lines)

def run_suite(settings, notebook=False):
    with file_lock(Path(settings["output_dir"]).resolve().with_suffix(".lock")):
        return _run_suite(settings, notebook)

def _run_suite(settings, notebook=False):
    mode = settings.get("mode", "smoke")
    if mode not in ("smoke", "train", "evaluate"):
        raise ValueError("mode must be smoke/train/evaluate")
    output_path = settings["output_dir"]
    resume = settings.get("resume", False) or mode == "evaluate"
    if mode == "smoke" and resume:
        raise ValueError("Smoke suites are fresh-only; use train/evaluate for real resumable runs")
    config = dict(DEFAULT_TRAIN, **settings.get("training", {}))
    config["smoke"] = mode == "smoke"
    if mode == "smoke":
        config.update(epochs=2, batch_size=2, micro_batch=2, amp=False, status_seconds=.5)
    validate_training(config)
    devices = settings.get("devices", ["cuda:0", "cuda:1"])
    if devices == ["cpu"]:
        if mode != "smoke":
            raise ValueError("CPU is explicit local smoke only; no training fallback")
    else:
        if devices != ["cuda:0", "cuda:1"] or torch.cuda.device_count() < 2:
            raise RuntimeError("Requires two CUDA GPUs (T4 x2); no silent CPU fallback")
        props = [torch.cuda.get_device_properties(i) for i in (0,1)]
        if (props[0].name, props[0].total_memory) != (props[1].name, props[1].total_memory):
            raise RuntimeError("Paired GPU comparison requires equivalent devices")
    files = source_files()
    output = prepare_output(output_path, files, resume, settings.get("resume_input"))
    if mode == "smoke":
        paths, provenance = smoke_data(output / "fixture_data")
    else:
        paths, provenance = settings.get("paths", DEFAULT_PATHS), settings.get("scene_manifest")
    manifest = inspect_sources(paths, provenance, settings.get("allow_unverified_splits", False))
    selection = select_examples(manifest)
    names = settings.get("job_names", [job["name"] for job in JOBS])
    if not names or len(set(names)) != len(names) or set(names)-{job["name"] for job in JOBS}:
        raise ValueError("Select unique job names from the six defined A/B/C/D arms")
    jobs = [job for job in JOBS if job["name"] in names]
    identity_data = json.loads(json.dumps(manifest))
    for item in identity_data["files"].values():
        item.pop("path", None)
    identity = {"source_hash": source_identity(files), "config": config, "jobs": jobs,
                "data": identity_data, "selection": selection,
                "provisional_fr": settings.get("provisional_fr", False)}
    if resume and json.loads((output/"suite_identity.json").read_text(encoding="utf-8")) != identity:
        raise ValueError("Suite config/data/selection changed: use a new OUTPUT_DIR")
    json_write(output / "suite_identity.json", identity)
    json_write(output / "data_manifest.json", manifest)
    json_write(output / "selection.json", selection)
    json_write(output / "settings.json", settings)
    queues = {device: jobs[i::len(devices)] for i, device in enumerate(devices)}
    json_write(output / "gpu_queues.json", {k: [j["name"] for j in v] for k,v in queues.items()})
    active, completed = {}, []
    suite_started = time.monotonic()
    display_handle = None
    env = dict(os.environ, PYTHONPATH=str(output/"code"), PYTHONIOENCODING="utf-8",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    failure = None
    try:
        while any(queues.values()) or active:
            for device, queue in queues.items():
                if device not in active and queue:
                    job = queue.pop(0)
                    run = output / "runs" / job["name"]
                    run.mkdir(parents=True, exist_ok=True)
                    latest_exists = (run/"latest.pt").is_file()
                    if resume and not latest_exists and (run/"identity.json").exists():
                        raise FileNotFoundError(f"Missing latest for already-started job {job['name']}")
                    # Clear this parent's prior cooperative stop before spawning, not
                    # inside the child where a fresh parent stop could be lost.
                    (run/"STOP").unlink(missing_ok=True)
                    # Unstarted queue entries in a valid resumed suite legitimately start fresh.
                    spec = {"config": config, "job": job, "manifest": manifest, "run_dir": str(run),
                            "device": device, "source_hash": identity["source_hash"],
                            "resume": resume and latest_exists, "selection": selection,
                            "evaluate_only": mode == "evaluate", "provisional_fr": identity["provisional_fr"]}
                    spec_path = run / "worker_spec.json"
                    json_write(spec_path, spec)
                    log = (run/"worker.log").open("a", encoding="utf-8")
                    process = subprocess.Popen([sys.executable, "-m", "variable_scale.runner", "--worker", str(spec_path)],
                                               stdout=log, stderr=subprocess.STDOUT, env=env)
                    active[device] = (job, process, log)
            for device, (job, process, log) in list(active.items()):
                result = process.poll()
                if result is not None:
                    log.close()
                    del active[device]
                    if result != 0:
                        failure_path = output/"runs"/job["name"]/"failure.json"
                        detail = failure_path.read_text(encoding="utf-8") if failure_path.exists() else f"worker exit={result}; inspect worker.log"
                        failure = RuntimeError(f"{device} {job['name']} failed: {detail}")
                        raise failure
                    completed.append(job["name"])
            text = dashboard(queues, active, completed, output, suite_started)
            if notebook:
                from IPython.display import display
                if display_handle is None:
                    display_handle = display({"text/plain": text}, raw=True, display_id=True)
                else:
                    display_handle.update({"text/plain": text}, raw=True)
            else:
                # Overwrite a terminal screen; noninteractive runs can use the status JSON directly.
                print("\033[2J\033[H" + text, end="", flush=True)
            time.sleep(1 if mode == "smoke" else 5)
        suite_summary(output, jobs)
        archive = archive_suite(output)
        json_write(output/"execution_status.json", {"implementation_run_complete": True,
                   "smoke_only": mode == "smoke", "training_complete": mode != "smoke",
                   "validated_fr_complete": False, "remote_archive_verified": False,
                   "finished_kst": dt.datetime.now(KST).isoformat(),
                   "suite_seconds": time.monotonic()-suite_started, "archive": str(archive)})
        # Include final execution status in the archive and reverify checksums.
        archive = archive_suite(output)
        final_text = (f"{'Smoke 기능 검증' if mode == 'smoke' else '학습·RR 실행'} 완료: {output}\n"
                      f"ZIP: {archive}\nFR 지표 검증·원격 보관 검증은 미완료. "
                      "smoke 수치는 성능 개선 근거가 아닙니다.")
        if display_handle:
            display_handle.update({"text/plain": final_text}, raw=True)
        else:
            print("\n" + final_text)
        return output
    finally:
        for job, process, _ in active.values():
            if process.poll() is None:
                # Portable cooperative stop: Windows Popen does not support SIGINT.
                (output/"runs"/job["name"]/"STOP").touch()
        # Workers save at the next safe boundary; parent never force-kills a valid checkpoint.
        for _, process, log in active.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            log.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=["smoke", "train", "evaluate"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--jobs", nargs="+", choices=[job["name"] for job in JOBS])
    args = parser.parse_args()
    if args.worker:
        sys.exit(worker(json.loads(args.worker.read_text(encoding="utf-8"))))
    settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if args.mode:
        settings["mode"] = args.mode
    if args.output:
        settings["output_dir"] = str(args.output.resolve())
    if args.device:
        settings["devices"] = ["cpu"] if args.device == "cpu" else ["cuda:0", "cuda:1"]
    if args.jobs:
        settings["job_names"] = args.jobs
    if "output_dir" not in settings:
        parser.error("--output or config output_dir is required")
    run_suite(settings)

if __name__ == "__main__":
    main()
''',
'scoring.py': r'''"""Normalized QB RR metrics, with explicit ratio and no prediction clipping."""
import numpy as np
from skimage.metrics import structural_similarity

def rr_score(reference, prediction, scale):
    a, b = (np.asarray(x, dtype=np.float64) for x in (reference, prediction))
    if scale not in (2, 3, 4) or a.shape != b.shape or a.ndim != 3:
        raise ValueError("CHW shapes and supported ratio required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite RR input")
    band_mse = np.mean((a-b)**2, axis=(1, 2))
    norms = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    valid = norms > 1e-15
    cosine = np.divide(np.sum(a*b, axis=0), norms, out=np.ones_like(norms), where=valid)
    sam = float(np.degrees(np.arccos(np.clip(cosine[valid], -1, 1))).mean()) if valid.any() else 0.
    window = min(7, a.shape[-2], a.shape[-1])
    window -= 1 - window % 2
    return {"MSE": float(band_mse.mean()),
            "PSNR": float(np.mean(-10*np.log10(np.maximum(band_mse, 1e-30)))),
            "SAM": sam,
            "ERGAS": float(100/scale*np.sqrt(np.mean(band_mse/np.maximum(a.mean((1,2))**2,1e-30)))),
            "SSIM": float(np.mean([structural_similarity(x, y, data_range=1., win_size=window)
                                     for x, y in zip(a, b)]))}

def scene_means(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["scene_id"], []).append(row["metrics"])
    scenes = [{"scene_id": scene, "metrics": {k: float(np.mean([r[k] for r in items]))
               for k in items[0]}} for scene, items in sorted(groups.items())]
    mean = {k: float(np.mean([s["metrics"][k] for s in scenes])) for k in scenes[0]["metrics"]}
    return {"samples": rows, "scenes": scenes, "scene_mean": mean, "scene_count": len(scenes)}
''',
'sources.json': r'''{
  "upstream_commit": "a4ca40e407b12bf4c30f804384405ce321d11c51",
  "upstream_url": "https://raw.githubusercontent.com/zhouchuanxu/SSA-MRN/a4ca40e407b12bf4c30f804384405ce321d11c51/network.py",
  "upstream_sha256": "b263d99eb88715a8835ca44c33f6820eb02ee7a8314e31982e73fbb5c8bc1816",
  "adapter_origin": "SSA-MRN/src/ssamrn/models/ssa_mrn.py",
  "adapter_sha256": "c140c1fee05a7958b924c6058e14b31daab7ded51c1aae7122c3287c7afd8891",
  "metrics_origin": "SSA-MRN/src/ssamrn/metrics.py",
  "metrics_sha256": "26837190e6e0d69856bbfe277f1792efd009c46157bbc1d7216cc5b872d6f2ef",
  "upstream_download_or_original_sha256": "9321e77c7dcb9a767a5ca2ec033036096df469dbfcf6d3c4ba0081c08b38849a",
  "adapter_download_or_original_sha256": "0d8bb038d83543c44c5b76a97ea2bd4f989b811446287a803d66e465135944ff",
  "metrics_download_or_original_sha256": "c278b0d4057980dec7a943af403611a6657f775cebd0fc88e199a04f9cf19738",
  "snapshot_transform": "newline normalization only; classes/operations unchanged"
}''',
'upstream_snapshot.py': r'''import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class Resbackbone(nn.Module):
    def __init__(self, in_channels):
        super(Resbackbone, self).__init__()
        channel = 64
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channel, kernel_size=3, padding=1, stride=1,
                               bias=True)

        self.res1 = ResBlock(channel)
        self.res2 = ResBlock(channel)
        self.res3 = ResBlock(channel)
        self.res4 = ResBlock(channel)

        self.backbone = nn.Sequential(
            self.res1,
            self.res2,
            self.res3,
            self.res4,

        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=1, stride=1, kernel_size=3, padding=1,
                               bias=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.backbone(x)
        x = self.conv2(x)
        return x


class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 64

        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, padding=1, kernel_size=3, stride=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, padding=1, kernel_size=3, stride=1,
                                bias=True)
        self.relu = nn.ReLU(
            inplace=True) 
    def forward(self, x):
        rs1 = self.relu(self.conv20(x))
        rs2 = self.conv21(rs1)  
        # print(f"rs2:{rs2.shape},x:{x.shape}")
        rs = torch.add(x, rs2) 
        return rs


class SSA(nn.Module):
    def __init__(self,channels):
        super(SSA, self).__init__()

        self.fixed  = 4


        self.conv1t6 = nn.Conv2d(in_channels=1, out_channels=self.fixed , kernel_size=3, stride=1, padding=1,
                                 bias=True)
        self.conv6t12 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv6t6 = nn.Conv2d(in_channels=self.fixed , out_channels=self.fixed , kernel_size=3, stride=1, padding=1,
                                 bias=True)
        self.relu = nn.ReLU()
        self.conv7t6_3 = nn.Conv2d(in_channels=channels + 1, out_channels=self.fixed, kernel_size=3, stride=1,
                                   padding=1, bias=True)

    def forward(self, pan, ms_pro):  # ms_pro 6xhxw  pan 1xhxw
        pan_guide = self.relu(self.conv1t6(pan))
        pan_guide = self.conv6t6(pan_guide)  # 6*h*w
        ms_pro = self.conv7t6_3(ms_pro)
        # pan reshape -> 6xhw
        # ms_pro reshape -> 6xhw  -> transpose > 6xwh
        b, c, h, w = ms_pro.size()
        pan_reshape = pan_guide.view(b, c, -1)  # 6xhw
        ms_pro_transpose = ms_pro.permute(0, 1, 3, 2)
        ms_pro_reshape = ms_pro_transpose.reshape(b, c, -1)  # 6xhw
        # print(ms_pro_reshape.shape)
        # ms使用了转置

        # pan_matrix * ms_pro_matrix -> 6xhwxwh -> softmax ->6xhxw
        pan_matrix = torch.mul(pan_reshape, ms_pro_reshape)  # b x c x hw x wh
        # print(pan_matrix.shape)
        pan_matrix = pan_matrix.view(b, c, h, w)
        # print(pan_matrix.shape)
        pan_matrix = pan_matrix.permute(0, 1, 3, 2)
        pan_matrix = pan_matrix.contiguous().view(b, c, -1)
       
        pan_matrix = F.softmax(pan_matrix, dim=-1)
        # print(pan_matrix.shape)

        # transpose_R * ms_pro_matrix -> 6xhwxwh -> reshape -> output= 6xhxw
        output = torch.mul(pan_reshape, pan_matrix)  # b x c x hw
        output = output.contiguous().view(b, c, h, w)  # reshape to 6xhxw

        # print(f"res{res.shape}")
        return output







class PansharpeningNet(nn.Module):
    def __init__(self, channels):
        super(PansharpeningNet, self).__init__()

        self.fixed = 4

        # Downsampling and Upsampling layers
        self.downsample1 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.downsample2 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.downsample200 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        # Feature extraction and fusion layers with individual names
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()

        self.conv1t64 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv64tnum = nn.Conv2d(64, channels, kernel_size=3, padding=1)
        self.convnumt64 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)

        self.conv1t64_pan = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv64tnum_pan = nn.Conv2d(64, channels, kernel_size=3, padding=1)

        self.convnum3tnum1 = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)
        self.convnum3tnum2 = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)
        self.convnum3tnum3 = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)

        self.conv48tnum1 = nn.Conv2d(in_channels=self.fixed * channels, out_channels=channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)
        self.conv48tnum2 = nn.Conv2d(in_channels=self.fixed * channels, out_channels=channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)
        self.conv48tnum3 = nn.Conv2d(in_channels=self.fixed * channels, out_channels=channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)

        self.conv7t6_1 = nn.Conv2d(in_channels=channels + 1, out_channels=self.fixed, kernel_size=3, stride=1,
                                   padding=1, bias=True)
        self.conv7t6_2 = nn.Conv2d(in_channels=channels + 1, out_channels=self.fixed, kernel_size=3, stride=1,
                                   padding=1, bias=True)
        self.conv7t6_3 = nn.Conv2d(in_channels=channels + 1, out_channels=self.fixed, kernel_size=3, stride=1,
                                   padding=1, bias=True)

        self.conv2ctc1 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                   bias=True)
        self.conv2ctc2 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                   bias=True)
        self.conv2ctc3 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                   bias=True)

        self.SSA_blocks = nn.ModuleList([SSA(channels=channels) for _ in range(channels)])

        self.SSA_blocks1 = nn.ModuleList([SSA(channels=channels) for _ in range(channels)])
        self.SSA_blocks2 = nn.ModuleList([SSA(channels=channels) for _ in range(channels)])
        self.conv1t1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        self.cov2t64 = nn.Conv2d(in_channels=2 + channels, out_channels=64, kernel_size=3, stride=1, padding=1,
                                 bias=True)

        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.backbone = nn.Sequential(
                self.res1,
                self.res2,
                self.res3,
                self.res4,
            )
        self.conv64t48 = nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.fixed = 4;

        # Downsampling and Upsampling layers
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample1 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample100 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample100 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample101  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample102\
            = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Fusion layers
        self.resblock = Resblock()
        self.prelu = nn.PReLU()
        self.conv1t64 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv64tnum = nn.Conv2d(64, channels, kernel_size=3, padding=1)
        self.convnumt64 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)

        self.conv1t64bf2 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv64tnumbf2 = nn.Conv2d(64, channels, kernel_size=3, padding=1)
        self.convnumt64bf2 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)

        self.convnumtnum = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1)
        self.convctc1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # convctc1
        self.convctc2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # convctc2
        self.convnum3tnum = nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels * 6, stride=1, kernel_size=3, padding=1,
                               bias=True)

        self.SSA_blocks = nn.ModuleList(
            [SSA(channels=channels) for _ in range(channels)])  # Adding multi-scale blocks

        self.conv48tnum1 = nn.Conv2d(in_channels=self.fixed * channels, out_channels=channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)
        self.conv48tnum2 = nn.Conv2d(in_channels=self.fixed * channels, out_channels=channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)

        self.conv48t48_1 = nn.Conv2d(in_channels=6 * channels, out_channels=6 * channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)
        self.conv48t48_2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                     bias=True)
        self.conv48t48_3 = nn.Conv2d(in_channels=6 * channels, out_channels=6 * channels, kernel_size=3, stride=1,
                                     padding=1, bias=True)

        self.conv7t6 = nn.Conv2d(in_channels=channels + 1, out_channels=self.fixed, kernel_size=3, stride=1, padding=1,
                                 bias=True)

        self.convnum3tn64 = nn.Conv2d(in_channels=channels * 3, out_channels=64, kernel_size=3, stride=1, padding=1,
                                      bias=True)

        self.group_size = 6 
        self.channels = channels
        self.cov2t64 = nn.Conv2d(in_channels=2 + channels, out_channels=64, kernel_size=3, stride=1, padding=1,
                                 bias=True)
        self.conv64t48 = nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1t1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)

        self.convnumt48 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                    bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.conv8t4 = nn.Conv2d(in_channels=8, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                    bias=True)



   
    def forward(self, pan, lms):
        # Stage 1: Downsample and upsample each input
        pan_down_up = self.upsample1(self.downsample1(pan))
        pan_updown = self.prelu(self.conv1t1(pan_down_up))
        input = torch.cat([lms, pan, pan_updown], dim=1)

        y = self.cov2t64(input)
        y = self.backbone(y)
        y1 = self.conv64t48(y)
        rs = y1 + lms

        # Process input channels separately
        split_input = [rs[:, i:i + self.group_size, :, :] for i in range(0, 48, self.group_size)]
        split_ms = [lms[:, i, :, :].unsqueeze(1) for i in range(self.channels)]

        outputs = []
        for input_channel, ms_channel, ssa in zip(split_input, split_ms, self.SSA_blocks):
            x = torch.cat([rs, ms_channel], dim=1)

            ms_one = ssa(pan, x)
            outputs.append(ms_one)

        pan_ms = torch.cat(outputs, dim=1)
        pan_ms = self.prelu1(self.conv48tnum1(pan_ms))
        fsm_msb = rs + pan_ms
        res = fsm_msb

        res_down = self.downsample100(res)

        # Stage 2: MS upsample and next stage fusion
        ms_up = self.upsample100(ms)
        rs = res_down

        split_input = [rs[:, i:i + self.group_size, :, :] for i in range(0, 48, self.group_size)]
        split_ms = [ms_up[:, i, :, :].unsqueeze(1) for i in range(self.channels)]

        outputs = []
        for input_channel, ms_channel, ssa in zip(split_input, split_ms, self.SSA_blocks1):
            x = torch.cat([rs, ms_channel], dim=1)

            pan_down = self.downsample100(pan)
            ms_one = ssa(pan_down, x)
            outputs.append(ms_one)

        pan_ms = torch.cat(outputs, dim=1)
        pan_ms = self.prelu2(self.conv48tnum2(pan_ms))
        fsm_msb = rs + pan_ms
        res_down = fsm_msb

        # Stage 3: Small scale stage
        res_small = self.downsample200(res_down)
        rs = res_small

        split_input = [rs[:, i:i + self.group_size, :, :] for i in range(0, 48, self.group_size)]
        split_ms = [ms[:, i, :, :].unsqueeze(1) for i in range(self.channels)]

        outputs = []
        for input_channel, ms_channel, ssa in zip(split_input, split_ms, self.SSA_blocks2):
            x = torch.cat([rs, ms_channel], dim=1)

            pan_down = self.downsample2(pan)
            ms_one = ssa(pan_down, x)
            outputs.append(ms_one)

        pan_ms = torch.cat(outputs, dim=1)
        pan_ms = self.prelu3(self.conv48tnum3(pan_ms))
        res_small = pan_ms

        # Output processing
        ms_output = ms
        ms_output = torch.cat([ms_output, res_small], dim=1)
        ms_output = self.conv2ctc1(ms_output)

        ms_output = self.upsample101(ms_output)
        ms_output = torch.cat([ms_output, res_down], dim=1)
        ms_output = self.conv2ctc2(ms_output)

        ms_output = self.upsample102(ms_output)
        ms_output = torch.cat([ms_output, res], dim=1)
        ms_output = self.conv2ctc3(ms_output)




        return ms_output
''',
}
EMBEDDED_HASHES = {'__init__.py': 'ab4acc047ee17c90208f6a40e6dbb28574d5110e304d2876ac75e44e06874ca6', 'adapter_snapshot.py': 'c140c1fee05a7958b924c6058e14b31daab7ded51c1aae7122c3287c7afd8891', 'data.py': '3f252c8ddba19d4f250d284db913e9973ad71e22cbdc6e0b22969495e8cf3da2', 'engine.py': '0d11aeb13b1f7b1a096450d8d9f1042cb01a261ab4f047692f6712f2205eefca', 'metrics_snapshot.py': '26837190e6e0d69856bbfe277f1792efd009c46157bbc1d7216cc5b872d6f2ef', 'model.py': '973d4de322602888398c0da3833a63d1c9fcdf40514cc809c657978de5a2f723', 'reference.py': 'e5c81567976efaa8e52af65305ddc5d2df55c2487408a21a10230a0bba153f75', 'reporting.py': 'cebb7f2908398edb51010dc5739ff8bfdcbb9508c8953a084084462204d47233', 'runner.py': '34247e425441d36e8bbda6a36ce9a4ec50d2c6fbbed5e196bb0460e7bbc97325', 'scoring.py': 'e5e1c1fb861605b70e71d4d56258f02a0373dd770ccacd31d8dacf0bbc7a91e3', 'sources.json': 'd41e0df568739bf02cc68f813576aeb7f5b031218814e06ce5b5c8d3a6637476', 'upstream_snapshot.py': 'b263d99eb88715a8835ca44c33f6820eb02ee7a8314e31982e73fbb5c8bc1816'}
BUNDLE_HASH = 'c60ab4b80d0d39c425e0f0a6206031726543505334196b9ec7b02caf9075fafc'

import hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path
for module, package in [("torch", "torch"), ("h5py", "h5py"), ("numpy", "numpy"),
                        ("scipy", "scipy"), ("skimage", "scikit-image"), ("matplotlib", "matplotlib")]:
    if importlib.util.find_spec(module) is None:
        # Never install an arbitrary Torch build into a Kaggle CUDA image.
        if module == "torch":
            raise RuntimeError("Select the Kaggle Python GPU image with PyTorch installed")
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
import torch
if torch.cuda.device_count() < 2:
    raise RuntimeError("Select GPU T4 x2; this cell does not silently fall back to CPU")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
runtime = Path("/kaggle/working/.variable_scale_cell_runtime") / BUNDLE_HASH
package_dir = runtime / "variable_scale"
package_dir.mkdir(parents=True, exist_ok=True)
for name, content in EMBEDDED_SOURCES.items():
    if hashlib.sha256(content.encode("utf-8")).hexdigest() != EMBEDDED_HASHES[name]:
        raise ValueError("Embedded source checksum failed: " + name)
    destination = package_dir / name
    if destination.exists() and destination.read_text(encoding="utf-8") != content:
        raise ValueError("Existing runtime source differs: " + name)
    if not destination.exists():
        destination.write_text(content, encoding="utf-8", newline="\n")
# A previous execution can leave imported modules in a persistent notebook kernel.
for name in list(sys.modules):
    if name == "variable_scale" or name.startswith("variable_scale."):
        del sys.modules[name]
sys.path.insert(0, str(runtime))
from variable_scale.runner import run_suite
settings = {
    "mode": MODE, "output_dir": OUTPUT_DIR, "resume": RESUME,
    "resume_input": RESUME_INPUT or None, "scene_manifest": SCENE_MANIFEST or None,
    "allow_unverified_splits": ALLOW_UNVERIFIED_SPLITS, "provisional_fr": PROVISIONAL_FR,
    "devices": ["cuda:0", "cuda:1"],
    "job_names": JOB_NAMES,
    "paths": {"train": TRAIN_H5, "val": VAL_H5, "RR": RR_H5, "FR": FR_H5},
    "training": {"epochs": EPOCHS, "seed": SEED, "batch_size": EFFECTIVE_BATCH,
                 "micro_batch": MICRO_BATCH, "lr": LEARNING_RATE, "amp": AMP,
                 "deterministic": True, "checkpoint_seconds": 60, "status_seconds": 15},
}
result_dir = run_suite(settings, notebook=True)
from IPython.display import FileLink, display
display(FileLink(str(result_dir.with_suffix(".zip"))))
display(FileLink(str(result_dir.with_suffix(".zip.sha256.json"))))
