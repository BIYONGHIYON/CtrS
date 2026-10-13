# Paste this ENTIRE file into ONE Kaggle code cell. Select T4 x2 and Internet.
# This cell trains only the two arms of the current experiment.
REQUIRE_RESUME = False
RESUME_INPUT = ""  # Exact suite folder, in /kaggle/working or attached Saved Output.
DATA_ROOT = "/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset"
TRAIN_H5 = f"{DATA_ROOT}/QuickBird__Training_Dataset__train_qb.h5"
VAL_H5 = f"{DATA_ROOT}/QuickBird__Training_Dataset__valid_qb.h5"
RR_H5 = f"{DATA_ROOT}/QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5"
FR_H5 = f"{DATA_ROOT}/QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5"
# Configure Persistence -> Files only BEFORE running; save/download Output.
# Checkpoints exist on the VM until that storage is lost. This is not a cloud backup.
# Missing/corrupt resume state stops BOTH arms before either is allowed to train.
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ARMS = ("B1", "B1_alpha_bounded")
PINNED = "be02b7cbf03fe00681194ae9fc9419661f24cd56"
UPSTREAM = "a4ca40e407b12bf4c30f804384405ce321d11c51"
EXPECTED_DATA = {
    "train": "7d45e191cd03f3df12eb76b5f27624eaba1b464970788eb61723c3574dd14371",
    "val": "b307bcbb0d69056278a8a0b4970f1cd1b58dfff19bbe949be224e106f1a18470",
    "RR": "9842a1232ad2d5b9a61c9fd7fb353e8fc6214eeac6e947d888f221ad6ecdda35",
    "FR": "4583c2890009e047ddc1e0daf88db3f5db43041fc301fa9ff8eb52a37f84b88f",
}
EXPECTED_SOURCE = {
    "SSA-MRN/src/ssamrn/__init__.py": "4068ee713442b7896609dffeeb5c1a54d22cd19abfb31da1202b3ca45d3ca08a",
    "SSA-MRN/src/ssamrn/models/__init__.py": "8f3d7aaa5ff196e99a95c2c2926e11131f1814dd1cb235929c0cfb3124de4b88",
    "SSA-MRN/src/ssamrn/models/ssa_mrn.py": "c140c1fee05a7958b924c6058e14b31daab7ded51c1aae7122c3287c7afd8891",
    "SSA-MRN/src/ssamrn/metrics.py": "26837190e6e0d69856bbfe277f1792efd009c46157bbc1d7216cc5b872d6f2ef",
    "SSA-MRN/references/upstream/network.py": "9321e77c7dcb9a767a5ca2ec033036096df469dbfcf6d3c4ba0081c08b38849a",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def ensure_qb_data(paths, download=None):
    # Honor every existing explicit path. Search only the returned dataset for missing files.
    missing = {split: p for split, p in paths.items() if not Path(p).is_file()}
    if not missing:
        return dict(paths)
    handle = "biyonghiyon/ctrs-pan-sharpening-dataset"
    print(f"[CURRENT DATA] Missing {len(missing)} QB files; attaching/downloading {handle}", flush=True)
    if download is None:
        if importlib.util.find_spec("kagglehub") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "kagglehub"],
                           check=True, timeout=180)
        import kagglehub
        download = kagglehub.dataset_download
    try:
        root = Path(download(handle))
    except Exception as error:
        raise RuntimeError("QB download failed. Enable Internet and dataset access, or attach it with Add Input. No training started.") from error
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset download returned no directory: {root}")
    result = dict(paths)
    for split, p in missing.items():
        direct = root / Path(p).name
        matches = [direct] if direct.is_file() else sorted(root.rglob(Path(p).name))
        matches = [m for m in matches if m.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"{split}: expected one {Path(p).name}; found {matches}. No training started.")
        result[split] = str(matches[0].resolve())
    if len({str(Path(p).resolve()) for p in result.values()}) != 4:
        raise ValueError("All four dataset splits must be distinct")
    return result


MODEL = r'''
import hashlib
import math
import torch
from torch import nn
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet

class SharedThreeStage(nn.Module):
    def __init__(self, bounded=False):
        super().__init__()
        self.core = RestoredPansharpeningNet(4, 4)
        self.error_encoder = nn.Conv2d(4, 4, 3, padding=1, bias=True)
        self.bounded = bounded
        if bounded:
            self.a = nn.Parameter(torch.tensor(math.log(0.1 / 0.9), dtype=torch.float32))
        else:
            self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def effective_alpha(self):
        return self.a.sigmoid() if self.bounded else self.alpha

    def raw_alpha(self):
        return self.a if self.bounded else self.alpha

    def forward(self, pan, lms, ms):
        current = lms
        for _ in range(3):
            conditioned = current + self.error_encoder(torch.zeros_like(current))
            proposal = self.core(pan, conditioned, ms)
            current = current + self.effective_alpha() * (proposal - current)
        return current

def build(arm):
    if arm not in ("B1", "B1_alpha_bounded"):
        raise ValueError(arm)
    return SharedThreeStage(arm == "B1_alpha_bounded")

def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def common_hash(model):
    return state_hash({k:v for k,v in model.state_dict().items()
                       if k.startswith(("core.", "error_encoder."))})
'''

SUPPORT = r'''
import hashlib
import json
import math
import os
import shutil
import uuid
from pathlib import Path
import numpy as np
import torch

ARMS = ("B1", "B1_alpha_bounded")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    with tmp.open("r+b") as stream: os.fsync(stream.fileno())
    os.replace(tmp, path)

def cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value, list): return [cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu(v) for v in value)
    return value

def order_hash(order):
    return hashlib.sha256(order.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def environment():
    import importlib.metadata
    import platform
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "h5py", "scipy", "scikit-image", "matplotlib", "Pillow")}
    return dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                python=platform.python_version(), packages=packages, gpu="Tesla T4",
                capability=list(torch.cuda.get_device_capability(0)))

def validate_state(ck, protocol, arm):
    required = {"protocol", "arm", "epoch", "mid", "best", "best_epoch", "model", "optimizer",
                "scaler", "history", "cpu_rng", "cuda_rng", "order_rng", "python_rng", "numpy_rng"}
    if not required <= ck.keys(): raise ValueError("Incomplete recovery state")
    if ck["protocol"] != protocol or ck["arm"] != arm: raise ValueError("Protocol/arm mismatch")
    e = ck["epoch"]; history = ck["history"]; total = protocol["epochs"]
    if type(e) is not int or not 0 <= e <= total or len(history) != e:
        raise ValueError("Epoch/history cursor mismatch")
    for i, row in enumerate(history):
        keys = ("train_mse", "val_mse", "val_psnr", "val_sam", "alpha", "alpha_raw", "train_seconds", "validation_seconds")
        if row["epoch"] != i + 1 or not all(math.isfinite(row[k]) for k in keys):
            raise ValueError("Invalid measured history")
        if len(row["sample_order_sha256"]) != 64: raise ValueError("Order proof missing")
        if arm == ARMS[1] and not 0 <= row["alpha"] <= 1: raise ValueError("Invalid bounded alpha")
    if history:
        first_min = min(history, key=lambda row: row["val_mse"])
        if ck["best_epoch"] != first_min["epoch"] or ck["best"] != first_min["val_mse"]:
            raise ValueError("Best was not selected solely by earliest minimum validation MSE")
    elif ck["best_epoch"] != 0 or ck["best"] is not None:
        raise ValueError("Invalid epoch-zero best")
    expected_param = "a" if arm == ARMS[1] else "alpha"
    other_param = "alpha" if arm == ARMS[1] else "a"
    if expected_param not in ck["model"] or other_param in ck["model"]:
        raise ValueError("Alpha parameterization mismatch")
    if ck["model"][expected_param].numel() != 1: raise ValueError("Scalar alpha required")
    if not all(torch.isfinite(v).all().item() for v in ck["model"].values()):
        raise ValueError("Nonfinite model state")
    if not ck["optimizer"].get("param_groups") or not isinstance(ck["scaler"], dict):
        raise ValueError("Optimizer/scaler state missing")
    if not {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"} <= ck["scaler"].keys():
        raise ValueError("Enabled AMP scaler state incomplete")
    if not math.isfinite(ck["scaler"]["scale"]) or ck["scaler"]["scale"] <= 0:
        raise ValueError("Invalid AMP scale")
    if "optimizer" in protocol:
        cfg = protocol["optimizer"]
        for group in ck["optimizer"]["param_groups"]:
            if group["lr"] != protocol["lr"] or tuple(group["betas"]) != tuple(cfg["betas"]):
                raise ValueError("Optimizer learning rate/betas differ")
            for key in ("eps", "weight_decay", "amsgrad", "fused", "foreach"):
                if group[key] != cfg[key]: raise ValueError("Optimizer implementation differs: " + key)
    import random
    random.Random().setstate(ck["python_rng"])
    nr = ck["numpy_rng"]
    np.random.RandomState().set_state((nr[0], np.array(nr[1], dtype=np.uint32), *nr[2:]))
    for k in ("cpu_rng", "cuda_rng", "order_rng"):
        if not isinstance(ck[k], torch.Tensor) or ck[k].dtype != torch.uint8:
            raise ValueError("RNG state missing: " + k)
    mid = ck["mid"]
    if mid is not None:
        n, c, h, w = protocol["data"]["train"]["shapes"]["gt"]
        batch = protocol["batch"]; nb = math.ceil(n / batch)
        if e >= total or type(mid["batch"]) is not int or not 0 <= mid["batch"] <= nb:
            raise ValueError("Invalid batch cursor")
        order = mid["order"]
        if order.dtype != torch.int64 or order.shape != (n,) or not torch.equal(order.sort().values, torch.arange(n)):
            raise ValueError("Invalid saved permutation")
        expected_count = min(mid["batch"] * batch, n) * c * h * w
        if mid["count"] != expected_count or order_hash(order) != mid["order_sha256"]:
            raise ValueError("Batch order/accumulated-loss cursor mismatch")
        if mid["sse"].numel() != 1 or not torch.isfinite(mid["sse"]).item() or mid["sse"].item() < 0:
            raise ValueError("Invalid accumulated SSE")
        if not math.isfinite(mid["seconds"]) or mid["seconds"] < 0:
            raise ValueError("Invalid elapsed time")
    return ck

def reference(folder, ref):
    p = (folder / ref["file"]).resolve()
    recovery = (folder / "recovery").resolve()
    if p.parent != recovery or not p.name.startswith("state_") or p.suffix != ".pt":
        raise ValueError("Unsafe recovery reference")
    if not p.is_file() or p.stat().st_size != ref["bytes"] or sha(p) != ref["sha256"]:
        raise ValueError("Recovery checkpoint hash mismatch")
    return p

def load_record(folder, protocol, arm):
    folder = Path(folder); errors = []
    for name in ("resume.json", "resume.prev.json"):
        try:
            record = json.loads((folder / name).read_text(encoding="utf-8"))
            ck = torch.load(reference(folder, record["latest"]), map_location="cpu", weights_only=True)
            validate_state(ck, protocol, arm)
            if ck["best_epoch"]:
                best = torch.load(reference(folder, record["best"]), map_location="cpu", weights_only=True)
                validate_state(best, protocol, arm)
                if best["mid"] is not None or best["epoch"] != ck["best_epoch"] or best["best"] != ck["best"]:
                    raise ValueError("Inconsistent latest/best transaction")
                if best["history"] != ck["history"][:best["epoch"]]:
                    raise ValueError("Inconsistent best history")
            elif record["best"] is not None: raise ValueError("Unexpected best reference")
            return ck, record, name
        except Exception as error:
            errors.append(name + ": " + str(error))
    raise RuntimeError(arm + ": resume required; no fresh start. " + "; ".join(errors))

def copy_original(source, target):
    target = Path(target)
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, tmp); os.replace(tmp, target)
    if sha(source) != sha(target): raise RuntimeError("Original checkpoint copy failed")

def export_record(folder, record):
    folder = Path(folder)
    for kind in ("latest", "best"):
        if record[kind] is not None:
            source = reference(folder, record[kind]); target = folder / (kind + ".pt")
            if not target.is_file() or sha(target) != record[kind]["sha256"]:
                copy_original(source, target)

def commit_checkpoint(folder, state, previous, new_best=False):
    # The pointer commits latest AND best together. Canonical exports are byte copies.
    folder = Path(folder); recovery = folder / "recovery"; recovery.mkdir(parents=True, exist_ok=True)
    path = recovery / ("state_" + uuid.uuid4().hex + ".pt")
    tmp = path.with_suffix(".tmp")
    torch.save(cpu(state), tmp)
    with tmp.open("r+b") as stream: os.fsync(stream.fileno())
    os.replace(tmp, path)
    ref = dict(file="recovery/" + path.name, bytes=path.stat().st_size, sha256=sha(path))
    record = dict(latest=ref, best=ref if new_best else (previous["best"] if previous else None))
    if previous: write(folder / "resume.prev.json", previous)
    write(folder / "resume.json", record)  # authoritative atomic commit
    export_record(folder, record)
    # Prune only our unreferenced checkpoint generations, keeping both transaction pointers.
    keep = {r["file"] for rec in (record, previous) if rec for r in rec.values() if r}
    for p in recovery.glob("state_*.pt"):
        if "recovery/" + p.name not in keep:
            if p.resolve().parent != recovery.resolve(): raise ValueError("Unsafe checkpoint cleanup")
            p.unlink()
    return record
'''

INITIALIZATION = r'''
import json
import os
import sys
from pathlib import Path
code = Path(sys.argv[1]).resolve(); dest = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(code / "SSA-MRN/src")); sys.path.insert(0, str(code))
import h5py
import torch
from alpha_models import build, state_hash, common_hash
from alpha_support import sha, write
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
models = {}
for arm in ("B1", "B1_alpha_bounded"):
    torch.manual_seed(42); models[arm] = build(arm).eval()
left, right = models.values()
if common_hash(left) != common_hash(right): raise RuntimeError("Core/encoder initialization differs")
for name, value in left.state_dict().items():
    if name.startswith(("core.", "error_encoder.")):
        if not torch.equal(value, right.state_dict()[name]): raise RuntimeError(name)
with h5py.File(sys.argv[3], "r") as f:
    inputs = [torch.from_numpy(f[k][:2]).float() / 2047. for k in ("pan", "lms", "ms")]
with torch.inference_mode():
    outputs = {arm: model(*inputs) for arm, model in models.items()}
torch.testing.assert_close(outputs["B1"], outputs["B1_alpha_bounded"], atol=1e-6, rtol=1e-5)
alphas = {arm: float(model.effective_alpha()) for arm, model in models.items()}
if any(abs(a - 0.1) > 1e-7 for a in alphas.values()): raise RuntimeError("Initial effective alpha mismatch")
states = {arm: model.state_dict() for arm, model in models.items()}
dest.mkdir(parents=True, exist_ok=True)
torch.save(dict(states=states, inputs=inputs), dest / "initial_states.pt")
proof = dict(seed=42, core_sha256=state_hash(left.core.state_dict()),
             encoder_sha256=state_hash(left.error_encoder.state_dict()), common_sha256=common_hash(left),
             alpha=alphas, output_max_abs=float((outputs["B1"] - outputs["B1_alpha_bounded"]).abs().max()),
             output_atol=1e-6, output_rtol=1e-5, alpha_atol=1e-7,
             input="first two validation patches; normalized peak2047; FP32 CPU",
             initial_states_sha256=sha(dest / "initial_states.pt"))
write(dest / "initialization_pair.json", proof)
print("CTRS_ALPHA_INIT=" + json.dumps(proof), flush=True)
'''

WORKER = r'''
import gc
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw
C = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")); ROOT = Path(C["root"])
CODE = ROOT / "provenance/code"
sys.path.insert(0, str(CODE / "SSA-MRN/src")); sys.path.insert(0, str(CODE))
from alpha_models import build, common_hash, state_hash
from alpha_support import (cpu, sha, write, order_hash, load_record, export_record,
                           commit_checkpoint, reference, validate_state, environment)
from ssamrn.metrics import rr_metrics, fr_metrics
ARM = C["arm"]; P = C["protocol"]
CKDIR = ROOT / "checkpoints" / ARM
RESULTS = ROOT / "results" / ARM
RUNTIME = ROOT / "runtime" / ARM
for d in (CKDIR, RESULTS, RUNTIME): d.mkdir(parents=True, exist_ok=True)
INPUTS = ("pan", "lms", "ms"); KEYS = INPUTS + ("gt",)
STOP = False

def stop(signum, frame):
    global STOP
    STOP = True
signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)

def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

def status(phase, **kw):
    value = dict(arm=ARM, phase=phase, session_nonce=C["session_nonce"], gpu=C["physical_gpu"],
                 updated=time.time(), allocated_gb=torch.cuda.memory_allocated()/1024**3,
                 reserved_gb=torch.cuda.memory_reserved()/1024**3,
                 peak_gb=torch.cuda.max_memory_allocated()/1024**3, **kw)
    write(RUNTIME / "status.json", value)
    print(json.dumps(value, allow_nan=False), flush=True)

def layout(value): return value.contiguous(memory_format=torch.channels_last)
def batch(data, ids): return {k:layout(v.index_select(0, ids)) for k,v in data.items()}
def forward(model, data): return model(*(data[k] for k in INPUTS))

def cache():
    size = sum(np.prod(P["data"][s]["shapes"][k])*4 for s in ("train", "val") for k in KEYS)
    free, total = torch.cuda.mem_get_info()
    if free < size + 4*1024**3:
        raise RuntimeError("Insufficient GPU memory for resident data + batch32; no batch/layout fallback")
    result = {}; done = 0
    for split in ("train", "val"):
        result[split] = {}
        with h5py.File(C["paths"][split], "r") as f:
            for key in KEYS:
                tensor = torch.empty(f[key].shape, device="cuda", dtype=torch.float32,
                                     memory_format=torch.channels_last)
                for i in range(0, len(tensor), 128):
                    a = np.asarray(f[key][i:i+128], dtype=np.float32)
                    if not np.isfinite(a).all(): raise ValueError("Nonfinite dataset")
                    tensor[i:i+len(a)].copy_(torch.from_numpy(a / 2047.))
                    if STOP: raise InterruptedError("Stopped during GPU cache load")
                result[split][key] = tensor; done += tensor.numel()*4
                status("cache", split=split, array=key, percent=100*done/size)
    return result

def gpu_initialization(model):
    initial = torch.load(ROOT / "provenance/initial_states.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(initial["states"][ARM], strict=True)
    proof = C["initialization"]
    if common_hash(model) != proof["common_sha256"]: raise RuntimeError("Initial core/encoder mismatch")
    alpha = float(model.effective_alpha().detach())
    if abs(alpha - 0.1) > proof["alpha_atol"]: raise RuntimeError("Effective alpha initialization mismatch")
    inputs = [layout(v.cuda()) for v in initial["inputs"]]; model.eval()
    with torch.inference_mode():
        fp32 = model(*inputs).float().cpu()
        with torch.autocast("cuda", dtype=torch.float16): amp = model(*inputs).float().cpu()
    if not torch.isfinite(fp32).all() or not torch.isfinite(amp).all():
        raise RuntimeError("Nonfinite initial GPU output")
    output = dict(fp32=fp32, amp=amp, alpha=alpha, common_sha256=common_hash(model))
    torch.save(output, ROOT / "provenance" / (ARM + "_initial_gpu.pt"))
    return dict(common_sha256=output["common_sha256"], alpha=alpha)

@torch.inference_mode()
def validate(model, data, epoch):
    model.eval(); sse = torch.zeros((), device="cuda"); psnr = sse.clone(); sam = sse.clone()
    count = 0; n = len(data["gt"]); last = time.monotonic()
    for i in range(0, n, 32):
        b = batch(data, torch.arange(i, min(i+32, n), device="cuda"))
        pred = forward(model, b).float(); gt = b["gt"]; d = (pred-gt).square()
        sse += d.sum(); count += gt.numel()
        psnr += (-10*d.mean((2,3)).clamp_min(1e-30).log10()).mean(1).sum()
        norm = pred.norm(dim=1)*gt.norm(dim=1); valid = norm > 0
        angle = torch.rad2deg(torch.acos(((pred*gt).sum(1)/norm.clamp_min(1e-30)).clamp(-1,1)))
        sam += ((angle*valid).flatten(1).sum(1)/valid.flatten(1).sum(1).clamp_min(1)).sum()
        if i == 0 or time.monotonic()-last >= 15:
            status("validation", epoch=epoch, batch=i//32+1, batches=math.ceil(n/32),
                   alpha=float(model.effective_alpha().detach())); last = time.monotonic()
    values = dict(val_mse=float(sse/count), val_psnr=float(psnr/n), val_sam=float(sam/n))
    if not all(math.isfinite(v) for v in values.values()): raise RuntimeError("Nonfinite validation")
    return values

def train():
    ck, record, origin = (load_record(CKDIR, P, ARM) if C["resume"] else (None, None, None))
    if ck is not None:
        if record["latest"]["sha256"] != C["resume_sha256"]: raise RuntimeError("Resume source changed after preflight")
        export_record(CKDIR, record)
    seed(); model = build(ARM).cuda().to(memory_format=torch.channels_last)
    proof = gpu_initialization(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8,
                           weight_decay=0, amsgrad=False, foreach=False, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    gen = torch.Generator(device="cuda").manual_seed(42)
    history = []; best = None; best_epoch = 0; start = 0; mid = None
    data = cache()
    if ck is not None:
        model.load_state_dict(ck["model"], strict=True); opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["cpu_rng"]); torch.cuda.set_rng_state(ck["cuda_rng"])
        gen.set_state(ck["order_rng"]); random.setstate(ck["python_rng"])
        nr = ck["numpy_rng"]; np.random.set_state((nr[0], np.array(nr[1], dtype=np.uint32), *nr[2:]))
        history = ck["history"]; best = ck["best"]; best_epoch = ck["best_epoch"]
        start = ck["epoch"]; mid = ck["mid"]
        del ck
    def snapshot(e, m):
        nr = np.random.get_state()
        return dict(protocol=P, arm=ARM, epoch=e, mid=m, best=best, best_epoch=best_epoch,
                    model=model.state_dict(), optimizer=opt.state_dict(), scaler=scaler.state_dict(), history=history,
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), order_rng=gen.get_state(),
                    python_rng=random.getstate(), numpy_rng=[nr[0], nr[1].tolist(), *nr[2:]])
    if not C["resume"]: record = commit_checkpoint(CKDIR, snapshot(0, None), None)
    write(RUNTIME / "ready.json", dict(session_nonce=C["session_nonce"], **proof))
    status("ready", epoch=start, batch=mid["batch"] if mid else 0, alpha=float(model.effective_alpha().detach()))
    # Both arms must pass the parent's GPU output, checkpoint and initialization checks.
    while not STOP:
        gate = ROOT / "runtime/start_training.json"
        if gate.is_file() and json.loads(gate.read_text())["session_nonce"] == C["session_nonce"]: break
        time.sleep(0.2)
    if STOP: status("paused", epoch=start); return False
    n = len(data["train"]["gt"]); nb = math.ceil(n/32); last_save = time.monotonic()
    for e in range(start, 100):
        if mid is not None:
            order = mid["order"].cuda(); cursor = mid["batch"]; sse = mid["sse"].cuda()
            count = mid["count"]; elapsed_before = mid["seconds"]
        else:
            order = torch.randperm(n, generator=gen, device="cuda"); cursor = 0
            sse = torch.zeros((), device="cuda"); count = 0; elapsed_before = 0.
        ohash = order_hash(order); tick = time.monotonic(); last_status = tick; model.train()
        def partial(b):
            return dict(order=order, order_sha256=ohash, batch=b, sse=sse, count=count,
                        seconds=elapsed_before+time.monotonic()-tick)
        # Preserve this epoch's permutation even if stopped before its first batch.
        if STOP:
            record = commit_checkpoint(CKDIR, snapshot(e, partial(cursor)), record)
            status("paused", epoch=e+1, batch=cursor); return False
        for j in range(cursor, nb):
            b = batch(data["train"], order[j*32:(j+1)*32]); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = forward(model, b); loss = F.mse_loss(pred.float(), b["gt"])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sse += loss.detach()*b["gt"].numel(); count += b["gt"].numel()
            now = time.monotonic()
            if j == cursor or now-last_status >= 15 or j+1 == nb:
                torch.cuda.synchronize(); elapsed = elapsed_before+time.monotonic()-tick
                rate = elapsed/max(j+1, 1)
                val_sec = float(np.mean([r["validation_seconds"] for r in history[-5:]])) if history else 0.
                status("train", epoch=e+1, batch=j+1, batches=nb, train_mse=float(sse/count),
                       alpha=float(model.effective_alpha().detach()),
                       eta_seconds=rate*((nb-j-1)+(99-e)*nb)+(100-e)*val_sec,
                       eta_basis="measured train + validation" if history else "measured train only; validation unmeasured")
                if not torch.isfinite(sse).item(): raise RuntimeError("Nonfinite training")
                last_status = time.monotonic()
            if STOP or time.monotonic()-last_save >= 60:
                record = commit_checkpoint(CKDIR, snapshot(e, partial(j+1)), record)
                last_save = time.monotonic()
            if STOP: status("paused", epoch=e+1, batch=j+1); return False
        record = commit_checkpoint(CKDIR, snapshot(e, partial(nb)), record)
        torch.cuda.synchronize(); train_seconds = elapsed_before+time.monotonic()-tick
        vt = time.monotonic(); values = validate(model, data["val"], e+1); torch.cuda.synchronize()
        row = dict(epoch=e+1, train_mse=float(sse/count), alpha=float(model.effective_alpha().detach()),
                   alpha_raw=float(model.raw_alpha().detach()), train_seconds=train_seconds,
                   validation_seconds=time.monotonic()-vt, sample_order_sha256=ohash, **values)
        history.append(row); improved = best is None or values["val_mse"] < best
        if improved: best = values["val_mse"]; best_epoch = e+1
        record = commit_checkpoint(CKDIR, snapshot(e+1, None), record, new_best=improved)
        write(RESULTS / "history.json", history)
        status("epoch_complete", epoch=e+1, batch=nb, batches=nb, best_epoch=best_epoch,
               train_mse=row["train_mse"], alpha=row["alpha"], **values)
        mid = None; last_save = time.monotonic()
        if STOP: status("paused", epoch=e+1); return False
    write(RESULTS / "history.json", history)
    del model, opt, scaler, data; gc.collect(); torch.cuda.empty_cache()
    return True

def panels(a, pred, path, index):
    bands = [2,1,0]; ref = a["gt"][bands].transpose(1,2,0)
    lo, hi = np.percentile(ref, [1,99], axis=(0,1))
    def rgb(value):
        x = value[bands].transpose(1,2,0)
        return Image.fromarray((np.clip((x-lo)/np.maximum(hi-lo,1e-8),0,1)*255).astype("uint8"))
    pan = a["pan"][0]; p0, p1 = np.percentile(pan, [1,99]); h, w = pan.shape
    gray = Image.fromarray((np.clip((pan-p0)/max(p1-p0,1e-8),0,1)*255).astype("uint8")).convert("RGB")
    canvas = Image.new("RGB", (4*w,h+40), "white"); draw = ImageDraw.Draw(canvas)
    images = [rgb(a["ms"]), gray, rgb(pred), rgb(a["gt"])]
    for j, (name, img) in enumerate(zip(("LR MS", "PAN", "Prediction", "GT"), images)):
        canvas.paste(img.resize((w,h), Image.Resampling.NEAREST), (j*w,40))
        draw.text((j*w+5,5), name, fill="black")
    draw.text((5,23), f"RR scene {index} | full scene | common GT stretch", fill="black")
    canvas.save(path)
    return dict(lo=lo.tolist(), hi=hi.tolist(), pan_range=[float(p0),float(p1)])

def evaluate():
    latest, record, _ = load_record(CKDIR, P, ARM); export_record(CKDIR, record)
    if latest["epoch"] != 100 or latest["mid"] is not None: raise RuntimeError("Training incomplete")
    ck = torch.load(reference(CKDIR, record["best"]), map_location="cpu", weights_only=True)
    model = build(ARM).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(ck["model"], strict=True); model.eval()
    result = dict(arm=ARM, best_epoch=ck["epoch"], best_validation_mse=ck["best"],
                  best_alpha=float(model.effective_alpha().detach()), metrics={}, images=[], weights={})
    for kind in ("best", "latest"):
        p = CKDIR / (kind+".pt"); w = torch.load(p, map_location="cpu", weights_only=True)
        result["weights"][kind] = dict(file=str(p.relative_to(ROOT)), epoch=w["epoch"],
                                      sha256=sha(p), bytes=p.stat().st_size)
    for split in ("RR", "FR"):
        rows = []
        with h5py.File(C["paths"][split], "r") as f:
            for i in range(20):
                if STOP: status("paused_evaluation", split=split, scene=i); return False
                keys = INPUTS + (("gt",) if split == "RR" else ())
                a = {k:np.asarray(f[k][i], dtype=np.float32) for k in keys}
                b = {k:layout(torch.from_numpy(a[k]/2047.).unsqueeze(0).cuda()) for k in INPUTS}
                with torch.inference_mode(): pred = forward(model, b)[0].float().cpu().numpy()*2047.
                if not np.isfinite(pred).all(): raise RuntimeError("Nonfinite prediction")
                m = rr_metrics(a["gt"], pred) if split == "RR" else fr_metrics(pred, a["lms"], a["ms"], a["pan"])
                if split == "RR":
                    m["MSE_peak1"] = float(np.mean(((pred.astype(np.float64)-a["gt"])/2047.)**2))
                    if i in P["selection"]["scene_ids"]:
                        name = f"RR_scene_{i:02}.png"; display = panels(a, pred, RESULTS/name, i)
                        result["images"].append(dict(scene_id=i, file=name, display=display))
                if not all(math.isfinite(v) for v in m.values()): raise RuntimeError("Nonfinite metric")
                rows.append(dict(scene_id=i, **m)); status("evaluate", split=split, scene=i+1, scenes=20,
                                                          epoch=100, best_epoch=ck["epoch"], alpha=result["best_alpha"])
        write(RESULTS / (split+"_per_scene.json"), rows)
        result["metrics"][split] = dict(count=20, mean={k:float(np.mean([r[k] for r in rows])) for k in m}, rows=rows)
    if len(result["images"]) != 5: raise RuntimeError("Five fixed RR panels required")
    write(RESULTS / "results.json", result); status("complete", epoch=100, best_epoch=ck["epoch"], alpha=result["best_alpha"])
    return True

if __name__ == "__main__":
    try:
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True; torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.set_num_threads(2)
        if environment() != P["environment"]:
            raise RuntimeError("Worker library/device environment differs; no training started")
        write(ROOT / "provenance" / (ARM+"_environment.json"),
              dict(torch=str(torch.__version__), cuda=torch.version.cuda, python=sys.version,
                   gpu=torch.cuda.get_device_name(0), cublas=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                   deterministic=torch.are_deterministic_algorithms_enabled(), tf32=False))
        if not train(): sys.exit(75)
        if not evaluate(): sys.exit(75)
    except Exception:
        write(RUNTIME / "failure.json", dict(traceback=traceback.format_exc())); raise
'''


def bootstrap_sources(code, resume):
    code = Path(code)
    for name, expected in EXPECTED_SOURCE.items():
        p = code / name
        if not resume:
            p.parent.mkdir(parents=True, exist_ok=True)
            url = (f"https://raw.githubusercontent.com/zhouchuanxu/SSA-MRN/{UPSTREAM}/network.py"
                   if name.endswith("upstream/network.py") else
                   f"https://raw.githubusercontent.com/BIYONGHIYON/CtrS/{PINNED}/{name}")
            p.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        if not p.is_file() or sha(p) != expected:
            raise RuntimeError(f"Pinned source missing/changed: {name}; no training started")
    for name, source in (("alpha_models.py", MODEL), ("alpha_support.py", SUPPORT),
                         ("alpha_worker.py", WORKER), ("alpha_init.py", INITIALIZATION)):
        ast.parse(source)
        p = code / name
        if resume:
            if not p.is_file() or p.read_bytes() != source.encode("utf-8"):
                raise RuntimeError("Resume source differs: " + name)
        else:
            p.write_text(source, encoding="utf-8", newline="\n")
    return {str(p.relative_to(code)).replace("\\", "/"): sha(p) for p in sorted(code.rglob("*.py"))}


def load_support(code):
    # A unique module avoids notebook module caching when this cell is run again.
    spec = importlib.util.spec_from_file_location("ctrs_current_alpha_support", Path(code) / "alpha_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_manifest(paths):
    import h5py
    manifest = {}
    expected_nhw = {"train": (17139,64,64), "val": (1905,64,64), "RR": (20,256,256), "FR": (20,512,512)}
    for split, path in paths.items():
        if not Path(path).is_file(): raise FileNotFoundError(path)
        with h5py.File(path, "r") as f:
            keys = ("pan", "lms", "ms") + (() if split == "FR" else ("gt",))
            shape = {k:list(f[k].shape) for k in keys}
            n, h, w = expected_nhw[split]
            expected = dict(pan=[n,1,h,w], lms=[n,4,h,w], ms=[n,4,h//4,w//4])
            if split != "FR": expected["gt"] = [n,4,h,w]
            if shape != expected: raise ValueError(f"{split}: expected {expected}; found {shape}")
        digest = sha(path)
        if digest != EXPECTED_DATA[split]:
            raise RuntimeError(f"{split}: H5 SHA-256 differs from the fixed QB dataset; no training started")
        manifest[split] = dict(sha256=digest, shapes=shape)
        print(f"[CURRENT DATA] {split}: shape and SHA-256 verified", flush=True)
    return manifest


def environment():
    import importlib.metadata
    import platform
    import torch
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "h5py", "scipy", "scikit-image", "matplotlib", "Pillow")}
    return dict(torch=str(torch.__version__), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                python=platform.python_version(), packages=packages, gpu="Tesla T4",
                capability=list(torch.cuda.get_device_capability(0)))


def check_resume_input():
    if not REQUIRE_RESUME:
        if RESUME_INPUT: raise ValueError("RESUME_INPUT is set but REQUIRE_RESUME=False; refuse accidental fresh run")
        return None
    if not RESUME_INPUT: raise ValueError("Resume requires the exact RESUME_INPUT folder; no fresh fallback")
    src = Path(RESUME_INPUT).resolve()
    for rel in ("provenance/protocol.json", "provenance/initialization_pair.json", "provenance/initial_states.pt"):
        if not (src / rel).is_file(): raise FileNotFoundError("Required resume evidence missing: " + str(src / rel))
    for arm in ARMS:
        if not any((src / "checkpoints" / arm / name).is_file() for name in ("resume.json", "resume.prev.json")):
            raise FileNotFoundError(arm + ": resume checkpoint record missing; no fresh fallback")
    return src


def initialize_pair(code, out, val_path, resume):
    if not resume:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONHASHSEED="42")
        p = subprocess.run([sys.executable, "-I", "-u", str(code / "alpha_init.py"), str(code),
                            str(out / "provenance"), str(val_path)], env=env,
                           capture_output=True, text=True, timeout=180)
        if p.returncode != 0:
            raise RuntimeError("CPU initialization/output check failed; no training started:\n" + p.stderr)
        if "CTRS_ALPHA_INIT=" not in p.stdout: raise RuntimeError("Initialization proof not returned")
    proof = json.loads((out / "provenance/initialization_pair.json").read_text(encoding="utf-8"))
    if sha(out / "provenance/initial_states.pt") != proof["initial_states_sha256"]:
        raise RuntimeError("Shared initialization file hash differs")
    if not proof["output_max_abs"] >= 0 or any(abs(a-0.1) > 1e-7 for a in proof["alpha"].values()):
        raise ValueError("Invalid initialization proof")
    return proof


def protocol(data, sources, init, env):
    return dict(id="ssa_b1_alpha_bounded_k4_v1", arms=list(ARMS), seed=42, k=4, stages=3,
                epochs=100, batch=32, micro=32, drop_last=False, lr=1e-4,
                optimizer=dict(name="Adam", fused=True, foreach=False, betas=[0.9,0.999], eps=1e-8,
                               weight_decay=0, amsgrad=False),
                loss="final output MSE", amp="float16 + GradScaler", layout="channels_last",
                deterministic=True, cudnn_benchmark=False, tf32=False, cublas=":4096:8",
                alpha=dict(B1="unconstrained scalar 0.1", B1_alpha_bounded="sigmoid(a); a=log(0.1/0.9)"),
                validation="FP32; minimum MSE; earliest epoch on ties",
                evaluation="best; FP32; full RR20 and FR20; no test-based selection; no output clipping",
                data=data, sources=sources, initialization=init, environment=env,
                selection=dict(seed=42, scene_ids=[1,8,11,13,19], indexing="zero-based H5 row",
                               tile="whole scene", order="ascending", rgb_bands=[2,1,0],
                               selected_before_training=True, policy="fixed in source; never changed by metrics"),
                source_repo_commit=PINNED, upstream_commit=UPSTREAM)


def reject_existing_workers():
    import psutil
    names = {"kaggle_worker.py", "controlled_worker.py", "alpha_worker.py", "pair_worker.py",
             "train.py", "train_controlled.py"}
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid(): continue
        try:
            args = process.info["cmdline"] or []
            if any(Path(arg).name in names for arg in args):
                raise RuntimeError(f"Existing training process PID {process.pid}; no duplicate GPU launch")
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
    try:
        p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True, timeout=5)
        other = {int(row.strip()) for row in p.stdout.splitlines() if row.strip().isdigit()} - {os.getpid()}
        if other: raise RuntimeError(f"Other GPU processes active: {sorted(other)}; GPUs left untouched")
    except (OSError, subprocess.SubprocessError): pass


def snapshot_launcher(out):
    candidates = []
    if "__file__" in globals() and Path(__file__).is_file():
        candidates.append(Path(__file__).read_text(encoding="utf-8"))
    try:
        from IPython import get_ipython
        candidates.append(get_ipython().history_manager.input_hist_raw[-1])
    except (AttributeError, ImportError, IndexError): pass
    for candidate in candidates:
        if all(marker in candidate for marker in ("WORKER = r", "def main():", "ssa_b1_alpha_bounded_k4_v1")):
            (out / "provenance/run_cell.py").write_text(candidate, encoding="utf-8", newline="\n")
            return True
    return False


def run_workers(out, p, init, paths, support, resume_states):
    import torch
    import uuid
    from IPython.display import display, HTML, clear_output
    from html import escape
    session = uuid.uuid4().hex
    runtime = out / "runtime"; runtime.mkdir(exist_ok=True)
    processes = []; logs = []; current = {}
    clear_output(wait=True)
    live = display(HTML("<pre>Preparing current B1 / B1_alpha_bounded experiment...</pre>"), display_id=True)
    def refresh():
        lines = ["Current experiment: B1 vs B1_alpha_bounded", f"Output: {out}"]
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True, check=True, timeout=3)
            lines.extend(gpu.stdout.strip().splitlines())
        except (OSError, subprocess.SubprocessError): lines.append("GPU utilization unavailable")
        for gpu, arm in enumerate(ARMS):
            s = current.get(arm, {}); phase = s.get("phase", "starting")
            lines.append(f"GPU{gpu} {arm}: {phase} | epoch {s.get('epoch','-')}/100 | batch {s.get('batch','-')}/{s.get('batches','-')}")
            details = []
            for key in ("allocated_gb", "reserved_gb", "peak_gb", "alpha", "train_mse", "val_mse", "val_psnr", "val_sam"):
                if key in s: details.append(f"{key}={s[key]:.6g}")
            if phase == "cache": details.append(f"{s['split']}/{s['array']} {s['percent']:.1f}%")
            if phase == "evaluate": details.append(f"{s['split']} scene {s['scene']}/{s['scenes']}")
            if "eta_seconds" in s:
                details.append(f"train/validation ETA={s['eta_seconds']/60:.1f} min; {s['eta_basis']}")
            if details: lines.append("  " + " | ".join(details))
        lines.append("ETA uses measured current-run throughput; bootstrap, test and packaging are outside it.")
        live.update(HTML("<pre style='white-space:pre-wrap'>" + escape("\n".join(lines)) + "</pre>"))
    def statuses():
        for arm in ARMS:
            path = runtime / arm / "status.json"
            if path.is_file():
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("session_nonce") == session: current[arm] = value  # replace, never merge old fields
    try:
        for gpu, arm in enumerate(ARMS):
            cfg = dict(root=str(out), arm=arm, physical_gpu=gpu, protocol=p, initialization=init,
                       paths=paths, session_nonce=session, resume=REQUIRE_RESUME,
                       resume_sha256=resume_states.get(arm))
            write(out / "provenance" / (arm + "_config.json"), cfg)
            (runtime / arm).mkdir(exist_ok=True)
            log = (runtime / arm / "worker.log").open("w", encoding="utf-8", buffering=1); logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONHASHSEED="42")
            processes.append(subprocess.Popen([sys.executable, "-I", "-u", str(out / "provenance/code/alpha_worker.py"),
                                               str(out / "provenance" / (arm + "_config.json"))],
                                              env=env, stdout=log, stderr=subprocess.STDOUT))
        refresh(); gate_opened = False
        while any(process.poll() is None for process in processes):
            statuses(); refresh()
            failed = [process.returncode for process in processes if process.poll() not in (None,0)]
            if failed:
                for process in processes:
                    if process.poll() is None: process.terminate()  # only processes this cell launched
                for process in processes: process.wait(timeout=180)
                raise RuntimeError(f"Current worker stopped/failed {failed}; preserve Output and inspect current runtime logs")
            if not gate_opened and all((runtime / arm / "ready.json").is_file() for arm in ARMS):
                ready = {a:json.loads((runtime/a/"ready.json").read_text()) for a in ARMS}
                if all(v["session_nonce"] == session for v in ready.values()):
                    outputs = {a:torch.load(out/"provenance"/(a+"_initial_gpu.pt"),
                                           map_location="cpu", weights_only=True) for a in ARMS}
                    left, right = outputs.values()
                    if any(v["common_sha256"] != init["common_sha256"] for v in outputs.values()):
                        raise RuntimeError("GPU initial core/encoder differ; no training started")
                    if any(abs(v["alpha"]-0.1) > 1e-7 for v in outputs.values()): raise RuntimeError("GPU alpha differs")
                    torch.testing.assert_close(left["fp32"], right["fp32"], atol=1e-6, rtol=1e-5)
                    torch.testing.assert_close(left["amp"], right["amp"], atol=1e-3, rtol=1e-3)
                    write(out/"provenance/initialization_gpu_pair.json",
                          dict(common_sha256=init["common_sha256"], alpha={a:v["alpha"] for a,v in outputs.items()},
                               fp32_atol=1e-6, fp32_rtol=1e-5, amp_atol=1e-3, amp_rtol=1e-3,
                               fp32_max_abs=float((left["fp32"]-right["fp32"]).abs().max()),
                               amp_max_abs=float((left["amp"]-right["amp"]).abs().max()),
                               passed_before_either_arm_trained=True))
                    write(runtime / "start_training.json", dict(session_nonce=session)); gate_opened = True
            time.sleep(5)
        statuses(); refresh()
        if not gate_opened or any(process.returncode != 0 for process in processes):
            raise RuntimeError("Pair did not complete; preserve the current run for explicit resume")
    except BaseException:
        for process in processes:
            if process.poll() is None: process.terminate()
        for process in processes:
            if process.poll() is None:
                try: process.wait(timeout=180)
                except subprocess.TimeoutExpired: pass  # Never force-kill training or discard its state.
        raise
    finally:
        for log in logs: log.close()
    return live


def finish(out, p, support):
    import csv
    import math
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    res = out / "results"
    results = {a:json.loads((res/a/"results.json").read_text(encoding="utf-8")) for a in ARMS}
    histories = {}
    for arm in ARMS:
        ckdir = out/"checkpoints"/arm
        ck, record, _ = support.load_record(ckdir, p, arm); support.export_record(ckdir, record)
        if ck["epoch"] != 100 or ck["mid"] is not None: raise RuntimeError("Incomplete training")
        history = ck["history"]; histories[arm] = history
        disk = json.loads((res/arm/"history.json").read_text(encoding="utf-8"))
        if disk != history: raise RuntimeError("History differs from original latest checkpoint")
        if results[arm]["best_epoch"] != ck["best_epoch"] or results[arm]["best_validation_mse"] != ck["best"]:
            raise RuntimeError("Evaluation used the wrong best epoch")
        for split in ("RR", "FR"):
            metric = results[arm]["metrics"][split]; rows = metric["rows"]
            disk_rows = json.loads((res/arm/(split+"_per_scene.json")).read_text())
            if metric["count"] != 20 or [r["scene_id"] for r in rows] != list(range(20)) or disk_rows != rows:
                raise RuntimeError("Whole-scene evaluation incomplete")
            for key, value in metric["mean"].items():
                if not math.isfinite(value) or not math.isclose(value, float(np.mean([r[key] for r in rows])), rel_tol=1e-12, abs_tol=1e-12):
                    raise RuntimeError("Metric mean mismatch")
        if [i["scene_id"] for i in results[arm]["images"]] != p["selection"]["scene_ids"]:
            raise RuntimeError("Fixed example selection changed")
        for item in results[arm]["images"]:
            with Image.open(res/arm/item["file"]) as im:
                im.verify()
            with Image.open(res/arm/item["file"]) as im:
                if im.size != (1024,296): raise RuntimeError("RR four-panel shape mismatch")
        for kind, w in results[arm]["weights"].items():
            f = out/w["file"]
            if sha(f) != w["sha256"] or f.stat().st_size != w["bytes"]:
                raise RuntimeError("Original weight changed after evaluation")
        # A compact CSV, always derived from measured checkpoint rows.
        with (res/arm/"epochs.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
    order_equal = [r["sample_order_sha256"] for r in histories[ARMS[0]]] == [r["sample_order_sha256"] for r in histories[ARMS[1]]]
    if not order_equal: raise RuntimeError("Paired epoch batch orders differ; no controlled comparison claimed")
    left, right = (results[a] for a in ARMS)
    delta = {s:{k:right["metrics"][s]["mean"][k]-v for k,v in left["metrics"][s]["mean"].items()} for s in ("RR","FR")}
    per_scene = {s:[dict(scene_id=i, **{k:right["metrics"][s]["rows"][i][k]-v
                   for k,v in left["metrics"][s]["rows"][i].items() if k != "scene_id"}) for i in range(20)] for s in ("RR","FR")}
    directions = [("RR","PSNR",1), ("RR","SAM",-1), ("RR","MSE_peak1",-1),
                  ("FR","D_lambda",-1), ("FR","D_s",-1), ("FR","QNR",1)]
    good = [sign*delta[s][k] for s,k,sign in directions]
    rr_preserved = all(v >= 0 for v in good[:3]); fr_reduced = all(v >= 0 for v in good[3:]) and any(v > 0 for v in good[3:])
    decision = "개선" if rr_preserved and fr_reduced else "부분 개선" if any(v > 0 for v in good) else "개선 없음"
    comparison = dict(delta_candidate_minus_B1=delta, per_scene_delta=per_scene, decision=decision,
                      rr_non_degraded_in_this_seed=rr_preserved, fr_distortion_reduced_in_this_seed=fr_reduced,
                      comparison="descriptive sign of all six fixed metrics; no statistical non-inferiority claim",
                      common_initial_core_encoder=True, sample_order_equal_all_100_epochs=order_equal,
                      epoch_selection="validation MSE only; earliest minimum; no test selection")
    write(res/"paired_comparison.json", comparison); write(res/"results.json", results)
    write(res/"compact_epochs.json", histories)
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for ax,key in zip(axes.flat,("train_mse","val_mse","val_psnr","val_sam")):
        for a,h in histories.items(): ax.plot([r["epoch"] for r in h],[r[key] for r in h],label=a)
        ax.set_title(key); ax.set_xlabel("Epoch"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(res/"learning.png",dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for a,h in histories.items():
        axes[0].plot([r["epoch"] for r in h],[r["alpha"] for r in h],label=a)
        axes[1].plot([r["epoch"] for r in h],[r["alpha_raw"] for r in h],label=a)
    for ax in axes: ax.set_xlabel("Epoch"); ax.grid(alpha=.3); ax.legend()
    axes[0].set_title("Effective shared scalar alpha (measured)")
    axes[1].set_title("Optimizer parameter: alpha vs a (different coordinates)")
    fig.tight_layout(); fig.savefig(res/"alpha.png",dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(14,8))
    for ax,(split,key,_) in zip(axes.flat,directions):
        ax.bar(["B1","bounded"],[results[a]["metrics"][split]["mean"][key] for a in ARMS])
        ax.set_title(split+" "+key+(" (provisional)" if split=="FR" else ""))
    fig.tight_layout(); fig.savefig(res/"test_comparison.png",dpi=150); plt.close(fig)
    write(res/"completion_checks.json", dict(completed_epochs=100, rr_scenes_per_arm=20, fr_scenes_per_arm=20,
          fixed_rr_panels_per_arm=5, history_matches_latest=True, earliest_validation_best=True,
          common_initial_core_encoder=True, all_epoch_orders_equal=True,
          original_best_latest_hashes_valid=True, archive_remote_recovery_verified=False))
    write_report(out, p, results, comparison)
    files = {str(f.relative_to(out)).replace("\\","/"):dict(bytes=f.stat().st_size,sha256=sha(f))
             for f in sorted(out.rglob("*")) if f.is_file() and "runtime" not in f.relative_to(out).parts
             and "__pycache__" not in f.parts and f.suffix not in (".tmp",".pyc") and f.name != "manifest.json"}
    write(out/"manifest.json", dict(status="training and whole RR/FR evaluation completed", files=files,
                                   remote_archive_verified=False, limitations="Single QB seed; FR metrics provisional; no FR GT"))
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for rel in [*files,"manifest.json"]: z.write(out/rel,arcname=out.name+"/"+rel)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise RuntimeError("Archive CRC failed")
    write(archive.with_suffix(".sha256.json"), dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),
                                                   zip_crc_verified=True,remote_recovery_verified=False))
    return archive, decision


def write_report(out, p, results, comparison):
    d = comparison["delta_candidate_minus_B1"]
    lines = ["# 3단계 공유 복원의 보정 계수 범위 제한", "",
             "**변경 내용:** 공유 scalar α의 표현만 sigmoid(a)로 바꿨습니다.", "",
             "| 확인된 결과 | 상태 | 개선 여부 |", "|---|---|---|",
             f"| RR ΔPSNR {d['RR']['PSNR']:+.6f} dB, ΔSAM {d['RR']['SAM']:+.6f}°; 잠정 FR ΔQNR {d['FR']['QNR']:+.6f} | 평가 완료 | {comparison['decision']} |", "",
             "## 1. 목적과 상태", "", "두 모델을 처음부터 각각100에폭 학습하고 validation best로 RR20장·FR20장 전체를 평가했습니다. 원격 보관·복구는 아직 검증하지 않았습니다.", "",
             "## 2. 변경한 구조와 비교 조건", "",
             "QB, K4, seed42, 3단계 공유 core·error encoder·scalar α, error=0이며 encoder bias는 학습합니다. 기준 α는 제한 없는0.1, 후보는 a=log(0.1/0.9)에서 sigmoid(a)를 사용합니다.",
             "최종 출력 MSE, 고정 lr1e-4, effective=micro32, 마지막 불완전 batch 포함, 동일 fused Adam·AMP FP16·channels-last·결정론 설정을 적용했습니다. validation과 test는 FP32이고 출력을 clipping하지 않았습니다.",
             "공통 초기 core·encoder·유효 α·CPU/GPU 초기 출력 및100개 epoch 순서를 확인했습니다. [사전 고정 protocol](provenance/protocol.json)과 [초기 검사](provenance/initialization_pair.json)에 근거를 보존합니다.", "",
             "## 3. 정량 결과", "", "각 split20장 장면별 지표의 산술평균입니다. RR MSE는 peak1, PSNR은 peak2047 밴드별 평균, SAM은 도 단위입니다. FR에는 GT가 없으며 지표는 잠정 구현입니다.", "",
             "| 모델 | Best epoch | Validation MSE | Best α | RR PSNR | RR SAM | RR MSE | FR Dλ | FR Ds | FR QNR |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a,r in results.items():
        rr=r["metrics"]["RR"]["mean"]; fr=r["metrics"]["FR"]["mean"]
        lines.append(f"| {a} | {r['best_epoch']} | {r['best_validation_mse']:.9g} | {r['best_alpha']:.9g} | {rr['PSNR']:.6f} | {rr['SAM']:.6f} | {rr['MSE_peak1']:.9g} | {fr['D_lambda']:.6f} | {fr['D_s']:.6f} | {fr['QNR']:.6f} |")
    lines += ["", "장면별 전체 수치는 [결과 JSON](results/results.json)에 보존합니다.", "",
              "## 4. 대조군과의 수치 차이", "", "현재 실행의 후보−B1 차이입니다. 과거 체크포인트를 현재 대조군으로 재사용하지 않았습니다.", "",
              "| 지표 | 후보−B1 |", "|---|---:|"]
    for split,keys in (("RR",("PSNR","SAM","MSE_peak1")),("FR",("D_lambda","D_s","QNR"))):
        for key in keys: lines.append(f"| {split} {key} | {d[split][key]:+.9g} |")
    lines += ["", "[장면별 차이와 판정 기록](results/paired_comparison.json). 서로 반대 방향의 개선을 그대로 기록하며 test로 α 설정이나 epoch를 다시 선택하지 않습니다.", "",
              "## 5. 실측 곡선", "", "![학습·검증](results/learning.png)", "", "![유효 α와 원시 파라미터](results/alpha.png)", "", "![전체 RR/FR 비교](results/test_comparison.png)", "",
              "## 6. 고정 RR 5장면", "", "scene ID1,8,11,13,19를 학습 전에 고정했습니다. LR MS·PAN·예측·GT 순서입니다. 장면별 GT1-99% 대비를 MS 패널에 공통 적용하고 LR은 최근접 확대합니다. FR GT를 생성하지 않습니다."]
    for a,r in results.items():
        for im in r["images"]: lines += ["",f"![{a} RR scene {im['scene_id']}](results/{a}/{im['file']})"]
    lines += ["", "## 7. 가중치와 재현 근거", "", "| 모델 | 원본 | Epoch | SHA-256 |", "|---|---|---:|---|"]
    for a,r in results.items():
        for kind,w in r["weights"].items(): lines.append(f"| {a} | [{kind}]({w['file']}) | {w['epoch']} | {w['sha256']} |")
    lines += ["", "원본에는 model·optimizer·scaler·모든 RNG·배치 순서·커서·누적loss·history를 보존했습니다. resume.json은 best/latest 원본을 함께 가리킵니다. [완전성 검사](results/completion_checks.json)와 [manifest](manifest.json)에 파일 크기·해시를 기록합니다.",
              "Saved Output 보존·다운로드 후 ZIP해시, manifest, 두 모델 resume record와 실제 재개를 확인해야 원격 보관 완료로 판단합니다. 삭제된 VM 파일을 이 코드로 복구할 수 없습니다.", "",
              "## 8. 한계와 다음 판단", "",
              "α 제한의 효과는 실험 전 가설이었습니다. sigmoid는 범위와 함께 유효 업데이트 크기도 바꿉니다. 초기 dα/da=0.09이므로 관측 차이를 제한 없는 α의 과도한 증가만으로 설명하지 않습니다.",
              "단일 QB seed42·고정 GPU 배정의 기술적 비교이며 RR 유지의 통계적 비열등성 검정은 아닙니다. 모든 주요 RR/FR 지표의 차이를 함께 읽어야 합니다. QNR/Ds의 MATLAB 대응, Q2n/SCC 대응은 미검증이며 FR 수치로 HR 정답 정확도를 보장하지 않습니다."]
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")


def main():
    src = check_resume_input()  # Missing resume stops before downloads or any worker launch.
    for module, package in (("h5py","h5py"),("psutil","psutil"),("matplotlib","matplotlib"),
                            ("scipy","scipy"),("skimage","scikit-image"),("PIL","Pillow")):
        if importlib.util.find_spec(module) is None:
            subprocess.run([sys.executable,"-m","pip","install","--quiet",package],check=True,timeout=180)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Select T4 x2; no CPU/single-GPU fallback")
    if any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("This fixed experiment requires two T4 GPUs")
    reject_existing_workers()
    paths = dict(train=TRAIN_H5,val=VAL_H5,RR=RR_H5,FR=FR_H5)
    if len({str(Path(v).resolve()) for v in paths.values()}) != 4: raise ValueError("Distinct H5 splits required")
    paths = ensure_qb_data(paths)
    data = data_manifest(paths)
    out = Path("/kaggle/working")/("ssa_B1_alpha_K4_s42_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(exist_ok=False)
    if src:
        shutil.copytree(src,out,dirs_exist_ok=True,ignore=shutil.ignore_patterns("runtime","*.tmp","__pycache__","*.pyc"))
    code = out/"provenance/code"; code.mkdir(parents=True,exist_ok=True)
    sources = bootstrap_sources(code,src is not None)
    init = initialize_pair(code,out,paths["val"],src is not None)
    p = protocol(data,sources,init,environment())
    if src and json.loads((out/"provenance/protocol.json").read_text(encoding="utf-8")) != p:
        raise RuntimeError("Resume protocol/data/source/environment differs; neither arm launched")
    write(out/"provenance/protocol.json",p); write(out/"results/selection.json",p["selection"])
    write(out/"provenance/observation_manifest.json",
          dict(error="zero tensor at each of three stages",error_encoder="shared Conv2d(4,4,3,padding=1,bias=True)",
               bias_trainable=True,additional_observation_operator=None,registration="H5 LMS/PAN/MS as supplied; no crop/realignment",
               normalization_peak=2047,train_val_rr_fr_data=data))
    write(out/"provenance/launch.json",dict(resume=REQUIRE_RESUME,resume_input=str(src) if src else None,
          resolved_data_paths=paths,current_cell_source_saved=snapshot_launcher(out),
          output=str(out),no_previous_run_logs_copied=True))
    support = load_support(code); resume_states = {}
    if src:
        # Validate BOTH arms completely before either GPU process exists.
        for arm in ARMS:
            ck, record, origin = support.load_record(out/"checkpoints"/arm,p,arm)
            resume_states[arm] = record["latest"]["sha256"]
            write(out/"provenance"/(arm+"_resume_selection.json"),dict(pointer=origin,
                  checkpoint=record["latest"],completed_epochs=ck["epoch"],
                  completed_batches_in_epoch=ck["mid"]["batch"] if ck["mid"] else 0))
            del ck
    live = run_workers(out,p,init,paths,support,resume_states)
    archive, decision = finish(out,p,support)
    from IPython.display import HTML
    from html import escape
    live.update(HTML("<pre>Current experiment completed: 2 arms × 100 epochs, RR20 + FR20 each.\n"
                     + escape(f"{decision}; single seed; provisional FR metrics.\n{archive}\nSave/download Output; remote recovery has not been verified.")+"</pre>"))


if __name__ == "__main__":
    main()