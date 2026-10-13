
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
