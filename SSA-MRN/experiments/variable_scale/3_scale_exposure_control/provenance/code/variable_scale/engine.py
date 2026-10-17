"""Training, atomic batch-boundary recovery, current-epoch status and ETA."""
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
import uuid
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
        self.invocation_id = os.environ.get("CTRS_INVOCATION_ID", uuid.uuid4().hex)
        self.log_dir = self.run_dir / "logs" / self.invocation_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.invocation_updates = self.applied_updates
        self.invocation_timing = copy.deepcopy(self.timing)
        self.log_loss_sum = torch.zeros((), device=self.device)
        self.log_sample_count = 0
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
        self.status("resume_start" if resume else "start")

    @property
    def batches(self):
        # One exposure round contains a full pass over EVERY training ratio.
        return math.ceil(self.train.length / self.config["batch_size"]) * len(self.job["scales"])

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
        measured = self.timing["measured_steps"] - self.invocation_timing["measured_steps"]
        current_val = self.timing["val_seconds"][len(self.invocation_timing["val_seconds"]):]
        current_save = self.timing["save_seconds"][len(self.invocation_timing["save_seconds"]):]
        if measured:
            eta = remaining * (self.timing["train_seconds"]-self.invocation_timing["train_seconds"]) / measured
            if current_val:
                eta += (self.config["epochs"]-self.epoch) * np.mean(current_val)
            else:
                exclusions.append("validation not yet measured")
            if current_save:
                # Lower bound: epoch saves, plus measured periodic save rate.
                eta += (self.config["epochs"]-self.epoch) * np.mean(current_save)
            exclusions.append("additional periodic checkpoint saves")
        mem = {"allocated": 0, "reserved": 0, "peak": 0}
        if self.device.type == "cuda":
            mem = {"allocated": torch.cuda.memory_allocated(self.device),
                   "reserved": torch.cuda.memory_reserved(self.device),
                   "peak": torch.cuda.max_memory_allocated(self.device)}
        loss = float(self.log_loss_sum.cpu())/self.log_sample_count if self.log_sample_count else None
        if loss is not None and not math.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        value = {"experiment_id": "3_scale_exposure_control", "run_id": self.run_dir.name,
                 "invocation_id": self.invocation_id,
                 "updates_this_invocation": self.applied_updates-self.invocation_updates,
                 "updates_total": self.applied_updates,
                 "job": self.job["name"], "device": str(self.device), "stage": stage,
                 "epoch": min(self.epoch+1, self.config["epochs"]), "epochs": self.config["epochs"],
                 "batch": self.next_batch, "batches": self.batches,
                 "epoch_percent": 100*self.next_batch/self.batches,
                 "total_percent": 100*done/total, "lr": self.config["lr"], "train_mse": loss,
                 "current_validation": validation, "memory_bytes": mem,
                 "timing": {"train_seconds": self.timing["train_seconds"]-self.invocation_timing["train_seconds"],
                            "measured_steps": measured, "val_seconds": current_val, "save_seconds": current_save},
                 "eta_training_seconds": float(eta) if eta is not None else None,
                 "eta_training_kst": (dt.datetime.now(KST)+dt.timedelta(seconds=float(eta))).isoformat()
                                      if eta is not None else "측정 대기",
                 "eta_overall": "측정 대기: evaluation/plots/archive duration not yet measured",
                 "eta_exclusions": exclusions}
        json_write(self.log_dir / "status.json", value)
        with (self.log_dir / "progress.jsonl").open("a", encoding="utf-8") as stream:
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
        self.log_loss_sum.zero_()
        self.log_sample_count = 0
        self.order_history.append({"epoch": self.epoch+1,
                                   "order_sha256": tensor_hash({"indices": self.order}),
                                   "scales": self.scale_order,
                                   "stream_protocol": "same full sample permutation per ratio per exposure round",
                                   "per_scale_order_sha256": {str(r): tensor_hash({"indices": self.order}) for r in scales}})
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
                start = (self.next_batch // len(self.job["scales"])) * self.config["batch_size"]
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
                self.log_loss_sum.add_(effective_loss*len(ids))
                self.log_sample_count += len(ids)
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
