
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
