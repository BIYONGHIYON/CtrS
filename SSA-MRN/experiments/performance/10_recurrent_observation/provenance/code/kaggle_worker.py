
import gc, hashlib, json, math, os, random, sys, time, traceback
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

def batches(split, batch_size, generator=None, shuffle=False):
    n = C["data"][split]["shapes"]["gt"][0]
    if GPU_DATA is not None:
        order = torch.randperm(n, generator=generator, device="cuda") if shuffle else torch.arange(n, device="cuda")
        for start in range(0, n, batch_size):
            indices = order[start:start + batch_size]
            batch = {k: v.index_select(0, indices) for k, v in GPU_DATA[split].items()}
            if split == "train": batch["phase"] = TRAIN_PHASE.index_select(0, indices)
            yield batch
    else:
        raise RuntimeError("GPU residency is required")

def layout(b):
    return {k: (v.contiguous(memory_format=FORMAT) if v.ndim == 4 else v) for k, v in b.items()}

def build(): return model_for(C).cuda().to(memory_format=FORMAT)

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
    seed_all(C["seed"]); gen = torch.Generator(device="cuda").manual_seed(C["seed"])
    model = build(); opt = torch.optim.Adam(model.parameters(), lr=C["lr"], foreach=True)
    core_hash = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        # All parameters and buffers must match across B1/B2 at initialization.
        core_hash.update(name.encode()); core_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    write(RUN / "initialization.json", {"seed": C["seed"], "core_sha256": core_hash.hexdigest(),
          "excluded_candidate_layers": []})
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
        order_generator = torch.Generator(device="cuda").set_state(gen.get_state())
        order_hash = hashlib.sha256(torch.randperm(C["data"]["train"]["shapes"]["gt"][0],
                     generator=order_generator, device="cuda").cpu().numpy().tobytes()).hexdigest()
        model.train(); sse = torch.zeros((), device="cuda"); count = 0
        torch.cuda.synchronize(); tick = time.monotonic(); last = tick
        status("train", epoch=epoch + 1, batch=0, batches=total_batches)
        for step, b in enumerate(batches("train", C["batch_size"], gen, True), 1):
            b = layout(b); n = len(b["gt"]); opt.zero_grad(set_to_none=True)
            for offset in range(0, n, micro):
                end = min(offset + micro, n)
                with torch.autocast("cuda", dtype=torch.float16, enabled=C["amp"]):
                    pred = model(*(b[k][offset:end] for k in INPUTS), phase=b["phase"][offset:end])
                    loss = F.mse_loss(pred.float(), b["gt"][offset:end])
                scaler.scale(loss * ((end - offset) / n)).backward()
                sse += (pred.detach().float() - b["gt"][offset:end]).square().sum()
                count += b["gt"][offset:end].numel()
            scaler.step(opt); scaler.update()
            if time.monotonic() - last >= 10:
                elapsed = time.monotonic() - tick
                prior = [r["seconds"] for r in history[-5:]]
                measured = elapsed / step * total_batches
                estimate = np.mean(prior) if prior else measured * 1.15
                status("train", epoch=epoch + 1, batch=step, batches=total_batches,
                       train_remaining_seconds=max(0, estimate - elapsed) + (C["epochs"] - epoch - 1) * estimate)
                last = time.monotonic()
        train_mse = (sse / count).item()
        if not math.isfinite(train_mse): raise RuntimeError("Nonfinite training loss")
        torch.cuda.synchronize()
        training_seconds = time.monotonic() - tick
        status("validation", epoch=epoch + 1)
        val_started = time.monotonic()
        validation = validate(model)
        validation_seconds = time.monotonic() - val_started
        improved = validation["val_mse"] < best
        if improved: best, best_epoch = validation["val_mse"], epoch + 1
        torch.cuda.synchronize()
        row = {"epoch": epoch + 1, "train_mse": train_mse, **validation,
               "seconds": time.monotonic() - tick, "train_seconds": training_seconds,
               "validation_seconds": validation_seconds,
               "train_samples_per_second": C["data"]["train"]["shapes"]["gt"][0]/training_seconds,
               "scaler_scale": scaler.get_scale(),
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
    except BaseException:
        write(RUN / "failure.json", {"traceback": traceback.format_exc(), "time": time.time()})
        traceback.print_exc(); sys.exit(1)
