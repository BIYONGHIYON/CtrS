

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
