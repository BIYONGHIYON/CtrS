
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
