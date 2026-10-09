# Paste this entire file into ONE Kaggle Python cell. Select T4 x2 + Internet.
# A0: fixed gate=1. A1: learned band x direction x position gate. QB only.
EPOCHS = 100
SEEDS = [42]                   # Use [42, 43, 44] for three paired repetitions.
BATCH_SIZE = 32                # Matches the existing Kaggle baseline cell.
MICRO_BATCH = 32               # If changed, old B0 is no longer a strictly matched run.
LEARNING_RATE = 1e-4
WIDTH = 64
BLOCKS = 3
AMP = True
COMPILE = False                # Existing B0 was eager. True is a separate speed experiment.
COMPILE_MODE = "reduce-overhead"
DETERMINISTIC = True           # Matches the existing controlled B0; AMP still uses Tensor Cores.
ADAM_IMPL = "foreach"          # Existing B0: foreach Adam; fused is an optional separate profile.
ORDER_MODE = "legacy_cpu_precomputed"  # Same B0 permutations, precomputed/uploaded ONCE.
BASELINE_RUNS = {}             # e.g. {42: "/kaggle/input/old-results/runs/github_QB_baseline_k6_s42"}
# With no baseline artifact, A0/A1 run normally; B0 comparison is explicitly unverified.
# For a baseline from a different cell, the strict source audit may require adaptation.
SAVE_EVERY = 5                 # latest recovery checkpoint cadence; best saved immediately.
SWAP_GPUS = False
SAVE_PREDICTIONS = True
DATA_ROOT = "/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset"
TRAIN_H5 = f"{DATA_ROOT}/QuickBird__Training_Dataset__train_qb.h5"
VAL_H5 = f"{DATA_ROOT}/QuickBird__Training_Dataset__valid_qb.h5"
RR_H5 = f"{DATA_ROOT}/QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5"
FR_H5 = f"{DATA_ROOT}/QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5"
OUTPUT_DIR = "/kaggle/working/qb_wavelet_A0_A1_v1"
METRICS_COMMIT = "be02b7cbf03fe00681194ae9fc9419661f24cd56"

import ast, hashlib, importlib.util, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
for module, package in [("h5py", "h5py"), ("scipy", "scipy"),
                        ("skimage", "scikit-image"), ("PIL", "pillow"),
                        ("matplotlib", "matplotlib")]:
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
import h5py
import numpy as np
import torch
from IPython.display import clear_output, display, FileLink
if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
    raise RuntimeError("Enable two CUDA GPUs (T4 x2). One GPU is not sufficient for this cell.")
if not (EPOCHS > 0 and 0 < MICRO_BATCH <= BATCH_SIZE and WIDTH > 0 and BLOCKS > 0
        and LEARNING_RATE > 0 and SAVE_EVERY > 0 and SEEDS and len(set(SEEDS)) == len(SEEDS)):
    raise ValueError("Invalid experiment settings")
if ADAM_IMPL not in ("foreach", "fused") or ORDER_MODE not in ("legacy_cpu_precomputed", "gpu"):
    raise ValueError("Unknown optimizer or shuffle mode")
devices = [torch.cuda.get_device_properties(i) for i in (0, 1)]
if devices[0].name != devices[1].name or devices[0].total_memory != devices[1].total_memory:
    raise RuntimeError("Paired comparison requires two equivalent GPUs")
OUT = Path(OUTPUT_DIR)
OUT.mkdir(parents=True, exist_ok=True)
CODE = OUT / "code"
CODE.mkdir(exist_ok=True)

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""): h.update(block)
    return h.hexdigest()

def immutable_text(path, source):
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8") != source:
        raise ValueError(f"Source changed: {path}. Use a new OUTPUT_DIR.")
    path.write_text(source, encoding="utf-8")

def locate(override, suffix):
    if override and Path(override).is_file(): return Path(override).resolve()
    # Kaggle dataset mount names can change: accept an unambiguous suffix match.
    candidates = sorted(p for p in Path("/kaggle/input").rglob("*.h5") if p.name.endswith(suffix))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Set the H5 override for {suffix}; found {candidates}")
    return candidates[0].resolve()

paths = dict(zip(["train", "val", "RR", "FR"], [
    locate(TRAIN_H5, "train_qb.h5"), locate(VAL_H5, "valid_qb.h5"),
    locate(RR_H5, "test_qb_multiExm1.h5"), locate(FR_H5, "test_qb_OrigScale_multiExm1.h5")]))
if len(set(paths.values())) != 4: raise ValueError("Splits must be separate files")
data = {}
for split, path in paths.items():
    print(f"[DATA] Inspecting and hashing {split}: {path}", flush=True)
    with h5py.File(path, "r") as f:
        keys = ("ms", "pan", "lms") + (() if split == "FR" else ("gt",))
        shapes = {k: list(f[k].shape) for k in keys}
        n, c, h, w = shapes["lms"]
        if (c != 4 or n < (5 if split == "RR" else 1) or h % 4 or w % 4
            or shapes["ms"] != [n, c, h // 4, w // 4] or shapes["pan"] != [n, 1, h, w]
            or (split != "FR" and shapes["gt"] != [n, c, h, w])):
            raise ValueError(f"Expected QB DN NCHW arrays: {split}: {shapes}")
        if split in ("RR", "FR") and (h % 32 or w % 32):
            raise ValueError("Repository Q2n/QNR metrics require 32-pixel aligned scenes")
    data[split] = {"path": str(path), "shapes": shapes, "bytes": path.stat().st_size,
                   "sha256": sha(path), "normalization": "DN / 2047, no clipping"}
if len({v["sha256"] for v in data.values()}) != 4: raise ValueError("Duplicated split files")
manifest = OUT / "data_manifest.json"
if manifest.exists() and json.loads(manifest.read_text()) != data:
    raise ValueError("Data changed; use a new OUTPUT_DIR")
write_json(manifest, data)
selection = {"seed": 42, "zero_based_scene_indices": sorted(np.random.default_rng(42).choice(
    data["RR"]["shapes"]["pan"][0], 5, replace=False).tolist()),
    "protocol": "RR", "tile": "whole scene", "order": "ascending index",
    "rgb_bands_zero_based": [2, 1, 0], "fixed_before_training": True}
write_json(OUT / "selection.json", selection)

# Only the existing metric module is downloaded. Model code is fully embedded.
# No official SSA-MRN submodule, training checkout, or old weights are needed.
import urllib.request
metric_path = CODE / "metrics.py"
metric_url = f"https://raw.githubusercontent.com/BIYONGHIYON/CtrS/{METRICS_COMMIT}/SSA-MRN/src/ssamrn/metrics.py"
metric_identity = CODE / "metrics_source.json"
if not metric_path.exists():
    with urllib.request.urlopen(metric_url, timeout=45) as response:
        immutable_text(metric_path, response.read().decode("utf-8"))
metric_info = {"url": metric_url, "commit": METRICS_COMMIT, "sha256": sha(metric_path)}
if metric_identity.exists() and json.loads(metric_identity.read_text()) != metric_info:
    raise ValueError("Metric code changed; use a new OUTPUT_DIR")
write_json(metric_identity, metric_info)

MODEL = r'''
import torch
from torch import nn
from torch.nn import functional as F

def dwt(x):
    # Orthonormal Haar, top-left origin. Directions defined by these signs.
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (a+b+c+d)*0.5, torch.cat(((a-b+c-d)*0.5,
               (a+b-c-d)*0.5, (a-b-c+d)*0.5), dim=1)

def iwt(low, high):
    lh, hl, hh = high.chunk(3, dim=1)
    # Pixel shuffle expects four subpixels contiguous per output band.
    pixels = torch.stack((low+lh+hl+hh, low-lh+hl-hh,
                          low+lh-hl-hh, low-lh-hl+hh), dim=2)*0.5
    return F.pixel_shuffle(pixels.flatten(1, 2), 2)

class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                  nn.SiLU(), nn.Conv2d(width, width, 3, padding=1))
    def forward(self, x): return x + self.body(x)*0.1

class Stage(nn.Module):
    def __init__(self, bands, width, blocks):
        super().__init__()
        self.ms = nn.Sequential(nn.Conv2d(bands, width, 3, padding=1), nn.SiLU(),
                                *[ResBlock(width) for _ in range(blocks)])
        self.pan = nn.Sequential(nn.Conv2d(4, width, 3, padding=1), nn.SiLU(), ResBlock(width))
        self.joint = nn.Sequential(nn.Conv2d(2*width, width, 3, padding=1), nn.SiLU(),
                                   ResBlock(width))
        self.low = nn.Conv2d(width, bands, 3, padding=1)
        self.high_ms = nn.Conv2d(width, 3*bands, 3, padding=1)
        self.residual = nn.Conv2d(width, 3*bands, 3, padding=1)
        # PAN injection initially zero. All bands are mixed by regular convolutions.
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.low.weight); nn.init.zeros_(self.low.bias)
    def forward(self, x, pan_coeff, gate_head=None, diagnostics=False):
        fm, fp = self.ms(x), self.pan(pan_coeff)
        condition = torch.cat((fm, fp), 1)
        low = 2*x + self.low(fm)  # Learned LL; observed MS is not assumed GT LL.
        base = self.high_ms(fm)
        residual = self.residual(self.joint(condition))
        gate = None if gate_head is None else gate_head(condition).sigmoid()
        injected = residual if gate is None else gate*residual
        output = iwt(low, base + injected)
        if not diagnostics: return output
        if gate is None: gate = torch.ones_like(residual)
        # high channels are direction-major: [LH bands, HL bands, HH bands].
        shape = (len(x), 3, x.shape[1], *x.shape[-2:])
        g, r, j, m = [t.float().reshape(shape) for t in (gate, residual, injected, base)]
        axes = (0, 3, 4)
        stats = torch.stack((g.mean(axes), g.std(axes, unbiased=False),
             (g<0.05).float().mean(axes), (g>0.95).float().mean(axes),
             r.square().mean(axes), j.square().mean(axes), m.square().mean(axes)))
        return output, stats

class WaveletMS(nn.Module):
    def __init__(self, variant, bands=4, width=64, blocks=3):
        super().__init__()
        if variant not in ("A0", "A1"): raise ValueError(variant)
        self.variant = variant
        # Instantiate ALL shared modules before gates so same-seed weights match.
        self.stages = nn.ModuleList([Stage(bands, width, blocks) for _ in range(2)])
        self.gates = nn.ModuleList()
        if variant == "A1":
            for _ in range(2):
                head = nn.Conv2d(2*width, 3*bands, 1)
                nn.init.zeros_(head.weight); nn.init.constant_(head.bias, -2.)
                self.gates.append(head)
    def forward(self, pan, ms, diagnostics=False):
        fine_l, fine_h = dwt(pan)
        coarse_l, coarse_h = dwt(fine_l)
        coefficients = (torch.cat((coarse_l, coarse_h), 1), torch.cat((fine_l, fine_h), 1))
        x = ms; stats = []
        for s, stage in enumerate(self.stages):
            gate = self.gates[s] if self.variant == "A1" else None
            result = stage(x, coefficients[s], gate, diagnostics)
            if diagnostics: x, value = result; stats.append(value)
            else: x = result
        return (x, torch.stack(stats)) if diagnostics else x
'''
ast.parse(MODEL)
immutable_text(CODE / "model.py", MODEL)

WORKER = r'''
import gc, hashlib, json, math, os, random, sys, time, traceback
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from model import WaveletMS, dwt, iwt
from metrics import rr_metrics, fr_metrics
from PIL import Image, ImageDraw

C = json.loads(Path(sys.argv[1]).read_text()); RUN = Path(C["run_dir"])
RUN.mkdir(parents=True, exist_ok=True)
FORMAT = torch.channels_last

def write(path, obj):
    path = Path(path); tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); os.replace(tmp, path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8*1024**2), b""): h.update(b)
    return h.hexdigest()

def status(phase, **extra):
    write(RUN/"status.json", {"phase": phase, "variant": C["variant"],
        "epoch_target": C["epochs"], "time": time.time(),
        "allocated_gb": torch.cuda.memory_allocated()/1024**3,
        "peak_gb": torch.cuda.max_memory_allocated()/1024**3, **extra})

def seed_all():
    random.seed(C["seed"]); np.random.seed(C["seed"]); torch.manual_seed(C["seed"])
    torch.cuda.manual_seed_all(C["seed"])

def build(): return WaveletMS(C["variant"], 4, C["width"], C["blocks"]).cuda().to(memory_format=FORMAT)

def load_cache():
    keys = ("pan", "ms", "gt")  # LMS is not used by A and does not consume training VRAM.
    size = sum(np.prod(C["data"][s]["shapes"][k])*4 for s in ("train", "val") for k in keys)
    free = torch.cuda.mem_get_info()[0]
    if size > free*0.60:
        raise RuntimeError(f"FP32 data cache {size/1024**3:.2f} GB exceeds reserve. No CPU fallback.")
    cache = {}
    for split in ("train", "val"):
        cache[split] = {}
        with h5py.File(C["data"][split]["path"], "r") as f:
            for key in keys:
                shape = C["data"][split]["shapes"][key]
                tensor = torch.empty(shape, device="cuda", dtype=torch.float32, memory_format=FORMAT)
                for start in range(0, shape[0], 128):
                    a = np.asarray(f[key][start:start+128], dtype=np.float32)
                    if not np.isfinite(a).all() or a.min() < -1 or a.max() > 4095:
                        raise ValueError(f"Invalid DN array: {split}/{key}")
                    # Same NumPy FP32 DN normalization as the previous B0 cell;
                    # paid once at cache load, never on the training batch path.
                    if start == 0 and a.max() <= 2: raise ValueError("Expected DN H5, got normalized data")
                    tensor[start:start+len(a)].copy_(torch.from_numpy(a/2047.))
                # Reject normalized inputs rather than silently scaling twice.
                cache[split][key] = tensor
                status("loading_cache", array=f"{split}/{key}")
    torch.cuda.synchronize()
    return cache

def batches(cache, split, order, size):
    source = cache[split]
    for start in range(0, len(order), size):
        ids = order[start:start+size]
        # All random ordering and gathering stay on GPU; no DataLoader/CPU augmentation.
        yield {k: v.index_select(0, ids).contiguous(memory_format=FORMAT) for k,v in source.items()}

def adam(model):
    return torch.optim.Adam(model.parameters(), lr=C["lr"],
        foreach=C["adam_impl"]=="foreach", fused=C["adam_impl"]=="fused")

def order_plan(n):
    # CPU and CUDA randperm produce different permutations for the same seed.
    # Preserve old B0 order without a CPU sampler or transfer in the batch loop.
    device = "cpu" if C["order_mode"]=="legacy_cpu_precomputed" else "cuda"
    g = torch.Generator(device=device).manual_seed(C["seed"])
    plan = torch.stack([torch.randperm(n,device=device,generator=g) for _ in range(C["epochs"])])
    host = plan.cpu().numpy()
    hashes = [hashlib.sha256(row.tobytes()).hexdigest() for row in host]
    return plan.cuda(), hashes

def common_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith("gates."): continue
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def cpu_tree(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x, list): return [cpu_tree(v) for v in x]
    if isinstance(x, tuple): return tuple(cpu_tree(v) for v in x)
    return x

def save(ck, name):
    tmp = RUN/(name+".tmp"); torch.save(ck, tmp); os.replace(tmp, RUN/name)

@torch.inference_mode()
def validate(model, cache, runner):
    model.eval(); sums = torch.zeros(3, device="cuda", dtype=torch.float32)
    count = samples = 0; diagnostics = None
    n = len(cache["val"]["gt"])
    for b in batches(cache, "val", torch.arange(n, device="cuda"), C["micro"]):
        # Eager FP32 validation: independent of AMP/compile and no test-based selection.
        pred = model(b["pan"], b["ms"]).float(); gt = b["gt"]
        diff = (pred-gt).square()
        norms = pred.norm(dim=1)*gt.norm(dim=1); valid = norms > 0
        angles = torch.rad2deg(torch.acos(((pred*gt).sum(1)/norms.clamp_min(1e-30)).clamp(-1,1)))
        sam = ((angles*valid).flatten(1).sum(1)/valid.flatten(1).sum(1)).sum()
        sums += torch.stack((diff.sum(), (-10*diff.mean((2,3)).clamp_min(1e-30).log10()).mean(1).sum(), sam))
        count += gt.numel(); samples += len(gt)
        if diagnostics is None:
            # Fixed first validation batch; gate statistics are diagnostic, not a loss.
            _, diagnostics = model(b["pan"], b["ms"], diagnostics=True)
    vals = (sums/torch.tensor([count, samples, samples], device="cuda", dtype=torch.float32)).cpu().tolist()
    if not all(math.isfinite(v) for v in vals): raise RuntimeError("Nonfinite validation metrics")
    return dict(zip(("val_mse", "val_psnr_peak1", "val_sam_deg"), vals)), diagnostics.cpu().tolist()

def preflight(cache):
    status("preflight_and_compile")
    seed_all(); model = build()
    write(RUN/"initialization.json", {"common_sha256":common_hash(model),
        "parameters":sum(p.numel() for p in model.parameters()),
        "gate_parameters":sum(p.numel() for p in model.gates.parameters()),
        "seed":C["seed"], "exclude_from_common_hash":"gates only"})
    x = torch.randn(2, 4, 12, 16, device="cuda")
    low, high = dwt(x)
    if not torch.allclose(iwt(low, high), x, atol=1e-6): raise RuntimeError("Haar inversion failed")
    b = {k:v[:C["micro"]].contiguous(memory_format=FORMAT) for k,v in cache["train"].items()}
    with torch.no_grad():
        initial = model(b["pan"], b["ms"])
        zero_pan = model(torch.zeros_like(b["pan"]), b["ms"])
        if not torch.equal(initial, zero_pan): raise RuntimeError("PAN residual is not zero initialized")
        if initial.shape != b["gt"].shape: raise RuntimeError("Output must be 4x LR MS")
    runner = torch.compile(model, mode=C["compile_mode"], dynamic=False) if C["compile"] else model
    opt = adam(model)
    scaler = torch.amp.GradScaler("cuda", enabled=C["amp"])
    times = []
    for step in range(4):
        torch.cuda.synchronize(); tick = time.monotonic()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=C["amp"]):
            pred = runner(b["pan"], b["ms"])
            loss = F.mse_loss(pred.float(), b["gt"])
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize(); times.append(time.monotonic()-tick)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all().item() for g in grads):
        raise RuntimeError("Nonfinite preflight gradients")
    if C["variant"] == "A1" and not any(p.grad is not None and p.grad.abs().sum().item()>0 for p in model.gates.parameters()):
        raise RuntimeError("Learned gate has no gradient after residual warm-up")
    # Warm-up updates must not enter the actual experiment. Reset ALL state.
    seed_all(); fresh = WaveletMS(C["variant"],4,C["width"],C["blocks"])
    model.load_state_dict(fresh.state_dict()); del fresh
    opt = adam(model)
    scaler = torch.amp.GradScaler("cuda", enabled=C["amp"])
    gen = torch.Generator(device="cuda").manual_seed(C["seed"])
    write(RUN/"preflight.json", {"haar_inverse":True,"shape":list(initial.shape),
        "initial_pan_injection_zero":True,"finite_gradients":True,"reset_after_warmup":True,
        "warmup_seconds":times,"steady_microbatch_seconds":sum(times[-2:])/2,
        "compile":C["compile"], "micro_batch":C["micro"]})
    del b, x, low, high, initial, zero_pan, pred, loss, grads
    return model, runner, opt, scaler, gen

def train(cache):
    model, runner, opt, scaler, gen = preflight(cache)
    start = 0; best = float("inf"); best_epoch = 0; history = []
    if (RUN/"latest.pt").exists():
        ck = torch.load(RUN/"latest.pt", map_location="cpu", weights_only=True)
        if ck["config"] != C: raise ValueError("Resume config differs")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"]); gen.set_state(ck["order_rng"])
        torch.set_rng_state(ck["cpu_rng"]); torch.cuda.set_rng_state(ck["cuda_rng"])
        start, best, best_epoch, history = ck["epoch"], ck["best"], ck["best_epoch"], ck["history"]
        # Best may have been saved after latest before an interruption.
        if not (RUN/"best.pt").exists(): raise RuntimeError("Missing recovery best checkpoint")
        best_ck = torch.load(RUN/"best.pt", map_location="cpu", weights_only=True)
        if best_ck["config"] != C: raise ValueError("Best config differs")
        if best_ck["best"] < best: best, best_epoch = best_ck["best"], best_ck["best_epoch"]
        del ck, best_ck
    n = len(cache["train"]["gt"])
    orders, order_hashes = order_plan(n)
    expected = C.get("baseline_order_hashes")
    if expected is not None and expected != order_hashes:
        raise RuntimeError("Baseline sample order does not match. Do not claim controlled B0 comparison.")
    for epoch in range(start, C["epochs"]):
        model.train(); status("training", epoch=epoch+1)
        order = orders[epoch]; order_hash = order_hashes[epoch]
        torch.cuda.synchronize(); tick = time.monotonic(); sse = torch.zeros((), device="cuda",dtype=torch.float64)
        count = 0; last = tick; skip_before = scaler.get_scale()
        for step, b in enumerate(batches(cache,"train",order,C["batch"]),1):
            opt.zero_grad(set_to_none=True); size = len(b["gt"])
            for offset in range(0,size,C["micro"]):
                end = min(offset+C["micro"],size)
                with torch.autocast("cuda",dtype=torch.float16,enabled=C["amp"]):
                    if C["compile"]: torch.compiler.cudagraph_mark_step_begin()
                    pred = runner(b["pan"][offset:end],b["ms"][offset:end])
                    loss = F.mse_loss(pred.float(),b["gt"][offset:end])
                scaler.scale(loss*((end-offset)/size)).backward()
                sse += loss.detach().double()*b["gt"][offset:end].numel()
                count += b["gt"][offset:end].numel()
            scaler.step(opt); scaler.update()
            # No .item(), .cpu(), per-step CUDA synchronize, or progress-bar tensor reads.
            if time.monotonic()-last>20:
                status("training",epoch=epoch+1,batch=step,batches=math.ceil(n/C["batch"]))
                last=time.monotonic()
        mse = (sse/count).item(); torch.cuda.synchronize(); train_seconds = time.monotonic()-tick
        if not math.isfinite(mse): raise RuntimeError("Nonfinite training MSE")
        status("validation",epoch=epoch+1); vt=time.monotonic()
        validation, diagnostics = validate(model,cache,runner)
        improved = validation["val_mse"] < best
        if improved: best,best_epoch=validation["val_mse"],epoch+1
        row = {"epoch":epoch+1,"train_mse":mse,**validation,"train_seconds":train_seconds,
            "validation_seconds":time.monotonic()-vt,"samples_per_second":n/train_seconds,
            "sample_order_sha256":order_hash,"scaler_scale":scaler.get_scale(),
            "scale_at_epoch_start":skip_before,"peak_gb":torch.cuda.max_memory_allocated()/1024**3}
        history.append(row)
        write(RUN/"history.json",history)
        write(RUN/f"gate_epoch_{epoch+1:03}.json", {"epoch":epoch+1,"scope":"first fixed validation batch",
            "axes":["stage","statistic","direction","band"],"directions":["LH","HL","HH"],
            "statistics":["gate_mean","gate_std","fraction_lt_0.05","fraction_gt_0.95",
                          "raw_residual_energy","injected_energy","MS_high_energy"],"values":diagnostics})
        if improved or (epoch+1)%C["save_every"]==0 or epoch+1==C["epochs"]:
            status("checkpoint",epoch=epoch+1)
            ck={"config":C,"model":cpu_tree(model.state_dict()),"optimizer":cpu_tree(opt.state_dict()),
                "scaler":scaler.state_dict(),"epoch":epoch+1,"best":best,"best_epoch":best_epoch,
                "history":history,"order_rng":gen.get_state(),"cpu_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state()}
            if improved: save(ck,"best.pt")
            if (epoch+1)%C["save_every"]==0 or epoch+1==C["epochs"]: save(ck,"latest.pt")
            del ck
        print(json.dumps(row),flush=True)
    return history

def panels(arrays,pred,path,index):
    bands=[2,1,0]; gt=arrays["gt"][bands].transpose(1,2,0)
    lo,hi=np.percentile(gt,[1,99],axis=(0,1))
    def rgb(x):
        x=x[bands].transpose(1,2,0)
        return Image.fromarray((np.clip((x-lo)/np.maximum(hi-lo,1e-8),0,1)*255).astype("uint8"))
    p=arrays["pan"][0]; p0,p1=np.percentile(p,[1,99])
    pan=Image.fromarray((np.clip((p-p0)/max(p1-p0,1e-8),0,1)*255).astype("uint8")).convert("RGB")
    h,w=p.shape; canvas=Image.new("RGB",(4*w,h+40),"white"); draw=ImageDraw.Draw(canvas)
    for j,(label,image) in enumerate([("LR MS",rgb(arrays["ms"])),("PAN",pan),("Prediction",rgb(pred)),("GT",rgb(arrays["gt"]))]):
        canvas.paste(image.resize((w,h),Image.Resampling.NEAREST),(j*w,40)); draw.text((j*w+6,5),label,fill="black")
    draw.text((6,24),f"QB RR scene {index} | shared GT stretch | whole scene",fill="black"); canvas.save(path)
    return {"gt_rgb_lo":lo.tolist(),"gt_rgb_hi":hi.tolist(),"pan_range":[float(p0),float(p1)]}

def evaluate():
    status("evaluation")
    ck=torch.load(RUN/"best.pt",map_location="cpu",weights_only=True)
    model=build(); model.load_state_dict(ck["model"]); model.eval()
    result={"config":C,"best_epoch":ck["epoch"],"metrics":{},"images":[],
        "checkpoints":{k:{"sha256":sha(RUN/f"{k}.pt"),"bytes":(RUN/f"{k}.pt").stat().st_size} for k in ("best","latest")},
        "test_precision":"FP32 eager whole scene, no clipping", "FR_QNR":"provisional repository implementation"}
    del ck
    for split in ("RR","FR"):
        rows=[]
        if C["save_predictions"]: (RUN/"predictions"/split).mkdir(parents=True,exist_ok=True)
        with h5py.File(C["data"][split]["path"],"r") as f:
            for i in range(len(f["pan"])):
                status("test_"+split,scene=i+1,scenes=len(f["pan"]))
                keys=("pan","ms","lms")+(("gt",) if split=="RR" else ())
                arrays={k:np.asarray(f[k][i],dtype=np.float32) for k in keys}
                if any(not np.isfinite(a).all() for a in arrays.values()): raise ValueError("Nonfinite test input")
                inputs=[torch.from_numpy(arrays[k]/2047.).unsqueeze(0).cuda().contiguous(memory_format=FORMAT) for k in ("pan","ms")]
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                with torch.inference_mode():
                    begin.record(); output=model(*inputs); end.record(); end.synchronize()
                    milliseconds=begin.elapsed_time(end); pred=output[0].float().cpu().numpy()*2047.
                if not np.isfinite(pred).all(): raise RuntimeError("Nonfinite test output")
                if split=="RR":
                    values=rr_metrics(arrays["gt"],pred)
                    values["MSE_peak1"]=float(np.mean(((pred.astype(np.float64)-arrays["gt"])/2047.)**2))
                    if i in C["selection"]["zero_based_scene_indices"]:
                        name=f"RR_scene_{i:03}.png"
                        result["images"].append({"scene_index":i,"file":name,"display":panels(arrays,pred,RUN/name,i)})
                else: values=fr_metrics(pred,arrays["lms"],arrays["ms"],arrays["pan"])
                values["inference_ms"]=milliseconds
                if not all(math.isfinite(v) for v in values.values()): raise RuntimeError("Nonfinite scene metrics")
                rows.append({"scene_index":i,**values})
                if C["save_predictions"]:
                    np.savez_compressed(RUN/"predictions"/split/f"scene_{i:03}.npz",prediction_dn=pred,scene_index=i)
                write(RUN/f"{split}_per_scene.json",rows)
                del inputs,output
        result["metrics"][split]={"count":len(rows),"rows":rows,
            "mean":{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!="scene_index"}}
    if len(result["images"])!=5: raise RuntimeError("Missing fixed RR panels")
    write(RUN/"results.json",result); status("complete",best_epoch=result["best_epoch"])

if __name__ == "__main__":
    try:
        torch.cuda.set_device(0); torch.set_num_threads(2)
        torch.use_deterministic_algorithms(C["deterministic"])
        torch.backends.cudnn.benchmark=not C["deterministic"]
        # T4 lacks TF32; disabling it also keeps FP32 evaluation convention stable.
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        for name,expected in C["source_hashes"].items():
            if sha(Path(__file__).parent/name)!=expected: raise RuntimeError(f"Source hash mismatch: {name}")
        cache=load_cache(); train(cache)
        del cache; gc.collect(); torch.cuda.empty_cache()
        evaluate()
    except BaseException:
        write(RUN/"failure.json",{"traceback":traceback.format_exc()})
        traceback.print_exc(); sys.exit(1)
'''
ast.parse(WORKER)
immutable_text(CODE / "worker.py", WORKER)
source_hashes = {n: sha(CODE/n) for n in ("model.py", "worker.py", "metrics.py")}

def audit_baseline(seed, directory):
    # Reuse measured B0 results only when artifacts prove comparable conditions.
    # This audited worker is the current repository's original Kaggle comparison cell.
    allowed_worker = "3afeedda8b13cb9bb4ff162037c42515a77cadc025996a09e506c0a9265f75eb"
    run = Path(directory); root = run.parent.parent
    required = [run/"best.pt",run/"latest.pt",run/"results.json",run/"history.json",
                root/"code/kaggle_worker.py",root/"code/source.json",root/"environment.json",
                root/"artifact_manifest.json"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing: raise ValueError(f"Incomplete B0 evidence for seed {seed}: {missing}")
    archive_manifest=json.loads((root/"artifact_manifest.json").read_text())["artifacts"]
    for path in required[:-1]:
        # ZIP members may use POSIX or Windows separators; Kaggle writes POSIX.
        relative=path.relative_to(root).as_posix()
        record=archive_manifest.get(relative)
        if record is None or record["sha256"]!=sha(path):
            raise ValueError(f"B0 artifact manifest mismatch: {relative}")
    result = json.loads((run/"results.json").read_text()); old = result["config"]
    expected = {"sensor":"QB","k":6,"channels":4,"mode":"github","variant":"baseline",
        "epochs":EPOCHS,"seed":seed,"batch_size":BATCH_SIZE,"micro_batch":MICRO_BATCH,
        "lr":LEARNING_RATE,"amp":AMP,"channels_last":True,"controlled":DETERMINISTIC,
        "scale":2047.,"repo_commit":METRICS_COMMIT,
        "upstream_commit":"a4ca40e407b12bf4c30f804384405ce321d11c51"}
    mismatch = {k:{"B0":old.get(k),"A":v} for k,v in expected.items() if old.get(k)!=v}
    if COMPILE or ADAM_IMPL!="foreach" or ORDER_MODE!="legacy_cpu_precomputed":
        mismatch["execution"]="Audited B0 used eager/foreach Adam/CPU randperm"
    for split in ("train","val","RR","FR"):
        for key in ("sha256","shapes"):
            if old["data"][split][key]!=data[split][key]: mismatch[f"{split}/{key}"]="different data"
    if sha(root/"code/kaggle_worker.py")!=allowed_worker or old["worker_sha256"]!=allowed_worker:
        mismatch["worker_source"]="Unknown baseline trainer; audit it before comparing"
    sources = json.loads((root/"code/source.json").read_text())
    if sources["files"]["SSA-MRN/src/ssamrn/metrics.py"]["sha256"]!=source_hashes["metrics.py"]:
        mismatch["metrics"]="different evaluation implementation"
    env = json.loads((root/"environment.json").read_text())
    if env.get("torch")!=str(torch.__version__) or env.get("cuda")!=torch.version.cuda:
        mismatch["torch_cuda"]={"B0":[env.get("torch"),env.get("cuda")],"A":[str(torch.__version__),torch.version.cuda]}
    if any(g!=devices[0].name for g in env.get("gpus",[])) or not env.get("gpus"):
        mismatch["GPU"]="different or missing GPU type"
    if mismatch:
        write_json(OUT/f"baseline_rejected_s{seed}.json",mismatch)
        raise ValueError(f"B0 conditions differ: {mismatch}. Match settings or rerun B0; do not pool these results.")
    for kind in ("best","latest"):
        if sha(run/f"{kind}.pt")!=result["checkpoints"][kind]["sha256"]:
            raise ValueError(f"B0 {kind} checkpoint hash mismatch")
    latest=torch.load(run/"latest.pt",map_location="cpu",weights_only=True)
    best_ck=torch.load(run/"best.pt",map_location="cpu",weights_only=True)
    if latest["config"]!=old or best_ck["config"]!=old or latest["epoch"]!=EPOCHS:
        raise ValueError("B0 checkpoint/config/completion mismatch")
    history=json.loads((run/"history.json").read_text())
    if history!=latest["history"] or len(history)!=EPOCHS:
        raise ValueError("B0 lacks full original epoch history")
    if best_ck["epoch"]!=result["best_epoch"] or best_ck["best_epoch"]!=result["best_epoch"]:
        raise ValueError("B0 selected weight epoch mismatch")
    if best_ck["best_mse"]!=min(r["val_mse"] for r in history):
        raise ValueError("B0 selection is not validation MSE best")
    groups=latest["optimizer"]["param_groups"]
    if any(g["lr"]!=LEARNING_RATE or g["weight_decay"]!=0 or tuple(g["betas"])!=(0.9,0.999)
           or g["eps"]!=1e-8 or g.get("amsgrad",False) or not g.get("foreach") for g in groups):
        raise ValueError("B0 Adam settings differ")
    if any(r["micro_batch"]!=MICRO_BATCH for r in history):
        raise ValueError("B0 actual micro batch differs")
    for split in ("RR","FR"):
        block=result["metrics"][split]
        n=data[split]["shapes"]["pan"][0]
        if block["count"]!=n or [r["scene_index"] for r in block["rows"]]!=list(range(n)):
            raise ValueError("B0 did not score every test scene in the same order")
        for k,v in block["mean"].items():
            if not np.isclose(v,np.mean([r[k] for r in block["rows"]]),rtol=1e-12,atol=1e-12):
                raise ValueError("B0 scene average mismatch")
    params=sum(v.numel() for v in best_ck["model"].values())
    record={"seed":seed,"status":"conditions_verified","path":str(run),
        "best_epoch":result["best_epoch"],
        "config":old,"checkpoint_sha256":result["checkpoints"],"state_tensor_elements":params,
        "order_hashes":[r["sample_order_sha256"] for r in history],"metrics":result["metrics"],
        "architecture_difference":"B0 restored SSA-MRN K6 provided LMS; A raw MS coarse-to-fine Haar",
        "capacity":"Architecture and capacity intentionally differ; no equal-parameter claim",
        "reuse":"same evaluator source/data/whole-scene FP32; prior measured results, not new inference"}
    del latest,best_ck
    write_json(OUT/f"baseline_evidence_s{seed}.json",record)
    return record

baselines={}
for seed in SEEDS:
    directory=BASELINE_RUNS.get(seed,BASELINE_RUNS.get(str(seed)))
    if directory: baselines[seed]=audit_baseline(seed,directory)
write_json(OUT/"baseline_control.json",{"verified_seeds":list(baselines),
    "unverified_seeds":[s for s in SEEDS if s not in baselines],
    "defaults":"matched to local original Kaggle baseline cell; actual prior artifacts still required",
    "published_paper":"Reported paper numbers are external references; not a paired controlled baseline"})
if len(baselines)!=len(SEEDS):
    print("[B0 UNVERIFIED] A0/A1 will run; no B0 improvement claim without audited prior artifacts.",flush=True)
configs = []
for seed in SEEDS:
    for variant, device in [("A0", int(SWAP_GPUS)), ("A1", 1-int(SWAP_GPUS))]:
        run = OUT / "runs" / f"{variant}_QB_s{seed}"
        run.mkdir(parents=True, exist_ok=True)
        config = {"variant":variant,"seed":seed,"epochs":EPOCHS,"width":WIDTH,"blocks":BLOCKS,
            "batch":BATCH_SIZE,"micro":MICRO_BATCH,"lr":LEARNING_RATE,"amp":AMP,
            "compile":COMPILE,"compile_mode":COMPILE_MODE,"deterministic":DETERMINISTIC,
            "adam_impl":ADAM_IMPL,"order_mode":ORDER_MODE,
            "save_every":SAVE_EVERY,"save_predictions":SAVE_PREDICTIONS,"run_dir":str(run),
            "data":data,"selection":selection,"source_hashes":source_hashes,
            "physical_gpu":device,"loss":"final HR MSE","input":"original LR MS + PAN",
            "stages":2,"scale":4,"bands":4,"metrics_commit":METRICS_COMMIT}
        if seed in baselines: config["baseline_order_hashes"]=baselines[seed]["order_hashes"]
        cfg = CODE/f"{run.name}.json"
        if cfg.exists() and json.loads(cfg.read_text()) != config:
            raise ValueError("Settings changed. Choose a new OUTPUT_DIR; resume requires an exact match.")
        write_json(cfg, config); configs.append((config, cfg))
write_json(OUT/"environment.json", {"torch":str(torch.__version__),"cuda":torch.version.cuda,
    "python":sys.version,"gpus":[d.name for d in devices],"deterministic":DETERMINISTIC,
    "compile":COMPILE,"compile_mode":COMPILE_MODE,
    "architecture":"Two independent stages; Haar DWT PAN pyramid; MS LL + MS high + gated PAN residual",
    "B0":"previously measured externally; not rerun or reconstructed",
    "comparison":"A0 vs A1; same core initialization/order/batch/MSE; A1 adds gate parameters",
    "paper_claim":"new design hypothesis; not reproduction of official SSA-MRN architecture"})

# Both processes are started before waiting; each sees exactly one physical GPU.
# Multiple seeds form a queue on each GPU; no NCCL/DDP or cross-GPU gradient sync.
pending = list(configs); active = {}; completed = []
try:
    while pending or active:
        for device in (0,1):
            if device in active: continue
            eligible = next((i for i,(c,p) in enumerate(pending) if c["physical_gpu"]==device),None)
            if eligible is None: continue
            c,cfg=pending.pop(eligible); run=Path(c["run_dir"])
            write_json(run/"status.json",{"phase":"starting"})
            env=os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES":str(device),"OMP_NUM_THREADS":"2",
                        "MKL_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2",
                        "TORCHINDUCTOR_COMPILE_THREADS":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8"})
            log=(run/"worker.log").open("a",encoding="utf-8")
            try: proc=subprocess.Popen([sys.executable,"-u",str(CODE/"worker.py"),str(cfg)],env=env,stdout=log,stderr=subprocess.STDOUT)
            except BaseException: log.close(); raise
            active[device]=(proc,log,run)
        clear_output(wait=True)
        print(f"QB A0/A1 | finished {len(completed)}/{len(configs)} | epoch target {EPOCHS}",flush=True)
        for device,(proc,log,run) in list(active.items()):
            state=json.loads((run/"status.json").read_text())
            print(f"GPU {device}: {run.name} | {state.get('phase')} | epoch {state.get('epoch','-')} "
                  f"| batch {state.get('batch','-')} | peak {state.get('peak_gb',0):.2f} GB",flush=True)
            if proc.poll() is not None:
                log.close(); del active[device]
                if proc.returncode:
                    raise RuntimeError(f"Worker failed: {run}\n"+(run/"worker.log").read_text()[-8000:])
                completed.append(run)
        # Stop/interrupt the notebook cell to terminate both child workers.
        if pending or active: time.sleep(5)
finally:
    for proc,log,run in active.values(): proc.terminate()
    for proc,log,run in active.values():
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired: proc.kill(); proc.wait()
        log.close()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
results = {Path(c["run_dir"]).name:json.loads((Path(c["run_dir"])/"results.json").read_text()) for c,p in configs}
comparisons=[]; table=[]
for seed in SEEDS:
    names=[f"{v}_QB_s{seed}" for v in ("A0","A1")]
    init=[json.loads((OUT/"runs"/n/"initialization.json").read_text()) for n in names]
    histories=[json.loads((OUT/"runs"/n/"history.json").read_text()) for n in names]
    if init[0]["common_sha256"]!=init[1]["common_sha256"]: raise RuntimeError("Shared initialization mismatch")
    if [r["sample_order_sha256"] for r in histories[0]]!=[r["sample_order_sha256"] for r in histories[1]]:
        raise RuntimeError("Sample ordering mismatch")
    delta={s:{k:results[names[1]]["metrics"][s]["mean"][k]-v for k,v in results[names[0]]["metrics"][s]["mean"].items()} for s in ("RR","FR")}
    comparisons.append({"seed":seed,"common_initialization_equal":True,"sample_order_equal":True,
        "parameter_counts":dict(zip(names,[i["parameters"] for i in init])),"A1_minus_A0":delta})
    if seed in baselines:
        baseline = baselines[seed]
        if [r["sample_order_sha256"] for r in histories[0]]!=baseline["order_hashes"]:
            raise RuntimeError("B0 epoch order verification failed")
        comparisons[-1]["baseline_status"]="artifact conditions and full epoch ordering verified"
        comparisons[-1]["delta_from_B0"]={name:{s:{k:results[name]["metrics"][s]["mean"][k]-v
            for k,v in baseline["metrics"][s]["mean"].items() if k in results[name]["metrics"][s]["mean"] and k!="inference_ms"}
            for s in ("RR","FR")} for name in names}
    else: comparisons[-1]["baseline_status"]="unverified; B0 deltas intentionally not calculated"
    for name in names:
        r=results[name]
        for split in ("RR","FR"):
            for metric,value in r["metrics"][split]["mean"].items():
                table.append({"run":name,"best_epoch":r["best_epoch"],"protocol":split,"metric":metric,"value":value})
    fig,axes=plt.subplots(1,4,figsize=(18,4))
    for name,hist in zip(names,histories):
        for ax,key in zip(axes,["train_mse","val_mse","val_psnr_peak1","val_sam_deg"]):
            ax.plot([r["epoch"] for r in hist],[r[key] for r in hist],label=name); ax.set_title(key); ax.grid(alpha=.25)
    axes[0].set_yscale("log"); axes[1].set_yscale("log"); axes[-1].legend()
    fig.tight_layout(); fig.savefig(OUT/f"learning_s{seed}.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,(split,metric) in zip(axes.flat,[("RR","PSNR"),("RR","SAM"),("RR","MSE_peak1"),("RR","ERGAS"),("RR","Q2n"),("FR","QNR")]):
        ax.bar(["A0","A1"],[results[n]["metrics"][split]["mean"][metric] for n in names])
        ax.set_title(f"{split} {metric}"+(" (provisional)" if split=="FR" else ""))
    fig.tight_layout(); fig.savefig(OUT/f"test_comparison_s{seed}.png",dpi=150); plt.close(fig)
    if seed in baselines:
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for ax,(split,metric) in zip(axes.flat,[("RR","PSNR"),("RR","SAM"),("RR","MSE_peak1"),("RR","ERGAS"),("RR","Q2n"),("FR","QNR")]):
            values=[baselines[seed]["metrics"][split]["mean"][metric]]+[results[n]["metrics"][split]["mean"][metric] for n in names]
            ax.bar(["B0","A0","A1"],values); ax.set_title(f"{split} {metric}"+(" (provisional)" if split=="FR" else ""))
        fig.tight_layout(); fig.savefig(OUT/f"B0_test_comparison_s{seed}.png",dpi=150); plt.close(fig)
        for split in ("RR","FR"):
            for metric,value in baselines[seed]["metrics"][split]["mean"].items():
                table.append({"run":f"B0_QB_s{seed}","best_epoch":baselines[seed]["best_epoch"],
                    "protocol":split,"metric":metric,"value":value})
write_json(OUT/"results.json",results); write_json(OUT/"paired_comparison.json",comparisons)
with (OUT/"summary.csv").open("w",newline="",encoding="utf-8") as f:
    writer=csv.DictWriter(f,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
report=["# QB wavelet A0/A1", "", "Status: training and full RR/FR evaluation completed by this cell.",
    "A0 fixed gate=1; A1 learns band/direction/position gates. Two independent x2 Haar restoration stages.",
    "Original LR MS input; final MSE only. Best selected by validation MSE; FP32 whole-scene evaluation.",
    "Same common initialization, epoch sample order and batch verified. A1 has extra gate parameters.",
    f"B0 artifact-audited seeds: {list(baselines)}. Other seeds have no verified B0 comparison.",
    "B0 is previously measured restored SSA-MRN K6, not a new reproduction of published paper numbers.",
    "Structure, raw MS input path and model capacity are declared treatment differences.",
    "FR QNR is provisional; repository resize/metric MATLAB parity limitations remain.",
    "Timing measures training and validation separately; compilation/checkpoint time is excluded from training throughput.",
    "| Run | Best epoch | RR PSNR | RR SAM | FR QNR (provisional) |", "|---|---:|---:|---:|---:|"]
for name,r in results.items():
    report.append(f"| {name} | {r['best_epoch']} | {r['metrics']['RR']['mean']['PSNR']:.6f} | {r['metrics']['RR']['mean']['SAM']:.6f} | {r['metrics']['FR']['mean']['QNR']:.6f} |")
for seed in SEEDS: report.extend(["",f"![Learning](learning_s{seed}.png)",f"![Test](test_comparison_s{seed}.png)"])
for seed in baselines: report.extend(["",f"![B0 comparison](B0_test_comparison_s{seed}.png)"])
for name,r in results.items():
    report.extend(["",f"## {name} fixed RR examples"])
    report.extend(f"![Scene {i['scene_index']}](runs/{name}/{i['file']})" for i in r["images"])
(OUT/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
artifacts={str(p.relative_to(OUT)):{"bytes":p.stat().st_size,"sha256":sha(p)} for p in OUT.rglob("*")
    if p.is_file() and p.name not in ("artifact_manifest.json","status.json","worker.log","failure.json") and not p.name.endswith(".tmp")}
write_json(OUT/"artifact_manifest.json",{"artifacts":artifacts,"data":"external H5 identified by SHA256",
    "recovery":"real best/latest checkpoints include optimizer/scaler/RNG/config/history"})
archive=OUT.parent/(OUT.name+"_results.zip")
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for name in [*artifacts,"artifact_manifest.json"]: z.write(OUT/name,arcname=name)
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None: raise RuntimeError("Archive CRC failed")
write_json(archive.with_suffix(".sha256.json"),{"file":archive.name,"sha256":sha(archive)})
print(f"COMPLETE: {len(results)} runs. Saved to {OUT}")
display(FileLink(str(archive)))
