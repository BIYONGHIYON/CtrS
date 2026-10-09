
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
