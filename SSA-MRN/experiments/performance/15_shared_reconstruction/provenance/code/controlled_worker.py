
import gc, hashlib, json, math, os, random, shutil, signal, sys, time, traceback
from pathlib import Path
import h5py, numpy as np, torch
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
C = json.loads(Path(sys.argv[1]).read_text()); ROOT = Path(C["root"])
sys.path.insert(0, str(ROOT / "code" / "SSA-MRN" / "src"))
sys.path.insert(0, str(ROOT / "code"))
from controlled_models import build, core, state_hash
from ssamrn.metrics import rr_metrics, fr_metrics
RUN = ROOT / C["arm"]; RUN.mkdir(exist_ok=True)
INPUTS = ("pan", "lms", "ms"); KEYS = INPUTS + ("gt",)
STOP = False
def stop(signum, frame):
    global STOP
    STOP = True
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)

def write(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8*1024**2), b""): h.update(b)
    return h.hexdigest()
def cpu(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().clone()
    if isinstance(x, dict): return {k:cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [cpu(v) for v in x]
    if isinstance(x, tuple): return tuple(cpu(v) for v in x)
    return x
def save(ck, name):
    path = RUN / name; tmp = RUN / (name + ".tmp")
    torch.save(cpu(ck), tmp)
    if name == "latest.pt" and path.exists():
        shutil.copy2(path, RUN / "latest.prev.pt.tmp")
        os.replace(RUN / "latest.prev.pt.tmp", RUN / "latest.prev.pt")
    os.replace(tmp, path)
def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
def status(phase, **kw):
    value = dict(arm=C["arm"], phase=phase, gpu=C["physical_gpu"], updated=time.time(),
                 allocated_gb=torch.cuda.memory_allocated()/1024**3,
                 peak_gb=torch.cuda.max_memory_allocated()/1024**3, **kw)
    write(RUN / "status.json", value)
    print(json.dumps(value), flush=True)
def layout(x): return x.contiguous(memory_format=torch.channels_last)
def batch(data, ids): return {k:layout(v.index_select(0, ids)) for k,v in data.items()}
def forward(model, b): return model(*(b[k] for k in INPUTS))

def load_cache():
    result = {}
    for split in ("train", "val"):
        result[split] = {}
        with h5py.File(C["paths"][split], "r") as f:
            for k in KEYS:
                shape = f[k].shape
                t = torch.empty(shape, dtype=torch.float32, device="cuda")
                for i in range(0, len(t), 128):
                    a = np.asarray(f[k][i:i+128], dtype=np.float32)
                    if not np.isfinite(a).all(): raise ValueError("Nonfinite data")
                    t[i:i+len(a)].copy_(torch.from_numpy(a / 2047.))
                result[split][k] = t
                status("cache", split=split, array=k)
    return result

@torch.inference_mode()
def validate(model, data):
    model.eval(); sse = torch.zeros((), device="cuda"); psnr = sse.clone(); sam = sse.clone()
    count = 0; n = len(data["gt"]); started = time.monotonic(); last = started
    for i in range(0, n, 32):
        b = batch(data, torch.arange(i, min(i+32,n), device="cuda"))
        p = forward(model,b).float(); g = b["gt"]; d = (p-g).square()
        sse += d.sum(); count += g.numel()
        psnr += (-10*d.mean((2,3)).clamp_min(1e-30).log10()).mean(1).sum()
        norm = p.norm(dim=1)*g.norm(dim=1); valid = norm > 0
        angle = torch.rad2deg(torch.acos(((p*g).sum(1)/norm.clamp_min(1e-30)).clamp(-1,1)))
        sam += ((angle*valid).flatten(1).sum(1)/valid.flatten(1).sum(1)).sum()
        if time.monotonic()-last >= 15:
            status("validation", batch=i//32+1, batches=math.ceil(n/32)); last=time.monotonic()
    v = dict(val_mse=(sse/count).item(), val_psnr=(psnr/n).item(), val_sam=(sam/n).item())
    if not all(math.isfinite(x) for x in v.values()): raise RuntimeError("Nonfinite validation")
    return v

def recover():
    if not C["resume"]: return None
    errors=[]
    for name in ("latest.pt", "latest.prev.pt"):
        try:
            ck=torch.load(RUN/name,map_location="cpu",weights_only=True)
            if ck["protocol"] != C["protocol"] or ck["arm"] != C["arm"]: raise ValueError("Protocol mismatch")
            e=ck["epoch"]; mid=ck["mid"]
            if not 0 <= e <= 100 or len(ck["history"]) != e: raise ValueError("Epoch cursor")
            if mid:
                n=C["shapes"]["train"]["gt"][0]; order=mid["order"]
                if e>=100 or not 0<=mid["batch"]<=math.ceil(n/32): raise ValueError("Batch cursor")
                if order.dtype!=torch.int64 or not torch.equal(order.sort().values,torch.arange(n)): raise ValueError("Invalid permutation")
            if ck["best_epoch"]:
                best=torch.load(RUN/"best.pt",map_location="cpu",weights_only=True)
                if best["protocol"]!=C["protocol"] or best["epoch"]!=ck["best_epoch"] or best["best"]!=ck["best"]:
                    raise ValueError("Latest/best transaction mismatch; restore consistent Saved Output")
            status("resume", source=name, completed_epochs=e, next_batch=mid["batch"]+1 if mid else 1)
            return ck
        except Exception as ex: errors.append(name+": "+str(ex))
    raise RuntimeError("Resume required; no fresh start. "+"; ".join(errors))

def train():
    ck=recover()  # Validate recovery BEFORE allocating the dataset or doing any training.
    seed(); model=build(C["arm"]).cuda().to(memory_format=torch.channels_last)
    actual=state_hash(core(model).state_dict())
    if actual!=C["core_sha256"]: raise RuntimeError("Common initial core differs")
    write(RUN/"initialization.json",dict(core_sha256=actual,seed=42,k=4))
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,fused=True)
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    gen=torch.Generator(device="cuda").manual_seed(42)
    data=load_cache(); history=[]; best=float("inf"); best_epoch=0; start=0; mid=None
    if ck:
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["cpu_rng"]); torch.cuda.set_rng_state(ck["cuda_rng"])
        random.setstate(ck["python_rng"]); np.random.set_state(tuple([ck["numpy_rng"][0],np.array(ck["numpy_rng"][1],dtype=np.uint32),*ck["numpy_rng"][2:]]))
        gen.set_state(ck["order_rng"])
        history=ck["history"]; best=ck["best"]; best_epoch=ck["best_epoch"]; start=ck["epoch"]; mid=ck["mid"]
        del ck
    def snapshot(e, m):
        nr=np.random.get_state()
        return dict(protocol=C["protocol"],arm=C["arm"],epoch=e,mid=m,best=best,best_epoch=best_epoch,
                    model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),history=history,
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),order_rng=gen.get_state(),
                    python_rng=random.getstate(),numpy_rng=[nr[0],nr[1].tolist(),*nr[2:]])
    # Even interruption during the first minute has an epoch-zero recovery checkpoint.
    if not C["resume"]: save(snapshot(0,None),"latest.pt")
    n=len(data["train"]["gt"]); nb=math.ceil(n/32); last_save=time.monotonic()
    for e in range(start,100):
        if mid:
            order=mid["order"].cuda(); cursor=mid["batch"]; sse=mid["sse"].cuda()
            count=mid["count"]; elapsed_before=mid["seconds"]
        else:
            order=torch.randperm(n,generator=gen,device="cuda"); cursor=0
            sse=torch.zeros((),device="cuda"); count=0; elapsed_before=0.
        order_hash=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
        tick=time.monotonic(); last_status=tick; model.train()
        def partial(b):
            return dict(order=order,batch=b,sse=sse,count=count,seconds=elapsed_before+time.monotonic()-tick)
        for j in range(cursor,nb):
            b=batch(data["train"],order[j*32:(j+1)*32]); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                pred=forward(model,b); loss=F.mse_loss(pred.float(),b["gt"])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sse+=loss.detach()*b["gt"].numel(); count+=b["gt"].numel()
            now=time.monotonic()
            if now-last_status>=15 or j+1==nb:
                torch.cuda.synchronize(); elapsed=elapsed_before+time.monotonic()-tick
                seconds_batch=elapsed/max(j+1,1)
                # ETA includes measured validation after the first completed epoch.
                val_sec=np.mean([r["validation_seconds"] for r in history]) if history else 0.
                status("train",epoch=e+1,batch=j+1,batches=nb,train_mse=(sse/count).item(),
                       eta_seconds=seconds_batch*((nb-j-1)+(99-e)*nb)+(100-e)*val_sec,
                       eta_basis="measured train + validation" if history else "measured train only; validation not yet measured")
                if not torch.isfinite(sse).item(): raise RuntimeError("Nonfinite training")
                last_status=time.monotonic()
            if STOP or time.monotonic()-last_save>=60:
                save(snapshot(e,partial(j+1)),"latest.pt"); last_save=time.monotonic()
            if STOP: status("paused",epoch=e+1,batch=j+1); return False
        # Store a completed-training cursor before validation; restart validation if interrupted.
        save(snapshot(e,partial(nb)),"latest.pt")
        torch.cuda.synchronize(); train_seconds=elapsed_before+time.monotonic()-tick
        vt=time.monotonic(); values=validate(model,data["val"]); torch.cuda.synchronize()
        row=dict(epoch=e+1,train_mse=(sse/count).item(),train_seconds=train_seconds,
                 validation_seconds=time.monotonic()-vt,sample_order_sha256=order_hash,**values)
        history.append(row)
        if values["val_mse"]<best:
            best=values["val_mse"]; best_epoch=e+1; save(snapshot(e+1,None),"best.pt")
        save(snapshot(e+1,None),"latest.pt"); write(RUN/"history.json",history)
        status("epoch_complete",epoch=e+1,best_epoch=best_epoch,**values)
        mid=None; last_save=time.monotonic()
        if STOP: status("paused",epoch=e+1); return False
    # Regenerate the compact journal from checkpoint history after an epoch-end crash.
    write(RUN/"history.json",history)
    del model,opt,scaler,data; gc.collect(); torch.cuda.empty_cache()
    return True

def panels(a,p,path,index):
    bands=[2,1,0]; ref=a["gt"][bands].transpose(1,2,0); lo,hi=np.percentile(ref,[1,99],axis=(0,1))
    def rgb(x):
        x=x[bands].transpose(1,2,0)
        return Image.fromarray((np.clip((x-lo)/np.maximum(hi-lo,1e-8),0,1)*255).astype("uint8"))
    pan=a["pan"][0]; p0,p1=np.percentile(pan,[1,99]); h,w=pan.shape
    gray=Image.fromarray((np.clip((pan-p0)/max(p1-p0,1e-8),0,1)*255).astype("uint8")).convert("RGB")
    canvas=Image.new("RGB",(4*w,h+40),"white"); draw=ImageDraw.Draw(canvas)
    for j,(name,img) in enumerate(zip(["LR MS","PAN","Prediction","GT"],[rgb(a["ms"]),gray,rgb(p),rgb(a["gt"])])):
        canvas.paste(img.resize((w,h),Image.Resampling.NEAREST),(j*w,40)); draw.text((j*w+5,5),name,fill="black")
    draw.text((5,23),f"RR scene {index} | full scene | common GT stretch",fill="black"); canvas.save(path)
    return dict(lo=lo.tolist(),hi=hi.tolist(),pan_range=[float(p0),float(p1)])

def evaluate():
    ck=torch.load(RUN/"best.pt",map_location="cpu",weights_only=True)
    if ck["protocol"]!=C["protocol"]: raise RuntimeError("Best protocol mismatch")
    model=build(C["arm"]).cuda().to(memory_format=torch.channels_last); model.load_state_dict(ck["model"]); model.eval()
    result=dict(arm=C["arm"],best_epoch=ck["epoch"],best_validation_mse=ck["best"],metrics={},images=[],weights={})
    for kind in ("best","latest"):
        p=RUN/(kind+".pt"); w=torch.load(p,map_location="cpu",weights_only=True)
        result["weights"][kind]=dict(file=str(p.relative_to(ROOT)),epoch=w["epoch"],sha256=sha(p),bytes=p.stat().st_size)
    for split in ("RR","FR"):
        rows=[]; pred_dir=RUN/"predictions"/split; pred_dir.mkdir(parents=True,exist_ok=True)
        with h5py.File(C["paths"][split],"r") as f:
            for i in range(20):
                a={k:np.asarray(f[k][i],dtype=np.float32) for k in INPUTS+(("gt",) if split=="RR" else ())}
                b={k:layout(torch.from_numpy(a[k]/2047.).unsqueeze(0).cuda()) for k in INPUTS}
                with torch.inference_mode(): p=forward(model,b)[0].float().cpu().numpy()*2047.
                if not np.isfinite(p).all(): raise RuntimeError("Nonfinite prediction")
                m=rr_metrics(a["gt"],p) if split=="RR" else fr_metrics(p,a["lms"],a["ms"],a["pan"])
                if split=="RR":
                    m["MSE_peak1"]=float(np.mean(((p.astype(np.float64)-a["gt"])/2047.)**2))
                    if i in C["selection"]["scene_ids"]:
                        name=f"RR_scene_{i:02}.png"; display=panels(a,p,RUN/name,i)
                        result["images"].append(dict(scene_id=i,file=name,display=display))
                if not all(math.isfinite(v) for v in m.values()): raise RuntimeError("Nonfinite metric")
                rows.append(dict(scene_id=i,**m)); np.savez_compressed(pred_dir/f"scene_{i:02}.npz",prediction_dn=p)
                status("evaluate",split=split,scene=i+1,scenes=20)
        write(RUN/(split+"_per_scene.json"),rows)
        result["metrics"][split]=dict(count=20,mean={k:float(np.mean([r[k] for r in rows])) for k in m},rows=rows)
    if len(result["images"])!=5: raise RuntimeError("Missing panels")
    write(RUN/"results.json",result); status("complete",epoch=100,best_epoch=result["best_epoch"])

if __name__=="__main__":
    try:
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False; torch.set_num_threads(2)
        write(RUN/"environment.json",dict(torch=str(torch.__version__),cuda=torch.version.cuda,
              python=sys.version,gpu=torch.cuda.get_device_name(0),cublas=os.environ["CUBLAS_WORKSPACE_CONFIG"]))
        if train(): evaluate()
        else: sys.exit(75)
    except Exception:
        write(RUN/"failure.json",dict(traceback=traceback.format_exc())); raise
