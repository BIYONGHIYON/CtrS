# Paste this ENTIRE file into ONE Kaggle code cell. T4 x2; Internet enabled.
MODE = "train"  # First run check; after CHECK PASS change ONLY this to "train".
REQUIRE_RESUME = False
RESUME_INPUT = ""  # Explicit suite directory: /kaggle/working/... or attached Saved Output.
# Fresh execution always creates an independent timestamped directory.
DATA_ROOT = "/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset"
TRAIN_H5 = f"{DATA_ROOT}/QuickBird__Training_Dataset__train_qb.h5"
VAL_H5 = f"{DATA_ROOT}/QuickBird__Training_Dataset__valid_qb.h5"
RR_H5 = f"{DATA_ROOT}/QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5"
FR_H5 = f"{DATA_ROOT}/QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5"
# Persistence -> Files only BEFORE running; preserve/download Saved Output.
# Deleted Kaggle files cannot be recovered by this code. Resume never falls back to fresh.
import ast, hashlib, importlib.util, json, os, shutil, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path
from datetime import datetime, timezone

def sha(path, verbose=False):
    h = hashlib.sha256()
    read_bytes=0; total=Path(path).stat().st_size; pulse=time.monotonic()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024**2), b""):
            h.update(chunk); read_bytes+=len(chunk)
            if verbose and time.monotonic()-pulse>=15:
                print(f'[DATA HASH] {Path(path).name}: {100*read_bytes/total:.1f}%',flush=True); pulse=time.monotonic()
    return h.hexdigest()

def recovery_archive(out):
    files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in out.rglob('*') if p.is_file() and p.suffix not in ('.tmp','.pyc')
           and '__pycache__' not in p.parts and p.name!='recovery_manifest.json'}
    write(out/'recovery_manifest.json',dict(status='partial execution; not a completed experiment',artifacts=files))
    archive=out.parent/(out.name+'_recovery.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in [*files,'recovery_manifest.json']: z.write(out/name,arcname=out.name+'/'+name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip(): raise RuntimeError('Recovery ZIP CRC failed')
    write(archive.with_suffix('.sha256.json'),dict(file=archive.name,sha256=sha(archive),status='local recovery copy only'))
    print(f'[PARTIAL OUTPUT] {archive}; download/preserve Saved Output before ending session.',flush=True)

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def ensure_qb_data(paths, download=None):
    for split,path in paths.items():
        if not Path(path).is_file(): raise FileNotFoundError(f"{split}: {path}. Attach the confirmed dataset; no automatic fallback.")
    return dict(paths)

INITIALIZATION_CHECK = r'''
import json, sys
from pathlib import Path
code=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(code/"SSA-MRN/src"))
sys.path.insert(0,str(code))
import torch
from controlled_models import build, core, state_hash
hashes={}
for arm in ("E0","E1"):
    torch.manual_seed(42)
    model=build(arm)
    hashes[arm]=state_hash(core(model).state_dict())
    del model
if hashes["E0"]!=hashes["E1"]:
    raise RuntimeError("Initial core hashes differ")
print("CTRS_INIT_JSON="+json.dumps(hashes),flush=True)
'''

def verify_initialization(code):
    # A fresh CPU-only interpreter cannot reuse notebook sys.modules or sys.path.
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="")
    try:
        result=subprocess.run([sys.executable,"-I","-c",INITIALIZATION_CHECK,str(Path(code).resolve())],
                              env=env,capture_output=True,text=True,check=True,timeout=120)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Initial core verification failed before training:\n"+error.stderr) from error
    marker="CTRS_INIT_JSON="
    if marker not in result.stdout: raise RuntimeError("Initialization proof missing; no training started")
    hashes=json.loads(result.stdout.rsplit(marker,1)[1].strip())
    if hashes.get("E0")!=hashes.get("E1") or len(hashes.get("E0",""))!=64:
        raise RuntimeError("Invalid initial core hashes; no training started")
    return hashes

MODEL = r'''import torch
from torch import nn
from types import SimpleNamespace
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet

class CABlock(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.conv20=original.conv20; self.conv21=original.conv21; self.relu=original.relu
        self.ca=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(64,4,1),nn.ReLU(),nn.Conv2d(4,64,1))
        nn.init.zeros_(self.ca[-1].weight); nn.init.zeros_(self.ca[-1].bias)
    def forward(self,x):
        r=self.conv21(self.relu(self.conv20(x)))
        return x+2*torch.sigmoid(self.ca(r))*r

def build(arm):
    model=RestoredPansharpeningNet(4,4,1)
    if arm=="E1":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1042)
            for i in range(4):
                b=CABlock(model.backbone[i]); setattr(model,f"res{i+1}",b); model.backbone[i]=b
    elif arm!="E0": raise ValueError(arm)
    return model

def core(model):
    return SimpleNamespace(state_dict=lambda:{k:v for k,v in model.state_dict().items() if ".ca." not in k})

def state_hash(state):
    import hashlib
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
'''

WORKER = r'''

import gc, hashlib, json, math, os, random, shutil, signal, sys, time, traceback
from datetime import datetime, timezone, timedelta
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
    if isinstance(x, np.generic): return x.item()
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
                 peak_gb=torch.cuda.max_memory_allocated()/1024**3,
                 reserved_gb=torch.cuda.memory_reserved()/1024**3,lr=1e-4, **kw)
    if "eta_seconds" in value:
        value["training_end_kst"]=(datetime.now(timezone.utc)+timedelta(seconds=value["eta_seconds"])).astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")
        value["total_end_kst"]="Not yet estimated: test scoring and archive I/O remain"
    value["time_kst"]=datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")
    write(RUN / "status.json", value)
    print(json.dumps(value), flush=True)
def layout(x): return x.contiguous(memory_format=torch.channels_last)
def batch(data, ids): return {k:layout(v.index_select(0, ids)) for k,v in data.items()}
def forward(model, b): return model(*(b[k] for k in INPUTS))

def load_cache():
    required=sum(int(np.prod(s)) * 4 for split in ("train","val") for s in C["shapes"][split].values())
    free,total=torch.cuda.mem_get_info()
    if required+3*1024**3>free: raise RuntimeError(f"GPU resident cache requires {required/2**30:.2f} GiB plus 3 GiB reserve, free={free/2**30:.2f}; no silent CPU fallback")
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
        denom=valid.flatten(1).sum(1)
        if (denom==0).any(): raise RuntimeError("No valid SAM pixels")
        sam += ((angle*valid).flatten(1).sum(1)/denom).sum()
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
                best=ck["best_checkpoint"]
                if best["protocol"]!=C["protocol"] or best["epoch"]!=ck["best_epoch"] or best["best"]!=ck["best"]:
                    raise ValueError("Latest/best transaction mismatch; restore consistent Saved Output")
            if ck["best_epoch"]: save(ck["best_checkpoint"],"best.pt")
            status("resume", source=name, completed_epochs=e, next_batch=mid["batch"]+1 if mid else 1)
            return ck
        except Exception as ex: errors.append(name+": "+str(ex))
    raise RuntimeError("Resume required; no fresh start. "+"; ".join(errors))

def gpu_check():
    seed(); model=build(C["arm"]).cuda().to(memory_format=torch.channels_last)
    assert state_hash(core(model).state_dict())==C["core_sha256"]
    data=load_cache(); gen=torch.Generator(device="cuda").manual_seed(42)
    model.eval()
    with torch.inference_mode():
        initial=forward(model,batch(data['train'],torch.arange(2,device='cuda'))).float().cpu().contiguous()
    initial_output_sha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,fused=True)
    scaler=torch.amp.GradScaler("cuda")
    counter={"updates":0}
    opt.register_step_post_hook(lambda *args:counter.__setitem__("updates",counter["updates"]+1))
    def update(m,o,s,ids):
        m.train(); b=batch(data["train"],ids); o.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.float16):
            loss=F.mse_loss(forward(m,b).float(),b["gt"])
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite AMP diagnostic")
        s.scale(loss).backward(); s.step(o); s.update()
        return float(loss.detach().cpu())
    losses=[]
    for _ in range(3):
        losses.append(update(model,opt,scaler,torch.randperm(len(data["train"]["gt"]),generator=gen,device="cuda")[:32]))
    state=cpu(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
       cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),order_rng=gen.get_state()))
    torch.save(state,RUN/"diagnostic_resume.pt")
    restored=torch.load(RUN/"diagnostic_resume.pt",map_location="cpu",weights_only=True)
    ids=torch.randperm(len(data["train"]["gt"]),generator=gen,device="cuda")[:32]
    continuous_loss=update(model,opt,scaler,ids)
    expected=cpu(model.state_dict()); expected_opt=cpu(opt.state_dict()); expected_scaler=scaler.state_dict()
    other=build(C["arm"]).cuda().to(memory_format=torch.channels_last); other.load_state_dict(restored["model"])
    other_opt=torch.optim.Adam(other.parameters(),lr=1e-4,fused=True); other_opt.load_state_dict(restored["optimizer"])
    other_scaler=torch.amp.GradScaler("cuda"); other_scaler.load_state_dict(restored["scaler"])
    torch.set_rng_state(restored["cpu_rng"]); torch.cuda.set_rng_state(restored["cuda_rng"]); gen.set_state(restored["order_rng"])
    ids2=torch.randperm(len(data["train"]["gt"]),generator=gen,device="cuda")[:32]
    if not torch.equal(ids,ids2): raise RuntimeError("Resume permutation differs")
    resumed_loss=update(other,other_opt,other_scaler,ids2)
    def equal(a,b):
        if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    parity=(continuous_loss==resumed_loss and equal(expected,cpu(other.state_dict()))
            and equal(expected_opt,cpu(other_opt.state_dict())) and equal(expected_scaler,other_scaler.state_dict()))
    if not parity: raise RuntimeError("AMP checkpoint next-step parity failed; do not start training")
    if counter["updates"]==0: raise RuntimeError("All AMP updates skipped")
    # FP32 inference at training, RR and FR geometries; random probes are not test scores.
    model.eval()
    with torch.inference_mode():
        for h,w in ((64,64),(64,80),(256,256),(512,512)):
            p=forward(model,{"pan":layout(torch.rand(1,1,h,w,device="cuda")),
                "lms":layout(torch.rand(1,4,h,w,device="cuda")),"ms":layout(torch.rand(1,4,h//4,w//4,device="cuda"))})
            if p.shape!=(1,4,h,w) or not torch.isfinite(p).all(): raise RuntimeError("Inference geometry failed")
    write(RUN/"check.json",dict(passed=True,arm=C["arm"],full_training_started=False,
        initial_output_sha256=initial_output_sha256,
        resident_cache=True,batch=32,amp=True,losses=losses,actual_diagnostic_updates=counter["updates"],
        checkpoint_roundtrip=True,next_step_bitwise_equal=parity,
        geometry_checks=["64x64","64x80","256x256","512x512"],peak_gb=torch.cuda.max_memory_allocated()/2**30))
    status("check_complete")

def train():
    ck=recover()  # Validate recovery BEFORE allocating the dataset or doing any training.
    seed(); model=build(C["arm"]).cuda().to(memory_format=torch.channels_last)
    actual=state_hash(core(model).state_dict())
    if actual!=C["core_sha256"]: raise RuntimeError("Common initial core differs")
    write(RUN/"initialization.json",dict(core_sha256=actual,seed=42,k=4))
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,fused=True)
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    gen=torch.Generator(device="cuda").manual_seed(42)
    data=load_cache(); history=[]; best=float("inf"); best_epoch=0; start=0; mid=None; best_checkpoint=None
    counter={"updates":0,"attempts":0}
    opt.register_step_post_hook(lambda *args:counter.__setitem__("updates",counter["updates"]+1))
    if ck:
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["cpu_rng"]); torch.cuda.set_rng_state(ck["cuda_rng"])
        random.setstate(ck["python_rng"]); np.random.set_state(tuple([ck["numpy_rng"][0],np.array(ck["numpy_rng"][1],dtype=np.uint32),*ck["numpy_rng"][2:]]))
        gen.set_state(ck["order_rng"])
        counter.update(ck["update_counter"]); best_checkpoint=ck["best_checkpoint"]
        history=ck["history"]; best=ck["best"]; best_epoch=ck["best_epoch"]; start=ck["epoch"]; mid=ck["mid"]
        del ck
    def snapshot(e, m, include_best=True):
        nr=np.random.get_state()
        return dict(protocol=C["protocol"],arm=C["arm"],epoch=e,mid=m,best=best,best_epoch=best_epoch,
                    model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),history=history,
                    best_checkpoint=best_checkpoint if include_best else None,update_counter=dict(counter),
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),order_rng=gen.get_state(),
                    python_rng=random.getstate(),numpy_rng=[nr[0],nr[1].tolist(),*nr[2:]])
    # Even interruption during the first minute has an epoch-zero recovery checkpoint.
    if not C["resume"]: save(snapshot(0,None),"latest.pt")
    n=len(data["train"]["gt"]); nb=math.ceil(n/32); last_save=time.monotonic()
    for e in range(start,100):
        if mid:
            order=mid["order"].cuda(); cursor=mid["batch"]; sse=mid["sse"].cuda()
            count=mid["count"]; elapsed_before=mid["seconds"]; epoch_updates_before=mid["epoch_updates_before"]
        else:
            order=torch.randperm(n,generator=gen,device="cuda"); cursor=0
            sse=torch.zeros((),device="cuda"); count=0; elapsed_before=0.; epoch_updates_before=counter["updates"]
        order_hash=hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()
        tick=time.monotonic(); last_status=tick; model.train()
        def partial(b):
            return dict(order=order,batch=b,sse=sse,count=count,seconds=elapsed_before+time.monotonic()-tick,epoch_updates_before=epoch_updates_before)
        for j in range(cursor,nb):
            b=batch(data["train"],order[j*32:(j+1)*32]); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                pred=forward(model,b); loss=F.mse_loss(pred.float(),b["gt"])
            counter["attempts"]+=1
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
                       eta_basis="measured train (including checkpoint overhead) + validation" if history else "measured train only; validation not yet measured",
                       actual_updates=counter["updates"],attempted_updates=counter["attempts"],skipped_updates=counter["attempts"]-counter["updates"])
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
                 validation_seconds=time.monotonic()-vt,sample_order_sha256=order_hash,
                 actual_updates_epoch=counter["updates"]-epoch_updates_before,actual_updates_total=counter["updates"],
                 attempted_updates_total=counter["attempts"],skipped_updates_total=counter["attempts"]-counter["updates"],**values)
        history.append(row)
        if values["val_mse"]<best:
            best=values["val_mse"]; best_epoch=e+1; best_checkpoint=cpu(snapshot(e+1,None,include_best=False))
        save(snapshot(e+1,None),"latest.pt"); save(best_checkpoint,"best.pt"); write(RUN/"history.json",history)
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
        rows=[]; evaluation_started=time.monotonic()
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
                rows.append(dict(scene_id=i,**m))
                status("evaluate",split=split,scene=i+1,scenes=20,
                       split_eta_seconds=(time.monotonic()-evaluation_started)/(i+1)*(19-i),eta_scope="current split only; final archive time unmeasured")
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
        if C["mode"]=="check": gpu_check()
        elif train(): evaluate()
        else: sys.exit(75)
    except Exception:
        write(RUN/"failure.json",dict(traceback=traceback.format_exc())); raise
'''

def main():
    if MODE not in ("check","train"): raise ValueError("MODE must be check or train")
    if MODE=="check" and REQUIRE_RESUME: raise ValueError("check is a fresh diagnostic; use train to resume")
    import h5py, numpy as np, torch, psutil
    if not torch.cuda.is_available() or torch.cuda.device_count()!=2:
        raise RuntimeError("Select Kaggle T4 x2; no CPU/single-GPU fallback")
    if any("T4" not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError("This protocol requires two T4 GPUs")
    # Read-only process check. Never kill another cell's workers.
    for p in psutil.process_iter(["pid","cmdline"]):
        try:
            args=p.info["cmdline"] or []
            if any("kaggle_worker.py" in a or "controlled_worker.py" in a for a in args):
                raise RuntimeError(f"Training worker already present: PID {p.pid}; no duplicate launch")
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
    paths=dict(train=TRAIN_H5,val=VAL_H5,RR=RR_H5,FR=FR_H5)
    if len(set(paths.values()))!=4: raise ValueError("Distinct split files required")
    if REQUIRE_RESUME:
        if not RESUME_INPUT: raise ValueError("Set explicit RESUME_INPUT; no automatic experiment selection")
        src=Path(RESUME_INPUT).resolve()
        if not (src/"protocol.json").is_file(): raise FileNotFoundError("Resume protocol missing; no fresh start")
        for arm in ("E0","E1"):
            if not any((src/arm/n).is_file() for n in ("latest.pt","latest.prev.pt")):
                raise FileNotFoundError(f"{arm}: checkpoint missing; no fresh start")
    else: src=None
    paths=ensure_qb_data(paths)
    out=Path("/kaggle/working")/(f"backbone_channel_attention_{MODE}_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(exist_ok=False)
    if src: shutil.copytree(src,out,dirs_exist_ok=True,ignore=shutil.ignore_patterns("*.tmp","__pycache__"))
    try:
        cell_source=get_ipython().history_manager.input_hist_raw[-1]
        (out/"executed_cell.py").write_text(cell_source,encoding="utf-8")
    except NameError: pass
    code=out/"code"; code.mkdir(exist_ok=True)
    write(out/"controller_source.json",dict(mode=MODE,require_resume=REQUIRE_RESUME))
    pinned="54678464a559d3dabdccda8c12e9a8533bdcfeb1"
    upstream="a4ca40e407b12bf4c30f804384405ce321d11c51"
    names=["src/ssamrn/__init__.py","src/ssamrn/models/__init__.py","src/ssamrn/models/ssa_mrn.py","src/ssamrn/metrics.py"]
    urls={"SSA-MRN/"+n:f"https://raw.githubusercontent.com/BIYONGHIYON/CtrS/{pinned}/SSA-MRN/{n}" for n in names}
    urls["SSA-MRN/references/upstream/network.py"]=f"https://raw.githubusercontent.com/zhouchuanxu/SSA-MRN/{upstream}/network.py"
    expected_sources={
      "SSA-MRN/src/ssamrn/__init__.py":"4068ee713442b7896609dffeeb5c1a54d22cd19abfb31da1202b3ca45d3ca08a",
      "SSA-MRN/src/ssamrn/models/__init__.py":"8f3d7aaa5ff196e99a95c2c2926e11131f1814dd1cb235929c0cfb3124de4b88",
      "SSA-MRN/src/ssamrn/models/ssa_mrn.py":"c140c1fee05a7958b924c6058e14b31daab7ded51c1aae7122c3287c7afd8891",
      "SSA-MRN/src/ssamrn/metrics.py":"26837190e6e0d69856bbfe277f1792efd009c46157bbc1d7216cc5b872d6f2ef",
      "SSA-MRN/references/upstream/network.py":"9321e77c7dcb9a767a5ca2ec033036096df469dbfcf6d3c4ba0081c08b38849a"}
    for name,url in urls.items():
        path=code/name; path.parent.mkdir(parents=True,exist_ok=True)
        if not src: path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
        if not path.is_file(): raise FileNotFoundError(path)
        if sha(path)!=expected_sources[name]: raise ValueError("Pinned source hash mismatch: "+name)
    for name,text in [("controlled_models.py",MODEL),("controlled_worker.py",WORKER)]:
        ast.parse(text); path=code/name
        if src and path.read_text(encoding="utf-8")!=text: raise ValueError("Resume source differs")
        if not src: path.write_text(text,encoding="utf-8")
    sources={str(p.relative_to(code)):sha(p) for p in code.rglob("*.py")}
    manifest={}; shapes={}
    for split,path in paths.items():
        if not Path(path).is_file(): raise FileNotFoundError(path)
        with h5py.File(path,"r") as f:
            keys=("pan","lms","ms")+(() if split=="FR" else ("gt",))
            ranges={}
            for key in keys:
                print(f'[DATA CHECK] {split}/{key}: finite values and raw DN range',flush=True)
                pulse=time.monotonic()
                mn=float("inf"); mx=-float("inf")
                for at in range(0,len(f[key]),128):
                    values=np.asarray(f[key][at:at+128])
                    if not np.isfinite(values).all(): raise ValueError(f"{split}/{key}: nonfinite values")
                    mn=min(mn,float(values.min())); mx=max(mx,float(values.max()))
                    if time.monotonic()-pulse>=15:
                        print(f'[DATA CHECK] {split}/{key}: {100*min(at+128,len(f[key]))/len(f[key]):.1f}%',flush=True); pulse=time.monotonic()
                ranges[key]=[mn,mx]
            if ranges.get("gt",[0,0])[1]>2047: raise ValueError("GT exceeds declared QB peak2047")
            shape={k:list(f[k].shape) for k in keys}; n,c,h,w=shape["lms"]
            if c!=4 or h%4 or w%4 or shape["pan"]!=[n,1,h,w] or shape["ms"]!=[n,4,h//4,w//4]: raise ValueError(shape)
            if split!="FR" and shape["gt"]!=shape["lms"]: raise ValueError("GT shape")
            if split in ("RR","FR") and n!=20: raise ValueError("Full test must have 20 scenes")
            if split in ("train","val") and (h,w)!=(64,64): raise ValueError("Expected 64x64 training patches")
        shapes[split]=shape; manifest[split]=dict(sha256=sha(path,verbose=True),shapes=shape,raw_dn_ranges=ranges)
        print(f"[DATA] {split}: {shape}; raw DN ranges {ranges}",flush=True)
    selection=dict(seed=42,scene_ids=[1,8,11,13,19],
                   indexing="zero-based H5 row",tile="whole scene",order="ascending",rgb_bands=[2,1,0],
                   selection_rule="fixed indices from prior protocol; not random sampling",selected_before_training=True)
    protocol=dict(id="backbone_channel_attention_v1",mode=MODE,k=4,seed=42,epochs=100,batch=32,micro=32,
        optimizer="Adam fused=True; default betas/eps/weight_decay",lr=1e-4,loss="final output MSE",
        amp="float16 + GradScaler",layout="channels_last",deterministic=True,tf32=False,
        data=manifest,sources=sources,selection=selection,repo_commit=pinned,upstream_commit=upstream,
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,python=sys.version,
                         cudnn=torch.backends.cudnn.version(),gpus=[torch.cuda.get_device_name(i) for i in range(2)]))
    if src and json.loads((out/"protocol.json").read_text())!=protocol: raise ValueError("Resume protocol/data/source mismatch")
    write(out/"protocol.json",protocol); write(out/"selection.json",selection)
    hashes=verify_initialization(code)
    ha,hb=hashes["E0"],hashes["E1"]
    write(out/"initialization_pair.json",dict(E0=ha,E1=hb,equal=True))
    if src:
        # Check BOTH arms before either GPU is allowed to train.
        for arm in ("E0","E1"):
            failures=[]
            for name in ("latest.pt","latest.prev.pt"):
                try:
                    ck=torch.load(out/arm/name,map_location="cpu",weights_only=True)
                    if ck["protocol"]!=protocol or ck["arm"]!=arm: raise ValueError("Protocol mismatch")
                    if not 0<=ck["epoch"]<=100 or len(ck["history"])!=ck["epoch"]: raise ValueError("Epoch cursor")
                    if ck["best_epoch"]:
                        best=ck["best_checkpoint"]
                        if best["protocol"]!=protocol or best["epoch"]!=ck["best_epoch"] or best["best"]!=ck["best"]:
                            raise ValueError("Latest/best transaction mismatch")
                    required={"model","optimizer","scaler","cpu_rng","cuda_rng","order_rng","python_rng","numpy_rng","mid","update_counter","best_checkpoint"}
                    if not required<=ck.keys(): raise ValueError("Incomplete recovery state")
                    if ck["mid"]:
                        m=ck["mid"]; n=shapes["train"]["gt"][0]
                        if ck["epoch"]>=100 or not 0<=m["batch"]<=(n+31)//32: raise ValueError("Batch cursor")
                        if m["order"].dtype!=torch.int64 or not torch.equal(m["order"].sort().values,torch.arange(n)):
                            raise ValueError("Invalid permutation")
                    break
                except Exception as error: failures.append(name+": "+str(error))
            else: raise RuntimeError(arm+": cannot resume; neither model launched. "+"; ".join(failures))
    print(f"[RUN DIRECTORY] {out} | MODE={MODE}",flush=True)
    processes=[]; logs=[]
    try:
        for gpu,arm in enumerate(("E0","E1")):
            config=dict(root=str(out),arm=arm,physical_gpu=gpu,resume=REQUIRE_RESUME,protocol=protocol,
                        paths=paths,shapes=shapes,selection=selection,core_sha256=ha,mode=MODE)
            write(out/(arm+"_config.json"),config)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=":4096:8",PYTHONHASHSEED="42")
            (out/arm).mkdir(exist_ok=True); log=(out/arm/"worker.log").open("a",buffering=1); logs.append(log)
            processes.append(subprocess.Popen([sys.executable,"-u",str(code/"controlled_worker.py"),str(out/(arm+"_config.json"))],env=env,stdout=log,stderr=subprocess.STDOUT))
        from IPython.display import clear_output, display, HTML
        from html import escape
        clear_output(wait=True)
        live=display(HTML("<pre>현재 상태를 기다리는 중…</pre>"),display_id=True)
        last={}; current={}
        def refresh_progress():
            # Update ONE existing display object; never append status print output.
            completed=sum(s.get("phase") in ("complete","check_complete") for s in current.values())
            waiting=sum(not current.get(a) for a in ("E0","E1"))
            lines=["=== CtrS Kaggle GPU training + RR/FR evaluation ===",
                   f"Completed {completed}/2 | waiting {waiting} | GPUs 2"]
            try:
                gpu_info=subprocess.run(["nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader"],capture_output=True,text=True,check=True,timeout=3)
                lines.extend(gpu_info.stdout.strip().splitlines())
            except (OSError,subprocess.SubprocessError):
                lines.append("GPU utilization / device memory: unavailable")
            for gpu,arm in enumerate(("E0","E1")):
                s=current.get(arm,{})
                phase=s.get("phase","starting")
                label={"cache":"loading_cache","train":"training","epoch_complete":"epoch_complete"}.get(phase,phase)
                epoch=f"{s.get('epoch','-')}/100"
                pos=f"batch {s.get('batch','-')}/{s.get('batches','-')}" if phase in ("train","validation") else "batch -/-"
                allocated=f"{s['allocated_gb']:.2f}" if "allocated_gb" in s else "-"
                peak=f"{s['peak_gb']:.2f}" if "peak_gb" in s else "-"
                lines.append(f"GPU {gpu}: {arm}_QB_{arm}_k4_s42 | {label} | epoch {epoch} | {pos} | allocated {allocated} GB | peak {peak} GB")
                details=[]
                for key in (("train_mse",) if phase=="train" else ("val_mse","val_psnr","val_sam") if phase=="epoch_complete" else ()):
                    if key in s: details.append(f"{key}={s[key]:.6g}")
                if phase=="train" and "eta_seconds" in s:
                    details.append(f"training ETA {s['eta_seconds']/60:.1f} min ({s.get('eta_basis','')})")
                if "training_end_kst" in s:
                    details.append("training end "+s["training_end_kst"])
                    details.append("total end: test/archive time still unmeasured")
                if "reserved_gb" in s: details.append(f"reserved={s['reserved_gb']:.2f}GB")
                details.append("lr=0.0001")
                if phase=="cache":
                    arrays=[(split,key) for split in ("train","val") for key in ("pan","lms","ms","gt")]
                    sizes=[int(np.prod(shapes[split][key])) for split,key in arrays]
                    at=(s.get("split"),s.get("array"))
                    if at in arrays:
                        progress=100*sum(sizes[:arrays.index(at)+1])/sum(sizes)
                        details.append(f"H5 -> GPU {at[0]}/{at[1]}: {progress:.1f}%")
                if phase=="evaluate": details.append(f"{s.get('split','')} scene {s.get('scene','-')}/{s.get('scenes','-')} | split ETA {s.get('split_eta_seconds',0)/60:.1f} min")
                if details: lines.append("  "+" | ".join(details))
            lines.append(f"MODE={MODE} | Total ETA excludes unmeasured test/ZIP/I/O. All status lines are also saved in worker.log.")
            live.update(HTML("<pre style='white-space:pre-wrap'>"+escape("\n".join(lines))+"</pre>"))
        refresh_progress()
        while any(p.poll() is None for p in processes):
            changed=False
            for arm in ("E0","E1"):
                status=out/arm/"status.json"
                if status.exists():
                    text=status.read_text()
                    if last.get(arm)!=text:
                        current.setdefault(arm,{}).update(json.loads(text))
                        last[arm]=text; changed=True
            refresh_progress()  # GPU telemetry also refreshes when training status is unchanged.
            if any(p.poll() not in (None,0) for p in processes):
                for p in processes:
                    if p.poll() is None: p.terminate()
            time.sleep(5)
        for arm in ("E0","E1"):
            status=out/arm/"status.json"
            if status.exists(): current.setdefault(arm,{}).update(json.loads(status.read_text()))
        refresh_progress()
        if any(p.returncode!=0 for p in processes): raise RuntimeError(f"Worker stopped/failed: {[p.returncode for p in processes]}; preserve Output and inspect worker.log")
    except BaseException:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try: p.wait(timeout=30)
            except subprocess.TimeoutExpired: pass
        for arm in ('E0','E1'):
            path=out/arm/'worker.log'
            if path.exists(): print(f'[{arm} LOG TAIL]\n'+'\n'.join(path.read_text(errors='replace').splitlines()[-20:]),flush=True)
        if all(p.poll() is not None for p in processes):
            try: recovery_archive(out)
            except Exception as error: print(f'Recovery ZIP failed: {error}; preserve directory {out}',flush=True)
        else: print(f'Worker still finishing a safe boundary; do not rerun yet. Preserve {out}',flush=True)
        raise
    finally:
        for log in logs: log.close()
    if MODE=="check":
        checks={a:json.loads((out/a/"check.json").read_text()) for a in ("E0","E1")}
        if not all(c["passed"] for c in checks.values()): raise RuntimeError("GPU check failed")
        if checks['E0']['initial_output_sha256']!=checks['E1']['initial_output_sha256']:
            raise RuntimeError('Initial channels-last GPU outputs differ; inspect before training')
        write(out/"checks.json",checks)
        print("CHECK PASS: both GPUs, initial outputs equal, full resident train/val cache, batch32 AMP, checkpoint roundtrip and next-step parity passed. Full training has NOT started.",flush=True)
        print(f"Diagnostic output: {out}",flush=True)
        return
    results={arm:json.loads((out/arm/"results.json").read_text()) for arm in ("E0","E1")}
    histories={arm:json.loads((out/arm/"history.json").read_text()) for arm in results}
    if any(len(h)!=100 for h in histories.values()): raise RuntimeError("Incomplete history")
    if [r["sample_order_sha256"] for r in histories["E0"]]!=[r["sample_order_sha256"] for r in histories["E1"]]: raise RuntimeError("Data order differs")
    for a in results:
        for split in ("RR","FR"):
            rows=results[a]["metrics"][split]["rows"]
            if len(rows)!=20 or [r["scene_id"] for r in rows]!=list(range(20)): raise RuntimeError("Incomplete scene evaluation")
        if len(results[a]["images"])!=5: raise RuntimeError("Incomplete image selection")
    write(out/"update_counts.json",{a:{"attempts":histories[a][-1]["attempted_updates_total"],"actual":histories[a][-1]["actual_updates_total"],"skipped":histories[a][-1]["skipped_updates_total"]} for a in results})
    delta={s:{k:results["E1"]["metrics"][s]["mean"][k]-v for k,v in results["E0"]["metrics"][s]["mean"].items()} for s in ("RR","FR")}
    write(out/"paired_comparison.json",dict(delta_E1_minus_E0=delta,common_core=True,sample_order=True))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key in zip(axes.flat,("train_mse","val_mse","val_psnr","val_sam")):
        for arm,h in histories.items(): ax.plot([r["epoch"] for r in h],[r[key] for r in h],label=arm)
        ax.set_title(key); ax.set_xlabel("Epoch"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out/"learning.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,(s,k) in zip(axes.flat,[("RR","PSNR"),("RR","SAM"),("RR","MSE_peak1"),("RR","ERGAS"),("RR","Q2n"),("FR","QNR")]):
        ax.bar(["E0","E1"],[results[a]["metrics"][s]["mean"][k] for a in results]); ax.set_title(s+" "+k+(" (provisional)" if s=="FR" else ""))
    fig.tight_layout(); fig.savefig(out/"test_comparison.png",dpi=150); plt.close(fig)
    write(out/"results.json",results)
    lines=["# QB E0/E1 K4 controlled experiment", "",
        "| Reference | Change | Observed difference | Decision |",
        "|---|---|---|---|",
        f"| Fresh E0 | E1 backbone channel attention | RR PSNR {delta['RR']['PSNR']:+.6f} dB; provisional FR QNR {delta['FR']['QNR']:+.6f} | Single seed; confirm with additional paired seeds |",
        "", "## 1. Purpose and status", "",
        "Both models completed 100 epochs and full RR20/FR20 evaluation. Actual optimizer/skipped update counts are in update_counts.json. Remote archival has not been verified.",
        "", "## 2. Changes and conditions", "",
        "QB; K4; seed42; batch=micro32; fused Adam; fixed lr1e-4; final output MSE; AMP FP16; channels-last; strict deterministic; TF32 disabled.",
        "E0 is the original K4 adapter. E1 adds CA to the residual branch of its four 64-channel backbone blocks: GAP, 64-to-4-to-64, 2*sigmoid; last projection zero initialized. No change to SSAI, interpolation, loss or recurrence.",
        "Identical initial core hashes and all 100 epoch order hashes were checked. See protocol.json and initialization_pair.json.",
        "Best selected by FP32 validation MSE; test was not used for epoch/configuration selection.",
        "", "## 3. Quantitative results", "",
        "| Model | Best epoch | Validation MSE | RR PSNR | RR SAM | RR MSE peak1 | FR QNR provisional |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm,r in results.items():
        rr=r["metrics"]["RR"]["mean"]; fr=r["metrics"]["FR"]["mean"]
        lines.append(f"| {arm} | {r['best_epoch']} | {r['best_validation_mse']:.9g} | {rr['PSNR']:.6f} | {rr['SAM']:.6f} | {rr['MSE_peak1']:.9g} | {fr['QNR']:.6f} |")
    lines += ["", "Scene rows and all metrics: [results](results.json), [paired differences](paired_comparison.json).",
        "", "## 4. Previous research differences", "",
        "Historical interpolation, recurrence, output-detail and auxiliary-loss experiments differ in recipe. This comparison estimates the CA-module effect, including 2320 extra parameters; it does not isolate parameter count.",
        "Current E1-minus-E0 values are in paired_comparison.json. They are a single-seed observation.",
        "", "## 5. Measured curves", "", "![Measured learning](learning.png)", "", "![Test comparison](test_comparison.png)",
        "", "## 6. Fixed RR panels", "", "Selection was fixed before training: [selection.json](selection.json). MS panels share GT 1-99% contrast; LR uses nearest-neighbor enlargement. FR has no GT."]
    for arm,r in results.items():
        for item in r["images"]: lines += ["",f"![{arm} scene {item['scene_id']}]({arm}/{item['file']})"]
    lines += ["", "## 7. Weights and evidence", "", "| Model | Weight | Epoch | SHA-256 |", "|---|---|---:|---|"]
    for arm,r in results.items():
        for kind,w in r["weights"].items(): lines.append(f"| {arm} | [{kind}]({w['file']}) | {w['epoch']} | {w['sha256']} |")
    lines += ["", "Original best/latest include optimizer, scaler, RNG, batch cursor and measured history. artifact_manifest.json records artifact bytes/hashes; ZIP CRC and archive SHA-256 are verified locally. Preserve Saved Output and verify remote re-download/recovery before declaring archival complete.",
        "", "## 8. Limitations and next decision", "",
        "FR has no high-resolution GT. QNR/D_s use scikit-image bicubic rather than MATLAB-bit-exact resizing; Q2n/SCC parity also remains limited. Improvements are not guaranteed. Seed43 was not launched; decide after reviewing RR/FR differences and measured cost."]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    required=["learning.png","test_comparison.png","report.md","protocol.json","initialization_pair.json"]
    for a in results:
        required += [f"{a}/best.pt",f"{a}/latest.pt",f"{a}/history.json",f"{a}/RR_per_scene.json",f"{a}/FR_per_scene.json"]
        required += [f"{a}/RR_scene_{i:02}.png" for i in selection["scene_ids"]]
    if not all((out/p).is_file() for p in required): raise RuntimeError("Missing required artifact")
    artifacts={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob("*") if p.is_file() and p.suffix not in (".tmp",".pyc") and "__pycache__" not in p.parts and p.name!="artifact_manifest.json"}
    write(out/"artifact_manifest.json",dict(status="training + full RR/FR completed",artifacts=artifacts,
        limitations="FR has no GT. QNR/D_s bicubic MATLAB parity and Q2n/SCC parity remain unverified; seed43 requires a separate human decision."))
    archive=out.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for rel in [*artifacts,"artifact_manifest.json"]: z.write(out/rel,arcname=out.name+"/"+rel)
    with zipfile.ZipFile(archive) as z:
        if z.testzip(): raise RuntimeError("Archive CRC failed")
        for rel,record in artifacts.items():
            h=hashlib.sha256()
            with z.open(out.name+'/'+rel) as stream:
                for chunk in iter(lambda:stream.read(8*1024**2),b''): h.update(chunk)
            if h.hexdigest()!=record['sha256']: raise RuntimeError('ZIP artifact hash mismatch: '+rel)
    write(archive.with_suffix(".sha256.json"),dict(file=archive.name,sha256=sha(archive)))
    print(f"[COMPLETE] {archive}; verify/download Saved Output. Seed43 was NOT launched.",flush=True)
    try:
        from IPython.display import FileLink
        display(FileLink(str(archive)))
    except ImportError: pass

if __name__ == "__main__":
    main()