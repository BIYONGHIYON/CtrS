# Paste this ENTIRE file into ONE Kaggle code cell. T4 x2; Internet enabled.
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

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024**2), b""): h.update(chunk)
    return h.hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)

def ensure_qb_data(paths, download=None):
    missing={split:path for split,path in paths.items() if not Path(path).is_file()}
    if not missing: return dict(paths)
    handle="biyonghiyon/ctrs-pan-sharpening-dataset"
    print(f"[DATA] QB 파일 {len(missing)}개 없음. Kaggle 데이터셋 연결/다운로드: {handle}",flush=True)
    if download is None:
        if importlib.util.find_spec("kagglehub") is None:
            subprocess.run([sys.executable,"-m","pip","install","kagglehub"],check=True,timeout=180)
        import kagglehub
        download=kagglehub.dataset_download
    try:
        root=Path(download(handle))
    except Exception as error:
        raise RuntimeError("QB 자동 다운로드 실패. Kaggle Internet 설정과 데이터셋 접근 권한을 확인하거나 Add Input으로 데이터셋을 연결하세요. 학습은 시작하지 않았습니다.") from error
    if not root.is_dir(): raise FileNotFoundError(f"다운로드한 데이터셋 폴더 없음: {root}")
    result=dict(paths)
    for split,path in missing.items():
        name=Path(path).name; direct=root/name
        matches=[direct] if direct.is_file() else sorted(p for p in root.rglob(name) if p.is_file())
        if len(matches)!=1:
            raise FileNotFoundError(f"{split}: 데이터셋 안에서 {name} 파일을 하나로 확인할 수 없습니다: {matches}. 학습은 시작하지 않았습니다.")
        result[split]=str(matches[0].resolve())
        print(f"[DATA] {split}: {result[split]}",flush=True)
    return result

INITIALIZATION_CHECK = r'''
import json, sys
from pathlib import Path
code=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(code/"SSA-MRN/src"))
sys.path.insert(0,str(code))
import torch
from controlled_models import build, core, state_hash
hashes={}
for arm in ("B0","B1"):
    torch.manual_seed(42)
    model=build(arm)
    hashes[arm]=state_hash(core(model).state_dict())
    del model
if hashes["B0"]!=hashes["B1"]:
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
    if hashes.get("B0")!=hashes.get("B1") or len(hashes.get("B0",""))!=64:
        raise RuntimeError("Invalid initial core hashes; no training started")
    return hashes

MODEL = r'''
import torch
from torch import nn
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet

class SharedThreeStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = RestoredPansharpeningNet(4, 4)
        self.error_encoder = nn.Conv2d(4, 4, 3, padding=1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, pan, lms, ms):
        current = lms
        for _ in range(3):
            conditioned = current + self.error_encoder(torch.zeros_like(current))
            proposal = self.core(pan, conditioned, ms)
            current = current + self.alpha * (proposal - current)
        return current

def build(arm):
    if arm == "B0": return RestoredPansharpeningNet(4, 4)
    if arm == "B1": return SharedThreeStage()
    raise ValueError(arm)

def core(model):
    return model.core if isinstance(model, SharedThreeStage) else model

def state_hash(state):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
'''

WORKER = r'''
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
'''

def main():
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
        for arm in ("B0","B1"):
            if not any((src/arm/n).is_file() for n in ("latest.pt","latest.prev.pt")):
                raise FileNotFoundError(f"{arm}: checkpoint missing; no fresh start")
    else: src=None
    paths=ensure_qb_data(paths)
    out=Path("/kaggle/working")/("ssa_B0_B1_K4_s42_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(exist_ok=False)
    if src: shutil.copytree(src,out,dirs_exist_ok=True,ignore=shutil.ignore_patterns("*.tmp","__pycache__"))
    code=out/"code"; code.mkdir(exist_ok=True)
    pinned="be02b7cbf03fe00681194ae9fc9419661f24cd56"
    upstream="a4ca40e407b12bf4c30f804384405ce321d11c51"
    names=["src/ssamrn/__init__.py","src/ssamrn/models/__init__.py","src/ssamrn/models/ssa_mrn.py","src/ssamrn/metrics.py"]
    urls={"SSA-MRN/"+n:f"https://raw.githubusercontent.com/BIYONGHIYON/CtrS/{pinned}/SSA-MRN/{n}" for n in names}
    urls["SSA-MRN/references/upstream/network.py"]=f"https://raw.githubusercontent.com/zhouchuanxu/SSA-MRN/{upstream}/network.py"
    for name,url in urls.items():
        path=code/name; path.parent.mkdir(parents=True,exist_ok=True)
        if not src: path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
        if not path.is_file(): raise FileNotFoundError(path)
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
            shape={k:list(f[k].shape) for k in keys}; n,c,h,w=shape["lms"]
            if c!=4 or h%4 or w%4 or shape["pan"]!=[n,1,h,w] or shape["ms"]!=[n,4,h//4,w//4]: raise ValueError(shape)
            if split!="FR" and shape["gt"]!=shape["lms"]: raise ValueError("GT shape")
            if split in ("RR","FR") and n!=20: raise ValueError("Full test must have 20 scenes")
            if split in ("train","val") and (h,w)!=(64,64): raise ValueError("Expected 64x64 training patches")
        shapes[split]=shape; manifest[split]=dict(sha256=sha(path),shapes=shape)
        print(f"[DATA] {split}: {shape}",flush=True)
    selection=dict(seed=42,scene_ids=sorted(np.random.default_rng(42).choice(20,5,replace=False).tolist()),
                   indexing="zero-based H5 row",tile="whole scene",order="ascending",rgb_bands=[2,1,0],
                   selected_before_training=True)
    protocol=dict(id="ssa_b0_b1_k4_controlled_v1",k=4,seed=42,epochs=100,batch=32,micro=32,
        optimizer="Adam fused=True; default betas/eps/weight_decay",lr=1e-4,loss="final output MSE",
        amp="float16 + GradScaler",layout="channels_last",deterministic=True,tf32=False,
        data=manifest,sources=sources,selection=selection,repo_commit=pinned,upstream_commit=upstream)
    if src and json.loads((out/"protocol.json").read_text())!=protocol: raise ValueError("Resume protocol/data/source mismatch")
    write(out/"protocol.json",protocol); write(out/"selection.json",selection)
    hashes=verify_initialization(code)
    ha,hb=hashes["B0"],hashes["B1"]
    write(out/"initialization_pair.json",dict(B0=ha,B1=hb,equal=True))
    if src:
        # Check BOTH arms before either GPU is allowed to train.
        for arm in ("B0","B1"):
            failures=[]
            for name in ("latest.pt","latest.prev.pt"):
                try:
                    ck=torch.load(out/arm/name,map_location="cpu",weights_only=True)
                    if ck["protocol"]!=protocol or ck["arm"]!=arm: raise ValueError("Protocol mismatch")
                    if not 0<=ck["epoch"]<=100 or len(ck["history"])!=ck["epoch"]: raise ValueError("Epoch cursor")
                    if ck["best_epoch"]:
                        best=torch.load(out/arm/"best.pt",map_location="cpu",weights_only=True)
                        if best["protocol"]!=protocol or best["epoch"]!=ck["best_epoch"] or best["best"]!=ck["best"]:
                            raise ValueError("Latest/best transaction mismatch")
                    required={"model","optimizer","scaler","cpu_rng","cuda_rng","order_rng","python_rng","numpy_rng","mid"}
                    if not required<=ck.keys(): raise ValueError("Incomplete recovery state")
                    if ck["mid"]:
                        m=ck["mid"]; n=shapes["train"]["gt"][0]
                        if ck["epoch"]>=100 or not 0<=m["batch"]<=(n+31)//32: raise ValueError("Batch cursor")
                        if m["order"].dtype!=torch.int64 or not torch.equal(m["order"].sort().values,torch.arange(n)):
                            raise ValueError("Invalid permutation")
                    break
                except Exception as error: failures.append(name+": "+str(error))
            else: raise RuntimeError(arm+": cannot resume; neither model launched. "+"; ".join(failures))
    processes=[]; logs=[]
    try:
        for gpu,arm in enumerate(("B0","B1")):
            config=dict(root=str(out),arm=arm,physical_gpu=gpu,resume=REQUIRE_RESUME,protocol=protocol,
                        paths=paths,shapes=shapes,selection=selection,core_sha256=ha)
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
            completed=sum(s.get("phase")=="complete" for s in current.values())
            waiting=sum(not current.get(a) for a in ("B0","B1"))
            lines=["=== CtrS Kaggle GPU training + RR/FR evaluation ===",
                   f"Completed {completed}/2 | waiting {waiting} | GPUs 2"]
            try:
                gpu_info=subprocess.run(["nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader"],capture_output=True,text=True,check=True,timeout=3)
                lines.extend(gpu_info.stdout.strip().splitlines())
            except (OSError,subprocess.SubprocessError):
                lines.append("GPU utilization / device memory: unavailable")
            for gpu,arm in enumerate(("B0","B1")):
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
                if phase=="cache":
                    arrays=[(split,key) for split in ("train","val") for key in ("pan","lms","ms","gt")]
                    sizes=[int(np.prod(shapes[split][key])) for split,key in arrays]
                    at=(s.get("split"),s.get("array"))
                    if at in arrays:
                        progress=100*sum(sizes[:arrays.index(at)+1])/sum(sizes)
                        details.append(f"H5 -> GPU {at[0]}/{at[1]}: {progress:.1f}%")
                if phase=="evaluate": details.append(f"{s.get('split','')} scene {s.get('scene','-')}/{s.get('scenes','-')}")
                if details: lines.append("  "+" | ".join(details))
            lines.append("A total ETA is not asserted before each variant and evaluation speed are measured.")
            live.update(HTML("<pre style='white-space:pre-wrap'>"+escape("\n".join(lines))+"</pre>"))
        refresh_progress()
        while any(p.poll() is None for p in processes):
            changed=False
            for arm in ("B0","B1"):
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
        for arm in ("B0","B1"):
            status=out/arm/"status.json"
            if status.exists(): current.setdefault(arm,{}).update(json.loads(status.read_text()))
        refresh_progress()
        if any(p.returncode!=0 for p in processes): raise RuntimeError(f"Worker stopped/failed: {[p.returncode for p in processes]}; preserve Output and inspect worker.log")
    except BaseException:
        for p in processes:
            if p.poll() is None: p.terminate()
        # Cooperative termination saves the next completed batch; parent does not kill workers.
        raise
    finally:
        for log in logs: log.close()
    results={arm:json.loads((out/arm/"results.json").read_text()) for arm in ("B0","B1")}
    histories={arm:json.loads((out/arm/"history.json").read_text()) for arm in results}
    if any(len(h)!=100 for h in histories.values()): raise RuntimeError("Incomplete history")
    if [r["sample_order_sha256"] for r in histories["B0"]]!=[r["sample_order_sha256"] for r in histories["B1"]]: raise RuntimeError("Data order differs")
    delta={s:{k:results["B1"]["metrics"][s]["mean"][k]-v for k,v in results["B0"]["metrics"][s]["mean"].items()} for s in ("RR","FR")}
    write(out/"paired_comparison.json",dict(delta_B1_minus_B0=delta,common_core=True,sample_order=True))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key in zip(axes.flat,("train_mse","val_mse","val_psnr","val_sam")):
        for arm,h in histories.items(): ax.plot([r["epoch"] for r in h],[r[key] for r in h],label=arm)
        ax.set_title(key); ax.set_xlabel("Epoch"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out/"learning.png",dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,(s,k) in zip(axes.flat,[("RR","PSNR"),("RR","SAM"),("RR","MSE_peak1"),("RR","ERGAS"),("RR","Q2n"),("FR","QNR")]):
        ax.bar(["B0","B1"],[results[a]["metrics"][s]["mean"][k] for a in results]); ax.set_title(s+" "+k+(" (provisional)" if s=="FR" else ""))
    fig.tight_layout(); fig.savefig(out/"test_comparison.png",dpi=150); plt.close(fig)
    write(out/"results.json",results)
    lines=["# QB B0/B1 K4 controlled experiment", "",
        "| Reference | Change | Observed difference | Decision |",
        "|---|---|---|---|",
        f"| Fresh B0 | B1 shared three-stage restoration | RR PSNR {delta['RR']['PSNR']:+.6f} dB; provisional FR QNR {delta['FR']['QNR']:+.6f} | Review seed42 before deciding seed43 |",
        "", "## 1. Purpose and status", "",
        "Both models completed 100 epochs and full RR20/FR20 evaluation. Remote archival has not been verified.",
        "", "## 2. Changes and conditions", "",
        "QB; K4; seed42; batch=micro32; fused Adam; fixed lr1e-4; final output MSE; AMP FP16; channels-last; strict deterministic; TF32 disabled.",
        "B0 is the original single-output adapter. B1 repeats the same core/error encoder/scalar alpha three times with zero error, trainable encoder bias and unconstrained alpha initialized to 0.1. No B2.",
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
        "Historical K6 B1 minus separately trained B0: PSNR +0.236541 dB, SAM -0.133224 deg, MSE -1.36760636e-5, provisional QNR -0.037049. Those runs differed in optimizer/session/order/resume and are not controlled K4 results.",
        "Current B1-minus-B0 values are in paired_comparison.json. They are a single-seed observation.",
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
    artifacts={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob("*") if p.is_file() and p.suffix not in (".tmp",".pyc") and "__pycache__" not in p.parts and p.name!="artifact_manifest.json"}
    write(out/"artifact_manifest.json",dict(status="training + full RR/FR completed",artifacts=artifacts,
        limitations="FR has no GT. QNR/D_s bicubic MATLAB parity and Q2n/SCC parity remain unverified; seed43 requires a separate human decision."))
    archive=out.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for rel in [*artifacts,"artifact_manifest.json"]: z.write(out/rel,arcname=out.name+"/"+rel)
    with zipfile.ZipFile(archive) as z:
        if z.testzip(): raise RuntimeError("Archive CRC failed")
    write(archive.with_suffix(".sha256.json"),dict(file=archive.name,sha256=sha(archive)))
    print(f"[COMPLETE] {archive}; verify/download Saved Output. Seed43 was NOT launched.",flush=True)

if __name__ == "__main__": main()
