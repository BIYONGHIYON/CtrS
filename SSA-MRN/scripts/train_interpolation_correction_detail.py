"""A6000 single-change model ablations; same controlled MSE training protocol."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse, json, math, time, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ssamrn.data.pancollection import PanCollectionH5
from ssamrn.models.pan_variants import make_model
from ssamrn.observation import load_profile
from train import cpu_state, evaluate

def atomic_save(value, path):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp); os.replace(tmp, path)

class IndexedPanCollectionH5(PanCollectionH5):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["index"] = index
        return sample


def auxiliary(pred, gt, ms, kind, operator=None, margin=0, phases=None):
    if kind == "spectral":
        return (1-F.cosine_similarity(pred, gt, dim=1, eps=1e-8)).mean()
    if kind == "edge":
        return ((pred[...,1:,:]-pred[...,:-1,:])-(gt[...,1:,:]-gt[...,:-1,:])).square().mean() + ((pred[...,1:]-pred[...,:-1])-(gt[...,1:]-gt[...,:-1])).square().mean()
    if kind == "consistency":
        if operator is None or phases is None:
            raise ValueError("Consistency requires an audited observation operator and fixed phases")
        return operator.loss(pred, ms, margin, phases)
    return pred.new_zeros(())

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--resume",choices=["latest","best"])
    p.add_argument("--extend-epochs",action="store_true",help="Allow only a larger epoch target when resuming a screened candidate")
    p.add_argument("--stop-after-epoch",type=int,help="Verification only; stop after saving the named epoch")
    a=p.parse_args(); c=json.loads(a.config.read_text()); a.output.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    torch.manual_seed(c["seed"]); torch.cuda.manual_seed_all(c["seed"])
    gen=torch.Generator().manual_seed(c["seed"])
    tr=IndexedPanCollectionH5(Path(c["train"]),c["sensor"],limit=c.get("limit")); va=PanCollectionH5(Path(c["val"]),c["sensor"],limit=c.get("limit"))
    if Path(c["train"]).resolve()==Path(c["val"]).resolve(): raise ValueError("train/val overlap")
    if tr.channels != va.channels: raise ValueError("channels mismatch")
    loader=DataLoader(tr,batch_size=c["batch_size"],shuffle=True,generator=gen,num_workers=0)
    validation=DataLoader(va,batch_size=c["micro_batch"],shuffle=False,num_workers=0)
    operator, profile, phase_table = None, None, None
    if c["loss"] == "consistency":
        observation = c.get("observation")
        if not observation or not observation.get("sha256"):
            raise ValueError("Audited observation profile and hash required")
        path = Path(observation["profile"])
        if not path.is_absolute(): path = Path(__file__).resolve().parents[1] / path
        operator, profile = load_profile(path, observation["sha256"], c["sensor"], c["train"])
        operator = operator.float().cuda()
        phase_table = torch.tensor(profile["phases"], dtype=torch.long)
    model=make_model(c).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=c["lr"]); start=0; best=float("inf")
    if a.resume:
        ck=torch.load(a.output/(a.resume+".pt"),map_location="cpu",weights_only=True)
        if ck["config"]!=c:
            old=ck["config"]
            if not (a.extend_epochs and c["epochs"]>old["epochs"] and {k:v for k,v in old.items() if k!="epochs"}=={k:v for k,v in c.items() if k!="epochs"}):
                raise ValueError("resume config mismatch")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        torch.set_rng_state(ck["rng"]); torch.cuda.set_rng_state_all(ck["cuda_rng"]); gen.set_state(ck["loader_rng"])
        start=ck["epoch"]; best=ck["best_mse"]
    for epoch in range(start,c["epochs"]):
        model.train(); total=0.; count=0; tick=time.time()
        for batch_index,b in enumerate(loader):
            opt.zero_grad(set_to_none=True); n=len(b["gt"])
            for offset in range(0,n,c["micro_batch"]):
                pan,lms,ms,gt=(b[key][offset:offset+c["micro_batch"]].cuda() for key in ["pan","lms","ms","gt"])
                pred=model(pan,lms,ms); mse=F.mse_loss(pred,gt)
                phases = phase_table[b["index"][offset:offset+c["micro_batch"]]] if phase_table is not None else None
                loss=mse+c["weight"]*auxiliary(pred,gt,ms,c["loss"],operator,profile["lr_margin"] if profile else 0,phases)
                if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
                (loss*len(gt)/n).backward(); total+=mse.item()*len(gt); count+=len(gt)
            opt.step()
            if batch_index % 50 == 0:
                print(json.dumps(dict(epoch=epoch+1,batch=batch_index+1,batches=len(loader),train_mse=total/count)),flush=True)
        val=evaluate(model,validation,"cuda")
        if not math.isfinite(val): raise RuntimeError("nonfinite validation")
        improved=val<best; best=min(best,val)
        row=dict(epoch=epoch+1,train_mse=total/count,val_mse=val,seconds=time.time()-tick)
        print(json.dumps(row),flush=True)
        # Resume can replay an incomplete epoch; keep one record per completed epoch.
        history=a.output/"history.jsonl"
        rows=[json.loads(x) for x in history.read_text().splitlines()] if history.exists() else []
        rows=[x for x in rows if x["epoch"]<epoch+1]+[row]
        tmp=history.with_suffix(".tmp"); tmp.write_text("".join(json.dumps(x)+"\n" for x in rows)); os.replace(tmp,history)
        ck=dict(config=c,epoch=epoch+1,best_mse=best,model=cpu_state(model.state_dict()),optimizer=cpu_state(opt.state_dict()),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=gen.get_state())
        atomic_save(ck,a.output/"latest.pt")
        if improved: atomic_save(ck,a.output/"best.pt")
        if a.stop_after_epoch and epoch+1>=a.stop_after_epoch: return
    (a.output/"complete.json").write_text(json.dumps(dict(config=c,best_mse=best,epoch=c["epochs"]),indent=2))
if __name__=="__main__": main()
