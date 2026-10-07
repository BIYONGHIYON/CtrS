"""Verify supervised loss wiring with CUDA forward/backward only, no optimizer."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ssamrn.observation import load_profile
from ssamrn.data.pancollection import PanCollectionH5
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
from train_controlled import auxiliary


def main():
    torch.use_deterministic_algorithms(True)
    plan=json.loads((ROOT/"scripts/controlled_suite_plan.json").read_text())
    config=plan["observations"]["QB"]
    operator,profile=load_profile(ROOT/config["profile"],config["sha256"],"QB",plan["data"]["QB"]["train"])
    operator=operator.float().cuda();data=PanCollectionH5(Path(plan["data"]["QB"]["train"]),"QB")
    ids=[next(i for i,p in enumerate(profile["phases"]) if p==phase) for phase in [[2,2],[1,2],[2,1]]]
    samples=[data[i] for i in ids]
    pan,lms,ms,gt=(torch.stack([s[key] for s in samples]).cuda() for key in ["pan","lms","ms","gt"])
    phases=[profile["phases"][i] for i in ids]
    for k in [4,6]:
        torch.manual_seed(42)
        model=RestoredPansharpeningNet(channels=4,ssai_dimension=k).cuda().eval()
        prediction=model(pan,lms,ms)
        loss=F.mse_loss(prediction,gt)+0.01*auxiliary(prediction,gt,ms,"consistency",operator,profile["lr_margin"],phases)
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError("Model gradient failure")
        print(json.dumps(dict(k=k,status="passed",sample_ids=ids,phases=phases,finite_gradients=True,optimizer_steps=0)),flush=True)
    data.close()
if __name__=="__main__":main()
