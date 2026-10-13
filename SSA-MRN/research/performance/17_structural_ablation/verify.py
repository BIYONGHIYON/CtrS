"""Checks exact S0 parity, common initialization, gradients, and optimizer/RNG recovery."""
import json, sys, tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from common.model import RestoredPansharpeningNet
StructureNet = __import__('importlib').import_module('research.performance.17_structural_ablation.model').StructureNet; VARIANTS = __import__('importlib').import_module('research.performance.17_structural_ablation.model').VARIANTS
checkpoint = __import__('importlib').import_module('research.performance.17_structural_ablation.run').checkpoint; seed = __import__('importlib').import_module('research.performance.17_structural_ablation.run').seed

def main():
    seed();torch.backends.cudnn.benchmark=False
    pan=torch.rand(2,1,64,64,device='cuda');lms=torch.rand(2,4,64,64,device='cuda');ms=torch.rand(2,4,16,16,device='cuda')
    torch.manual_seed(42);ref=RestoredPansharpeningNet(4).cuda().eval()
    torch.manual_seed(42);base=StructureNet('S0').cuda().eval()
    with torch.no_grad():torch.testing.assert_close(base(pan,lms,ms),ref(pan,lms,ms),rtol=0,atol=0)
    rows=[]
    for v in VARIANTS:
        torch.manual_seed(42);m=StructureNet(v).cuda();common=0
        for k,x in m.state_dict().items():
            if k in base.state_dict() and x.shape==base.state_dict()[k].shape:
                torch.testing.assert_close(x,base.state_dict()[k],rtol=0,atol=0);common+=1
        y=m(pan,lms,ms);assert y.shape==lms.shape and torch.isfinite(y).all()
        y.square().mean().backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
        rows.append(dict(variant=v,parameters=sum(p.numel() for p in m.parameters()),common_tensors=common));del m
    torch.manual_seed(42);m=StructureNet('R2').cuda();opt=torch.optim.Adam(m.parameters(),lr=1e-4);sc=torch.amp.GradScaler('cuda')
    def step(model,optimizer,scaler):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=model(pan,lms,ms).square().mean()
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
    step(m,opt,sc)
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/'latest.pt';checkpoint(f,dict(model=m.state_dict(),optimizer=opt.state_dict(),scaler=sc.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        step(m,opt,sc);expected={k:v.clone() for k,v in m.state_dict().items()}
        ck=torch.load(f,map_location='cpu',weights_only=False);rest=StructureNet('R2').cuda();rest.load_state_dict(ck['model']);o=torch.optim.Adam(rest.parameters(),lr=1e-4);o.load_state_dict(ck['optimizer']);s=torch.amp.GradScaler('cuda');s.load_state_dict(ck['scaler']);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);step(rest,o,s)
        for k,x in rest.state_dict().items():torch.testing.assert_close(x,expected[k],rtol=0,atol=0)
    result=dict(s0_exact_parity=True,all_candidate_gradient_checks=True,optimizer_scaler_resume_exact=True,rows=rows)
    print(json.dumps(result),flush=True)
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
