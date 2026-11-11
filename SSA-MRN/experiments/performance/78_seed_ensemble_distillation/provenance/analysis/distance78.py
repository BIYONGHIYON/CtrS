"""78 local diagnostic (Apple MPS FP32, after the server evaluation; reference, not a judgement): on the 20 QB RR test scenes, how close is each model to the teacher and to GT (RMSE in DN).
Usage: python provenance/analysis/distance78.py results/teacher_distance_test.json"""
import sys, json, numpy as np, h5py, torch
from pathlib import Path as _P
E=str(_P(__file__).resolve().parents[2]); sys.path.insert(0,E); sys.path.append(str(_P(E).parents[2]))
import model as M, prepare as P
from pathlib import Path
torch.set_grad_enabled(False); dev='mps'; R=Path(E).parents[2]
cfg=json.load(open(E+'/config.json'))
def ld(path,kind,arm=None):
    ck=torch.load(R/path,map_location='cpu',weights_only=False)
    m=M.M54.Model('C2MS',4,ck['seed']) if kind=='C2MS' else M.Model(arm,4,42); m.load_state_dict(ck['model'],strict=True); return m.eval().to(dev)
nets={'C2MS':ld(cfg['controls']['QB'].replace('<seed>','42'),'C2MS')}
for a in M.ARMS:
    nets['SAME_'+a]=ld(cfg['same_structure_controls'][a]['path'],'S',a)
    nets['KD_'+a]=ld(f'experiments/performance/78_seed_ensemble_distillation/checkpoints/{a}_QB_s42/best.pt','S',a)
T=M.Teacher(P.teacher_paths(cfg)).to(dev).eval()
acc={k:np.zeros(2) for k in list(nets)+['TEACHER']}
with h5py.File(R/'data/test/test_qb_multiExm1.h5') as f:
    for i in range(20):
        x=[torch.from_numpy(np.asarray(f[k][i],dtype=np.float32)/2047.)[None].to(dev) for k in ('pan','lms','ms')]
        g=np.asarray(f['gt'][i],dtype=np.float64); t=T(*x)[0].cpu().double().numpy()*2047
        acc['TEACHER'][1]+=((t-g)**2).mean()
        for k,n in nets.items():
            y=n(*x)[0].float().cpu().double().numpy()*2047; acc[k]+=[((y-t)**2).mean(),((y-g)**2).mean()]
out={k:dict(rmse_to_teacher_DN=float(np.sqrt(v[0]/20)),rmse_to_gt_DN=float(np.sqrt(v[1]/20))) for k,v in acc.items()}
print(json.dumps(out,indent=1)); json.dump(out,open(sys.argv[1],'w'),indent=1)
