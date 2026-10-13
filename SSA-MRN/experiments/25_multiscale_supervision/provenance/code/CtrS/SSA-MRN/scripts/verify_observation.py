"""Audit fixed MTF geometry against TRAIN pairs; no model optimization."""
import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ssamrn.observation import mtf_kernels, GNYQ, SOURCE_COMMIT, module_hash
from ssamrn.data.pancollection import SENSOR_MAX

# Demo_DataSimu_qb.m: original, vertical flip, horizontal flip (not both).
PHASES = [(2,2),(1,2),(2,1)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--sensor",choices=list(GNYQ),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples",type=int,default=32)
    p.add_argument("--all-samples",action="store_true",help="Required for a usable training phase manifest")
    a=p.parse_args()
    if a.samples<8:p.error("At least 8 probes required")
    ratio,margin=4,5
    kernels=mtf_kernels(a.sensor)[:,0][None]
    phases=[];mses=[];probes=[]
    initial_stat=a.train.stat()
    with h5py.File(a.train,"r") as f:
        n=len(f["gt"])
        if min(f["ms"].shape[-2:])<=2*margin:raise ValueError("Patch interior too small")
        if f["gt"].shape[1]!=kernels.shape[1]:raise ValueError("Sensor bands mismatch")
        if f["gt"].shape[-2:]!=(f["ms"].shape[-2]*4,f["ms"].shape[-1]*4):raise ValueError("Scale mismatch")
        probe_ids=set(np.linspace(0,n-1,min(a.samples,n),dtype=int).tolist())
        ids=list(range(n)) if a.all_samples else sorted(probe_ids)
        for start in range(0,len(ids),64):
            batch_ids=ids[start:start+64]
            gt=np.asarray(f["gt"][batch_ids],dtype=np.float64)
            ms=np.asarray(f["ms"][batch_ids],dtype=np.float64)
            if not np.isfinite(gt).all() or not np.isfinite(ms).all():raise ValueError("Nonfinite data")
            # Whole-scene filtering precedes patch cropping. Exclude the unknown halo.
            blurred=fftconvolve(np.pad(gt,((0,0),(0,0),(20,20),(20,20)),mode="edge"),kernels,mode="valid",axes=(-2,-1))
            target=ms[...,margin:-margin,margin:-margin]
            errors=np.stack([np.mean((blurred[...,y::4,x::4][...,margin:-margin,margin:-margin]-target)**2,axis=(1,2,3)) for y,x in PHASES],axis=1)
            selected=np.argmin(errors,axis=1)
            for row,scene in enumerate(batch_ids):
                phase=list(PHASES[int(selected[row])]);mse=float(errors[row,selected[row]])
                phases.append(phase);mses.append(mse)
                if scene in probe_ids:
                    probes.append(dict(id=scene,phase=phase,rmse_dn=mse**.5,sha256=hashlib.sha256(gt[row].tobytes()+ms[row].tobytes()).hexdigest()))
            if start%1024==0:print(f"observation audit {start+len(batch_ids)}/{len(ids)}",flush=True)
    stat=a.train.stat()
    if (stat.st_size,stat.st_mtime_ns)!=(initial_stat.st_size,initial_stat.st_mtime_ns):raise ValueError("Data changed during audit")
    rmse=float(np.mean(mses)**.5);worst=max(mses)**.5
    # Prespecified raw-DN numerical tolerance; never relaxed based on results.
    passed=rmse<=1.0 and worst<=2.0
    usable=passed and a.all_samples
    report=dict(status="validated" if usable else ("probe_passed" if passed else "blocked"),sensor=a.sensor,train_path=str(a.train.resolve()),train_stat=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns),operator_sha256=module_hash(),source_commit=SOURCE_COMMIT,kernel="DLPan_MATLAB_radial_fwind1",gnyq=GNYQ[a.sensor],ratio=ratio,kernel_size=41,padding="replicate",sampling="decimate",phase_policy="fixed_per_training_sample",phase_candidates=[list(p) for p in PHASES],phases=phases if a.all_samples else [],sample_count=n,coverage=len(ids),lr_margin=margin,probes=probes,tolerance=dict(rmse_dn=1.,max_sample_rmse_dn=2.),rmse_dn=rmse,max_sample_rmse_dn=worst,normalized_mse=rmse**2/SENSOR_MAX[a.sensor]**2,phase_counts=dict(Counter(str(p) for p in phases)),interpretation="Fixed source-based MTF; geometry assigned only from original TRAIN GT/MS, never predictions or test data. Patch-interior agreement does not establish FR sensor physics.")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    compact=dict(report);compact["phases"]="__PHASES__"
    serialized=json.dumps(compact,indent=2,sort_keys=True).replace('"__PHASES__"',json.dumps(report["phases"],separators=(",",":")))
    a.output.write_text(serialized+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:report[k] for k in ["status","sensor","coverage","sample_count","rmse_dn","max_sample_rmse_dn","phase_counts"]},indent=2))
    return 0 if usable else 2
if __name__=="__main__":sys.exit(main())
