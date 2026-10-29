"""Frozen clean/blur source validation screening; never reads target test data."""
import json,sys
from pathlib import Path
import torch
from model import Model
from prepare import SENSORS
from train import T,write,now,run_name,blur_pan
HERE=Path(__file__).resolve().parent

def screen(cfg):
    T.setup(42);root=Path(cfg['runtime']);result={};original_batch=T.batch
    for sensor in cfg['sensors']:
        paths={a:root/'runs'/run_name(sensor,a,42)/'checkpoints/best.pt' for a in cfg['arms']}
        if 'C2MS_R' not in paths:paths['C2MS_R']=HERE.parents[2]/cfg['reference_controls'][sensor]['C2MS_R_70']
        rows={};cache=T.cache(cfg,sensor,'val')
        for arm,path in paths.items():
            ck=torch.load(path,map_location='cpu',weights_only=False)
            m=Model(arm if arm in cfg['arms'] else 'C2MS',SENSORS[sensor]['bands'],42).cuda()
            m.load_state_dict(ck['model'],strict=True);rows[arm]={}
            for sigma in (0.,.4,.8,1.2):
                def batch(c,idx,layout):
                    b=original_batch(c,idx,layout)
                    if sigma:b[0]=blur_pan(b[0],torch.full((len(b[0]),),sigma),4).contiguous(memory_format=torch.channels_last)
                    return b
                T.batch=batch
                try:rows[arm][str(sigma)]=T.validation(m,cache,cfg)
                finally:T.batch=original_batch
            del m
        ref=rows['C2MS_R'];gates={}
        for arm,r in rows.items():
            dp=r['0.0']['val_psnr']-ref['0.0']['val_psnr'];ds=r['0.0']['val_sam']-ref['0.0']['val_sam']
            bp=sum(r[str(s)]['val_psnr']-ref[str(s)]['val_psnr'] for s in (.4,.8,1.2))/3
            bs=sum(r[str(s)]['val_sam']-ref[str(s)]['val_sam'] for s in (.4,.8,1.2))/3
            passed=(dp>0 and ds<0) if cfg['experiment']==70 else (dp>=-.03 and ds<=.02 and bp>0 and bs<0)
            gates[arm]=dict(clean_delta_psnr=dp,clean_delta_sam=ds,blur_delta_psnr=bp,blur_delta_sam=bs,passed=passed)
        result[sensor]=dict(metrics=rows,gates=gates)
    write(root/'validation_screen.json',dict(time_kst=now(),results=result,selection='checkpoint by clean MSE; no test input; report all preregistered arms; no automatic repeats'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();screen(json.loads(Path(a.config).read_text()))
