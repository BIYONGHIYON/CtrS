"""Audit LIB alignment and write fixed per-scene integer RGB translations."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'SSA-MRN/src'))
from ssamrn.data.lib_hsi import cube_view
from ssamrn.data.registration import estimate, warp_rgb

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int)
    args=p.parse_args(); records={}; args.output.parent.mkdir(parents=True,exist_ok=True)
    for split in ('train','validation','test'):
        files=sorted(f for f in (args.data_root/split/'rgb').glob('*.png') if not f.name.startswith('._'))
        if args.limit: files=files[:args.limit]
        for rgb_path in tqdm(files,desc=split):
            rgb=np.array(Image.open(rgb_path).convert('RGB'))
            cube=cube_view(args.data_root/split/'reflectance_cubes'/(rgb_path.stem+'.hdr'))
            # Visible display bands used throughout this experiment; no model predictions.
            hsi=np.asarray(cube[:,:,[69,52,18]],dtype=np.float32)
            record=estimate(hsi,rgb); records[split+'/'+rgb_path.stem]=record
            if len(records)==1:
                aligned,_=warp_rgb(rgb,record['dy'],record['dx'])
                lo,hi=np.percentile(hsi,[1,99],axis=(0,1)); display=(np.clip((hsi-lo)/(hi-lo+1e-8),0,1)*255).astype('uint8')
                Image.fromarray(np.concatenate([display,rgb,aligned],axis=1)).save(args.output.with_suffix('.png'))
    result={'version':1,'method':'integer RGB translation; radius12; >=3 quadrant consensus; edge NCC>=.25; gain>=.01',
            'partial':args.limit is not None,'scenes':records,
            'summary':{'count':len(records),'accepted':sum(x['accepted'] for x in records.values()),
                       'median_before':float(np.median([x['before'] for x in records.values()])),
                       'median_applied_after':float(np.median([x['after_candidate'] if x['accepted'] else x['before'] for x in records.values()]))}}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(result['summary'])
if __name__=='__main__': main()
