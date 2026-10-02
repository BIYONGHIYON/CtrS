"""Offline homography audit for the grouped12/23tap experiment."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'SSA-MRN/src'))
from ssamrn.data.lib_hsi import cube_view, EXPECTED
from ssamrn.data.registration import estimate_projective

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int);p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        p.error('workers and limit must be positive')
    jobs=[(split,f) for split in ('train','validation','test') for f in
          sorted(p for p in (args.data_root/split/'rgb').glob('*.png') if not p.name.startswith('._'))[:args.limit]]
    if args.limit is None:
        for split, expected in EXPECTED.items():
            if sum(s==split for s,_ in jobs) != expected:
                p.error(f'Incomplete {split} split: expected {expected} scenes')
    def audit(job):
        split,path=job
        with Image.open(path) as im:rgb=np.array(im.convert('RGB'))
        cube=cube_view(args.data_root/split/'reflectance_cubes'/(path.stem+'.hdr'))
        hsi=np.asarray(cube[:,:,[69,52,18]],dtype='float32')
        return split+'/'+path.stem,estimate_projective(hsi,rgb)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records=dict(tqdm(pool.map(audit,jobs),total=len(jobs),desc='projective audit',mininterval=2.))
    result=dict(version=2,partial=args.limit is not None,method='ECC homography, moving RGB to fixed HSI, bounded 32px corners, regional checks; translation fallback',
                scenes=records,summary={key:sum(v['transform']==key for v in records.values()) for key in ('homography','translation','identity')})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,indent=2),encoding='utf-8')
    temporary.replace(args.output);print(result['summary'])
if __name__=='__main__':main()
