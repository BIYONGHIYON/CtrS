"""Re-evaluate retained milestone checkpoints against original QB H5 data."""
import argparse,importlib,json,os,shutil,sys
from pathlib import Path

def main():
    here=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target',type=int,choices=(50,100,150),default=150)
    p.add_argument('--variants',nargs='+',choices=('R0','W2','W3','F01','F11'),default=['R0','W2','W3','F01','F11'])
    p.add_argument('--checkpoint-root',type=Path,default=here/'checkpoints');p.add_argument('--history-root',type=Path,default=here/'results/QB')
    p.add_argument('--source-root',type=Path,default=here/'provenance/source/SSA-MRN');p.add_argument('--config',type=Path,default=here/'config.json')
    a=p.parse_args()
    if 'R0' not in a.variants:p.error('Include R0 for the comparison graph')
    sys.path.insert(0,str(a.source_root.resolve()))
    runner=importlib.import_module('experiments.performance.1_capacity_refinement.run')
    work=a.output.resolve()/'runtime';work.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.config.read_text());plan['variants']=a.variants;manifest={}
    for proto,name in [('RR','test_qb_multiExm1.h5'),('FR','test_qb_OrigScale_multiExm1.h5')]:
        paths=[x for x in a.data_root.rglob('*') if x.is_file() and (x.name==name or x.name.endswith('__'+name)) and not x.name.startswith('._') and '__._' not in x.name]
        if len(paths)!=1:raise ValueError(f'{proto}: expected one original H5, found {paths}')
        manifest[proto]={'path':str(paths[0].resolve()),'sha256':runner.sha(paths[0])}
    runner.write(work/'plan.json',plan);runner.write(work/'data_manifest.json',manifest)
    for v in a.variants:
        dest=work/v;dest.mkdir(exist_ok=True);name=f'best_at_{a.target}.pt'
        source=a.checkpoint_root/(v+'_QB_s42')/name
        if not source.exists():source=a.checkpoint_root/v/name
        target=dest/name
        if target.exists():
            if runner.sha(source)!=runner.sha(target):raise ValueError('Use a new output directory for different weights')
        else:
            try:os.link(source,target)
            except OSError:shutil.copy2(source,target)
        shutil.copy2(a.history_root/v/'history.json',dest/'history.json')
    runner.seed();runner.evaluate(argparse.Namespace(runtime=work,target=a.target))
    print(work/'evaluation'/str(a.target)/'metrics.json')
if __name__=='__main__':main()
