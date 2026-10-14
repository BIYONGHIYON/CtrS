"""Verify retained files and restore four models; does not re-evaluate RR/FR."""
import argparse,hashlib,json,shutil,sys,tempfile,zipfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output',type=Path)
    a=parser.parse_args();root=a.root.resolve();e=root/'experiments/ssai/5_mixed_gating'
    manifest=json.loads((e/'manifest.json').read_text())
    for name,record in manifest['files'].items():
        path=e/name
        if not path.is_file()or path.stat().st_size!=record['bytes']or sha(path)!=record['sha256']:
            raise ValueError('Manifest mismatch: '+name)
    refs=json.loads((e/'provenance/reference_assets.json').read_text())['files']
    reference_count=0
    for records in refs.values():
        for record in records.values():
            path=root/record['repository_path']
            if sha(path)!=record['sha256']:raise ValueError('Reference mismatch: '+str(path))
            reference_count+=1
    plan=json.loads((e/'provenance/plan.json').read_text())
    with tempfile.TemporaryDirectory(prefix='ssai5-restore-')as folder:
        temporary=Path(folder)
        with zipfile.ZipFile(e/'provenance/training_source.zip')as source:
            for name,h in plan['source_hashes'].items():
                value=source.read('source/'+name)
                if hashlib.sha256(value).hexdigest()!=h:raise ValueError('Training source mismatch: '+name)
            network=temporary/'references/upstream/network.py';network.parent.mkdir(parents=True)
            network.write_bytes(source.read('source/references/upstream/network.py'))
        # Use the committed loader and candidate implementation, supplying the archived official network.
        shutil.copytree(root/'common',temporary/'common',ignore=shutil.ignore_patterns('__pycache__'))
        dst=temporary/'experiments/ssai/5_mixed_gating';dst.mkdir(parents=True)
        shutil.copy2(e/'model.py',dst/'model.py')
        if sha(e/'model.py')!=plan['source_hashes']['experiments/ssai/5_mixed_gating/model.py']:
            raise ValueError('Committed model differs from actual training source')
        sys.path.insert(0,str(temporary))
        import torch
        from common import checkpoint
        torch.set_num_threads(1);checkpoint.RESEARCH_TRACK='ssai';weights={}
        for arm in ('M25','M50'):
            history=json.loads((e/f'results/training/{arm}/history.json').read_text())
            if [x['epoch']for x in history]!=list(range(1,101)):raise ValueError('Incomplete history')
            minimum=min(x['val_mse']for x in history)
            best=next(x['epoch']for x in history if x['val_mse']==minimum)
            for name in ('best','latest'):
                path=e/f'checkpoints/{arm}_QB_s42/{name}.pt';net,ck=checkpoint.load_model(5,path)
                if ck['epoch']!=(best if name=='best'else 100)or ck['config']['plan']!=plan or ck['config']['variant']!=arm:
                    raise ValueError('Checkpoint epoch/config mismatch')
                if name=='best'and ck['validation_mse']!=minimum:raise ValueError('Validation selection mismatch')
                if not ck['optimizer']['state']or not all(k in ck for k in ('torch_rng','cuda_rng','loader_rng','python_rng','numpy_rng')):
                    raise ValueError('Missing recovery state')
                if not all(torch.isfinite(v).all()for v in ck['model'].values()):raise ValueError('Nonfinite weights')
                with torch.inference_mode():
                    torch.manual_seed(42)
                    output=net(torch.rand(1,1,64,64),torch.rand(1,4,64,64),torch.rand(1,4,16,16))
                if output.shape!=(1,4,64,64)or not torch.isfinite(output).all():raise ValueError('Invalid inference')
                weights[f'{arm}/{name}']={'epoch':ck['epoch'],'sha256':sha(path),'strict_load':True,'finite_64x64_cpu_inference':True,'optimizer_rng_preserved':True}
    result={'manifest_files':len(manifest['files']),'reference_files':reference_count,'weights':weights,'verified':True,'scope':'file integrity and checkpoint recovery; not a new RR/FR performance evaluation'}
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
