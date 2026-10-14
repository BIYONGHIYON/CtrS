"""Verify retained SSAI6/7 files and restore weights; no RR/FR re-evaluation."""
import argparse,hashlib,json,shutil,sys,tempfile,zipfile
from pathlib import Path

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    a=parser.parse_args();root=a.root.resolve();base=root/'experiments/ssai'
    folders={6:'6_ms_conditioned_detail',7:'7_channel_relation_detail'}
    counts={};shared_code=set()
    for number,folder in folders.items():
        experiment=base/folder;manifest=json.loads((experiment/'manifest.json').read_text())
        for name,record in manifest['files'].items():
            p=experiment/name
            if not p.is_file()or p.stat().st_size!=record['bytes']or sha(p)!=record['sha256']:
                raise ValueError('Manifest mismatch: '+str(p))
        counts[str(number)]=len(manifest['files'])
        for name,record in manifest.get('shared_code',{}).items():
            path=root/name
            if path.stat().st_size!=record['bytes']or sha(path)!=record['sha256']:
                raise ValueError('Shared code mismatch: '+name)
            shared_code.add(name)
    refs=json.loads((base/folders[6]/'provenance/reference_assets.json').read_text())['files']
    reference_count=0
    for files in refs.values():
        for record in files.values():
            p=root/record['repository_path']
            if sha(p)!=record['sha256']:raise ValueError('Reference mismatch: '+str(p))
            reference_count+=1
    shared=json.loads((base/folders[7]/'provenance/shared_evidence.json').read_text())
    for record in shared.values():
        if isinstance(record,dict)and 'path'in record:
            if sha(base/folders[7]/record['path'])!=record['sha256']:raise ValueError('Shared evidence mismatch')
    plan=json.loads((base/folders[6]/'provenance/plan.json').read_text())
    if plan!=json.loads((base/folders[7]/'provenance/plan.json').read_text()):raise ValueError('Suite plans differ')
    with tempfile.TemporaryDirectory(prefix='ssai-detail-restore-')as directory:
        temporary=Path(directory)
        with zipfile.ZipFile(base/folders[6]/'provenance/training_source.zip')as archive:
            for name,h in plan['source_hashes'].items():
                value=archive.read('source/'+name)
                if hashlib.sha256(value).hexdigest()!=h:raise ValueError('Training source hash mismatch: '+name)
            p=temporary/'references/upstream/network.py';p.parent.mkdir(parents=True)
            p.write_bytes(archive.read('source/references/upstream/network.py'))
        shutil.copytree(root/'common',temporary/'common',ignore=shutil.ignore_patterns('__pycache__'))
        for number,folder in folders.items():
            dst=temporary/'experiments/ssai'/folder;dst.mkdir(parents=True)
            src=base/folder/'model.py';name=f'experiments/ssai/{folder}/model.py'
            if sha(src)!=plan['source_hashes'][name]:raise ValueError('Committed candidate differs from training')
            shutil.copy2(src,dst/'model.py')
        for name in('common/model.py','common/ssai_detail.py','common/ssai_suite.py'):
            if sha(root/name)!=plan['source_hashes'][name]:raise ValueError('Committed model/suite differs from training: '+name)
        sys.path.insert(0,str(temporary))
        import torch
        from common import checkpoint
        torch.set_num_threads(1);checkpoint.RESEARCH_TRACK='ssai';records={}
        for number,folder in folders.items():
            arm={6:'D6',7:'D7'}[number];e=base/folder
            history=json.loads((e/'results/training/history.json').read_text())
            if [x['epoch']for x in history]!=list(range(1,101)):raise ValueError('History incomplete')
            minimum=min(x['val_mse']for x in history);best=next(x['epoch']for x in history if x['val_mse']==minimum)
            for name in('best','latest'):
                path=e/f'checkpoints/{arm}_QB_s42/{name}.pt';net,ck=checkpoint.load_model(number,path)
                if ck['epoch']!=(best if name=='best'else 100)or ck['config']['variant']!=arm or ck['config']['plan']!=plan:
                    raise ValueError('Checkpoint conditions differ')
                if name=='best'and ck['validation_mse']!=minimum:raise ValueError('Validation selection mismatch')
                if not ck['optimizer']['state']or not all(k in ck for k in('torch_rng','cuda_rng','loader_rng','python_rng','numpy_rng')):
                    raise ValueError('Optimizer/RNG missing')
                if not all(torch.isfinite(v).all()for v in ck['model'].values()):raise ValueError('Nonfinite weights')
                detail_sum=sum(v.abs().sum().item()for k,v in ck['model'].items()if k.endswith('detail_out.weight'))
                if detail_sum<=0:raise ValueError('Detail correction not trained')
                with torch.inference_mode():
                    torch.manual_seed(42)
                    prediction=net(torch.rand(1,1,64,64),torch.rand(1,4,64,64),torch.rand(1,4,16,16))
                if prediction.shape!=(1,4,64,64)or not torch.isfinite(prediction).all():raise ValueError('Invalid recovered prediction')
                records[f'{arm}/{name}']={'epoch':ck['epoch'],'sha256':sha(path),'strict_load':True,'finite64x64_cpu_inference':True,'optimizer_rng_preserved':True,'detail_weight_abs_sum':detail_sum}
    result={'manifest_files':counts,'total_manifest_files':sum(counts.values()),'reference_files':reference_count,'shared_code_files':len(shared_code),'weights':records,'verified':True,'scope':'file integrity and checkpoint recovery; not a new RR/FR evaluation'}
    if a.output:a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
