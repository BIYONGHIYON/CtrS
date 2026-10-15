"""Read-only startup checkpoint, source and ETA verification; no RR/FR evaluation."""
import argparse, datetime, hashlib, importlib, json, math, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(1)


def main():
    module=importlib.import_module('experiments.ssai.10_pan_innovation_correction.model')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    a=parser.parse_args(); p=json.loads((a.runtime/'plan.json').read_text())
    for name,digest in p['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    results={};now=time.time()
    for arm in ('A1','A2'):
        out=a.runtime/arm; history=json.loads((out/'history.json').read_text())
        status=json.loads((out/'status.json').read_text())
        checkpoints={}
        for kind in ('best','latest'):
            path=out/(kind+'.pt');ck=torch.load(path,map_location='cpu',weights_only=False)
            assert ck['config']['variant']==arm and ck['config']['plan']==p
            assert ck['epoch']>=1 and math.isfinite(ck['validation_mse'])
            assert all(key in ck for key in ('optimizer','loader_rng','torch_rng','cuda_rng','python_rng','numpy_rng'))
            assert ck['optimizer']['state']
            m=module.PanInnovationNet(innovation=arm=='A2');m.load_state_dict(ck['model'],strict=True);m.eval()
            with torch.inference_mode():y=m(torch.rand(1,1,32,48),torch.rand(1,4,32,48),torch.rand(1,4,8,12))
            assert y.shape==(1,4,32,48) and torch.isfinite(y).all()
            checkpoints[kind]={'epoch':ck['epoch'],'bytes':path.stat().st_size,'strict_load':True,'finite_cpu_forward':True,'optimizer_rng_preserved':True}
        seconds=sum(row['seconds'] for row in history[-3:])/min(3,len(history))
        # Snapshot ETA from completed epochs; ignore the unfinished epoch to stay
        # conservative. The launch staggering is included in each worker's history.
        eta=now+(100-history[-1]['epoch'])*seconds
        results[arm]={'completed_epochs':history[-1]['epoch'],'status':status,'latest_history':history[-1],'mean_recent_epoch_seconds':seconds,'eta_epoch_seconds':eta,'eta_kst':datetime.datetime.fromtimestamp(eta,datetime.timezone(datetime.timedelta(hours=9))).isoformat(),'checkpoints':checkpoints}
    result={'verified':True,'checked_utc':now,'source_files_verified':len(p['source_hashes']),'checkpoint_verification_is_not_performance_evaluation':True,'arms':results}
    dest=a.runtime/'startup_verification.json';dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def retained_main():
    import tempfile, zipfile
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.root.resolve();e=root/'experiments/ssai/10_pan_innovation_correction'
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest=json.loads((e/'manifest.json').read_text())
    for name,record in manifest['files'].items():
        path=e/name
        assert path.is_file() and path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],name
    for name,record in manifest['shared_code'].items():
        assert (root/name).stat().st_size==record['bytes'] and sha(root/name)==record['sha256'],name
    refs=json.loads((e/'provenance/reference_assets.json').read_text())['files']['A0']
    for record in refs.values():assert sha(root/record['repository_path'])==record['sha256']
    with tempfile.TemporaryDirectory(prefix='ssai10-restore-') as temp:
        t=Path(temp)
        with zipfile.ZipFile(e/'provenance/training_source.zip') as z:
            plan=json.loads(z.read('plan.json'))
            assert plan==json.loads((e/'provenance/plan.json').read_text())
            for name,digest in plan['source_hashes'].items():assert hashlib.sha256(z.read('source/'+name)).hexdigest()==digest,name
            path=t/'references/upstream/network.py';path.parent.mkdir(parents=True);path.write_bytes(z.read('source/references/upstream/network.py'))
            for name in ('common/model.py','experiments/ssai/10_pan_innovation_correction/model.py'):
                assert sha(root/name)==plan['source_hashes'][name]
                path=t/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((root/name).read_bytes())
        (t/'common/checkpoint.py').write_bytes((root/'common/checkpoint.py').read_bytes())
        # Use submitted adapter/model/loader with the exact original upstream source.
        for key in list(sys.modules):
            if key=='common' or key.startswith('common.') or key=='experiments' or key.startswith('experiments.'):
                del sys.modules[key]
        sys.path.insert(0,str(t))
        from common import checkpoint
        checkpoint.RESEARCH_TRACK='ssai'
        initial=torch.load(e/'provenance/initial_state.pt',map_location='cpu',weights_only=False)
        digest=hashlib.sha256()
        for name,value in sorted(initial.items()):
            digest.update(name.encode());digest.update(str(tuple(value.shape)).encode());digest.update(str(value.dtype).encode());digest.update(value.contiguous().numpy().tobytes())
        assert digest.hexdigest()==plan['initial_common_sha256']
        records={};changes={}
        selection=json.loads((e/'results/evaluation/selection.json').read_text())
        for arm in ('A1','A2'):
            history=json.loads((e/f'results/training/{arm}/history.json').read_text())
            assert [row['epoch'] for row in history]==list(range(1,101))
            best=min(history,key=lambda row:row['val_mse'])
            done=json.loads((e/f'results/training/{arm}/complete.json').read_text())
            for kind in ('best','latest'):
                path=e/f'checkpoints/{arm}_QB_s42/{kind}.pt'
                net,ck=checkpoint.load_model(10,path)
                assert sha(path)==done[kind+'_sha256']
                assert ck['config']['plan']==plan and ck['config']['variant']==arm
                assert ck['epoch']==(best['epoch'] if kind=='best' else 100)
                assert ck['optimizer']['state'] and all(key in ck for key in ('loader_rng','torch_rng','cuda_rng','numpy_rng','python_rng'))
                assert all(torch.isfinite(value).all() for value in ck['model'].values())
                if kind=='best':
                    assert ck['validation_mse']==best['val_mse'] and selection['selected'][arm]['sha256']==sha(path)
                with torch.inference_mode():
                    y=net(torch.rand(1,1,32,48),torch.rand(1,4,32,48),torch.rand(1,4,8,12))
                assert y.shape==(1,4,32,48) and torch.isfinite(y).all()
                if kind=='latest':
                    changes[arm]={}
                    for stage in ('SSA_blocks','SSA_blocks1','SSA_blocks2'):
                        keys=[k for k in initial if k.startswith(stage+'.') and k.endswith('.weight') and '.conv6t12.' not in k]
                        num=sum((ck['model'][k].double()-initial[k].double()).square().sum().item() for k in keys)
                        den=sum(initial[k].double().square().sum().item() for k in keys)
                        value=math.sqrt(num/den)
                        assert abs(value-history[-1]['active_convolution_weight_relative_change'][stage])<1e-12
                        changes[arm][stage]=value
                records[arm+'/'+kind]={'sha256':sha(path),'epoch':ck['epoch'],'strict_load':True,'finite_cpu_forward':True,'optimizer_rng_preserved':True}
    for arm in ('A0','A1','A2'):
        for protocol in ('RR','FR'):
            report=json.loads((e/f'results/evaluation/{arm}/{protocol}.json').read_text())
            assert report['samples']==20 and len(report['rows'])==20
            assert report['checkpoint_sha256']==selection['selected'][arm]['sha256']
            assert len(report['images'])==(5 if protocol=='RR' else 0)
    result={'verified':True,'manifest_files':len(manifest['files']),'shared_code_files':len(manifest['shared_code']),'reference_files':len(refs),'checkpoints':records,'latest_weight_changes_recomputed':changes,'scope':'artifact integrity and checkpoint recovery; NOT RR/FR performance reevaluation'}
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    if '--runtime' in sys.argv:main()
    else:retained_main()
