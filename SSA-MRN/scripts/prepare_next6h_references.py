"""Move verified references; clean exact inactive remotely recoverable training copies."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import torch
import a6000_followup_suite as common


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')


def inactive(paths):
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU tasks active'
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            cwd=os.readlink(p/'cwd');cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        assert not any(cwd==str(x) or cwd.startswith(str(x)+'/') or str(x) in cmd for x in paths),(p.name,cwd,cmd)


def verify_assets(path,manifest):
    for name,a in manifest['artifacts'].items():
        p=path/name;assert p.is_file() and p.stat().st_size==a['bytes'] and sha(p)==a['sha256'],str(p)
    names={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    assert names<=set(manifest['artifacts'])|{'report_manifest.json'},names-set(manifest['artifacts'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply-cleanup',action='store_true');a=parser.parse_args()
    plan=json.loads((ROOT/'scripts/a6000_next_6h_plan.json').read_text())
    catalog=json.loads((ROOT/'references/committed_weight_catalog.json').read_text());known=catalog['weights'];commit=catalog['verified_remote_commit']
    old=Path('/home/gpu_04/CtrS-a6000-followup');arch=Path('/home/gpu_04/CtrS-a6000/SSA-MRN/experiments/a6000_architecture_qb')
    metadata=ROOT/'references/archive_evidence';metrics=json.loads((metadata/'followup_metrics.json').read_text())
    followup_manifest=json.loads((metadata/'followup_manifest.json').read_text());arch_manifest=json.loads((metadata/'architecture_manifest.json').read_text())
    reference_root=Path(plan['reference_root']);reference_root.mkdir(parents=True,exist_ok=True);refs={}
    if not a.apply_cleanup:
        for row in plan['references']:
            name=common.name(row);source=arch/name if row['sensor']=='QB' and row['seed']==42 and row['variant'] in ['baseline','interp23'] else old/'SSA-MRN/experiments/a6000_followup_123'/name
            dest=Path(row['path']);dest.mkdir(parents=True,exist_ok=True)
            complete=json.loads((source/'complete.json').read_text());history=[json.loads(x) for x in (source/'history.jsonl').read_text().splitlines()]
            assert complete['epoch']==100 and complete['config']==common.config(plan,row)
            assert history==metrics['runs'][name]['history'];refs[name]={'source':str(source),'files':{}}
            for filename in ['best.pt','latest.pt','complete.json','config.json','history.jsonl']:
                p=source/filename;q=dest/filename
                if filename.endswith('.pt'):
                    digest=sha(p);assert digest in known,(str(p),'full checkpoint not committed')
                    ck=torch.load(p,map_location='cpu',weights_only=True);assert ck['config']==complete['config']
                    assert {'model','config','optimizer','rng','cuda_rng','loader_rng'}<=set(ck)
                    if filename=='best.pt':assert ck['epoch']==min(history,key=lambda x:x['val_mse'])['epoch']
                    else:assert ck['epoch']==100
                shutil.copy2(p,q);assert sha(p)==sha(q)
                refs[name]['files'][filename]={'bytes':q.stat().st_size,'sha256':sha(q),'git_blobs':known.get(sha(q),[]) if filename.endswith('.pt') else []}
        shutil.copy2(old/'SSA-MRN/experiments/a6000_followup_123/inventory.json',reference_root/'inventory.json')
        diagnostics=ROOT/'references/next6h_diagnostics';diagnostics.mkdir(exist_ok=True)
        for variant in ['baseline','interp23']:
            name=f'WV3_{variant}_k6_s42';p=old/'SSA-MRN/experiments/a6000_followup_123'/name/'best.pt';assert sha(p) in known
            q=diagnostics/(name+'_best.pt');shutil.copy2(p,q);assert sha(p)==sha(q)
        write(reference_root/'reference_manifest.json',{'archive_commit':commit,'reference_runs':refs,'policy':'unaltered full checkpoints; external training data/env preserved'})
        print('PREPARED',len(refs),'reference runs,18 full checkpoints and2 WV3 diagnostic bests')
        return
    # Verify local immutable references again before removing any old source.
    ref_manifest=json.loads((reference_root/'reference_manifest.json').read_text())
    for name,record in ref_manifest['reference_runs'].items():
        for filename,artifact in record['files'].items():
            p=reference_root/name/filename;assert sha(p)==artifact['sha256']
    assert len(ref_manifest['reference_runs'])==9
    for name,record in metrics['runs'].items():
        source=arch/name if record['reused'] else old/'SSA-MRN/experiments/a6000_followup_123'/name
        for kind,checkpoint in record['checkpoints'].items():
            p=source/(kind+'.pt');assert sha(p)==checkpoint['sha256'] and sha(p) in known
        history=[json.loads(x) for x in (source/'history.jsonl').read_text().splitlines()];assert history==record['history']
        complete=json.loads((source/'complete.json').read_text());assert complete['epoch']==100 and complete['config']==record['config']
    verify_assets(old/'SSA-MRN/docs/assets/a6000_followup_123',followup_manifest)
    verify_assets(Path('/home/gpu_04/CtrS-a6000/SSA-MRN/docs/assets/a6000_architecture_qb'),arch_manifest)
    for name,h in followup_manifest['code'].items():assert sha(old/'SSA-MRN'/name)==h,name
    status=subprocess.check_output(['git','status','--porcelain'],cwd=old,text=True).splitlines()
    allowed={'SSA-MRN/src/ssamrn/models/pan_variants.py','SSA-MRN/docs/operations/a6000_followup.md','SSA-MRN/scripts/a6000_followup_plan.json','SSA-MRN/scripts/a6000_followup_suite.py','SSA-MRN/scripts/run_a6000_followup.sh','SSA-MRN/scripts/train_a6000_followup.py','SSA-MRN/scripts/verify_a6000_followup.py','SSA-MRN/scripts/evaluate_a6000_followup.py','SSA-MRN/docs/assets/a6000_followup_123/','SSA-MRN/experiments/a6000_followup_123/'}
    assert all(x[3:] in allowed for x in status),status
    candidates=[old,arch/'QB_baseline_k6_s42',arch/'QB_interp23_k6_s42',Path('/home/gpu_04/CtrS-a6000/SSA-MRN/docs/assets/a6000_architecture_qb')]
    old_checkpoints=Path('/home/gpu_04/CtrS_old/SSA-MRN/experiments/checkpoints')
    for relative in ['qb_full','gf2_full','wv3_full','k6/qb_full','k6/gf2_full','k6/wv3_full']:
        p=old_checkpoints/relative
        assert {str(f.relative_to(p)) for f in p.rglob('*') if f.is_file()}=={'latest.pt'},str(p)
        assert sha(p/'latest.pt') in known,str(p)
        ck=torch.load(p/'latest.pt',map_location='cpu',weights_only=True);assert 'model' in ck and 'epoch' in ck
        candidates.append(p)
    inactive(candidates)
    protected=[Path('/home/gpu_04/CtrS_old'),Path('/home/gpu_04/CtrS-a6000'),ROOT]
    assert not any(p in protected for p in candidates)
    assert not any(str(Path(row['path'])).startswith(str(p)+'/') for row in plan['references'] for p in candidates)
    records=[];before=common.quota_free()
    for p in candidates:
        assert p.exists() and not p.is_symlink()
        size=int(subprocess.check_output(['du','-sb',str(p)],text=True).split()[0]);records.append({'path':str(p),'logical_bytes':size,'reason':'inactive exact full weights and required outputs restored from GitHub; references moved before removal'})
    backup=Path('/home/gpu_04/a6000_followup_delivery.tar.gz')
    if backup.exists():
        assert sha(backup)=='aed60d045d9a5d31ed74d9912154a6066d66f5045d63654c29b0577f3f904291'
        records.append({'path':str(backup),'logical_bytes':backup.stat().st_size,'reason':'verified duplicate delivery archive; original weights restored remotely and active references copied'})
    # The list is fixed above; never recursively clean home, datasets or environments.
    for p in candidates:shutil.rmtree(p)
    if backup.exists():backup.unlink()
    record={'checked_at_unix':time.time(),'remote_archive_commit':commit,'deleted':records,'quota_free_before_bytes':before,'quota_free_after_bytes':common.quota_free(),'preserved':['/home/gpu_04/CtrS_old environment and unrelated RGB-HSI files','/home/gpu_04/CtrS-a6000 original dataset and code','QB_lr_correction_k6_s42 and QB_high_frequency_k6_s42: full Adam/RNG weights not committed',str(reference_root)]}
    write(ROOT/'docs/operations/a6000_next6h_cleanup.json',record)
    print('CLEANED',len(records),'exact paths; quota recovered',(record['quota_free_after_bytes']-before)/1024**2,'MiB')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
