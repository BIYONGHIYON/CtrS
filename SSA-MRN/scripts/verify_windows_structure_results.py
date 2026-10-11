"""Validate archived full structure-suite weights,450epochs,validation choice and80test rows."""
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('S0','S1','S2','S3','R1','R2','M1','I1','I2')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets',type=Path,default=ROOT/'docs/assets/windows_structure_qb_50');parser.add_argument('--check-torch',action='store_true');parser.add_argument('--data',type=Path);a=parser.parse_args();r=a.assets
    manifest=read(r/'report_manifest.json')
    for n,x in manifest['artifacts'].items():assert (r/n).stat().st_size==x['bytes'] and sha(r/n)==x['sha256'],n
    for n,h in manifest['delivery_source_hashes'].items():assert sha(ROOT/n)==h,n
    inv=read(r/'server_inventory.json')
    for n,x in inv['artifacts'].items():
        p=r/('runs/'+n if n.split('/')[0] in VARIANTS else n)
        assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],n
    plan=read(r/'plan.json');assert (plan['epochs'],plan['micro'],plan['effective_batch'],plan['k'],plan['seed'])==(50,256,256,4,42)
    for n,h in plan['source_hashes'].items():
        snapshot=r/'source_snapshot'/n
        assert sha(snapshot)==h,n
        assert (ROOT/n).read_bytes().replace(b'\r\n',b'\n')==snapshot.read_bytes().replace(b'\r\n',b'\n'),n
    assert sha(r/'data_manifest.json')==plan['data_manifest_sha256']
    ranking=read(r/'ranking.json')['rows'];assert {x['variant'] for x in ranking}==set(VARIANTS)
    histories={};selected={}
    for v in VARIANTS:
        d=r/'runs'/v;h=read(d/'history.json');histories[v]=h
        assert [x['epoch'] for x in h]==list(range(1,51))
        assert all(math.isfinite(x[k]) for x in h for k in ['train_mse','val_mse','val_sam','seconds'])
        best=min(h,key=lambda x:x['val_mse']);selected[v]=best
        row=next(x for x in ranking if x['variant']==v);assert row['best_mse']==best['val_mse'] and row['epoch']==best['epoch'] and row['val_sam']==best['val_sam']
        complete=read(d/'complete.json');assert complete['epochs']==50 and complete['variant']==v and complete['best_mse']==best['val_mse']
        for kind in ['best','latest']:
            assert sha(d/(kind+'.pt'))==complete[kind+'_sha256']==manifest['artifacts']['runs/'+v+'/'+kind+'.pt']['sha256']
            assert inv['checkpoints'][v][kind]['epoch']==(best['epoch'] if kind=='best' else 50)
    assert ranking==sorted(ranking,key=lambda x:(x['best_mse'],x['val_sam'],VARIANTS.index(x['variant'])))
    suite=read(r/'suite_complete.json');queue=read(r/'queue.json');assert suite['winner']==ranking[0]['variant']=='I1' and suite['epochs']==50
    assert set(queue['completed'])==set(VARIANTS) and not queue['failed'] and not queue['active']
    report=read(r/'evaluation/metrics.json');assert set(report['runs'])=={'S0','I1'}
    selection=read(r/'evaluation/selection.json');assert selection['indices']==[1,8,11,13,19] and selection==report['selection']==dict(seed=42,indices=plan['evaluation_selection']['indices'],order='ascending',tile='full scene',frozen_before_test=True)
    for v,item in report['runs'].items():
        assert item['best_epoch']==selected[v]['epoch'] and item['best_sha256']==sha(r/'runs'/v/'best.pt')
        for protocol,record in item['metrics'].items():
            assert [row['index'] for row in record['rows']]==list(range(20))
            for k,m in record['mean'].items():
                values=[x['metrics'][k] for x in record['rows']];assert all(math.isfinite(x) for x in values) and math.isclose(sum(values)/20,m,rel_tol=1e-12,abs_tol=1e-14)
        for index in selection['indices']:assert (r/f'evaluation/{v}_scene_{index:02}.png').is_file()
    if a.check_torch:
        import sys,torch
        sys.path.insert(0,str(ROOT/'src'));from ssamrn.models.windows_structure import StructureNet
        torch.set_num_threads(1)
        for v in VARIANTS:
            for kind in ['best','latest']:
                ck=torch.load(r/'runs'/v/(kind+'.pt'),map_location='cpu',weights_only=False)
                assert ck['config']==dict(plan=plan,variant=v)
                fingerprint=hashlib.sha256(json.dumps(ck['config'],sort_keys=True).encode()).hexdigest()
                assert ck['fingerprint']==fingerprint==read(r/'runs'/v/'complete.json')['fingerprint']
                assert ck['epoch']==(selected[v]['epoch'] if kind=='best' else 50) and ck['next_batch']==0
                assert all(k in ck for k in ['optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng'])
                model=StructureNet(v).eval();model.load_state_dict(ck['model'],strict=True)
                if kind=='best':
                    with torch.inference_mode():pred=model(torch.ones(1,1,64,64),torch.ones(1,4,64,64),torch.ones(1,4,16,16))
                    assert pred.shape==(1,4,64,64) and torch.isfinite(pred).all()
        if a.data:
            import h5py,numpy as np
            paths=list(a.data.rglob('test_qb_multiExm1.h5'));assert len(paths)==1
            assert sha(paths[0])==report['runs']['S0']['metrics']['RR']['data_sha256']
            with h5py.File(paths[0]) as f:
                tensors=[torch.from_numpy(np.asarray(f[k][0],dtype=np.float32)/2047.).unsqueeze(0) for k in ['pan','lms','ms']]
                gt=torch.from_numpy(np.asarray(f['gt'][0],dtype=np.float32)/2047.)
            for v in ['S0','I1']:
                ck=torch.load(r/'runs'/v/'best.pt',map_location='cpu',weights_only=False);model=StructureNet(v).eval();model.load_state_dict(ck['model'],strict=True)
                with torch.inference_mode():pred=model(*tensors)[0]
                mse=float((pred-gt).double().square().mean());expected=report['runs'][v]['metrics']['RR']['rows'][0]['metrics']['MSE_peak1']
                assert math.isclose(mse,expected,rel_tol=1e-4,abs_tol=1e-10),(v,mse,expected)
    print('PASS:18 original full checkpoints,450 epochs,validation I1,80 full-scene test rows,10 fixed four-panel images'+(';strict loads/inference passed' if a.check_torch else ''))
if __name__=='__main__':main()
