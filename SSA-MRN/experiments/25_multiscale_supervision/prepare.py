"""Audit retained results and weights; never train or overwrite original evidence."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def audit(source=None,archive=None):
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(2)
    original=read(HERE/'provenance/artifact_manifest.json')['artifacts']
    mapped=read(HERE/'provenance/retained_sources.json')['mapping']
    for item in mapped:
        p=HERE/item['retained']
        assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],str(p)
    original_count=None
    if source is not None:
        for name,item in original.items():
            p=source/name
            assert p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],name
        original_count=len(original)
    archive_check=None
    if archive is not None:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None,'ZIP CRC mismatch'
            assert set(original).issubset(z.namelist())
        archive_check={'bytes':archive.stat().st_size,'sha256':digest(archive),'CRC_passed':True,
                       'all_original_artifacts_present':True,'source_supplied_sha256_sidecar':False}
    suite=read(HERE/'provenance/suite_config.json')
    assert suite['k']==4 and suite['epochs']==200 and suite['seeds']==[42]
    selection=read(HERE/'provenance/selection.json')
    assert selection['zero_based_scene_indices']==[1,8,11,13,19]
    config=read(HERE/'config.json')
    assert config['experimenter']=='김윤지'
    models=importlib.util.spec_from_file_location('pyramid_current',HERE/'model.py')
    current=importlib.util.module_from_spec(models);models.loader.exec_module(current)
    sys.path.insert(0,str(HERE/'provenance/code/CtrS/SSA-MRN/src'))
    oldspec=importlib.util.spec_from_file_location('pyramid_original',HERE/'provenance/code/B0_DS_models.py')
    old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old)
    local=ast.parse((HERE/'run.py').read_text(encoding='utf-8'))
    literal={n.targets[0].id:ast.literal_eval(n.value) for n in local.body if isinstance(n,ast.Assign)
             and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('WORKER','PASTED_MODEL_SOURCE')}
    assert literal['WORKER'].encode()==(HERE/'provenance/code/kaggle_worker.py').read_bytes()
    assert literal['PASTED_MODEL_SOURCE'].encode()==(HERE/'provenance/code/B0_DS_models.py').read_bytes()
    assert digest(HERE/'run.py')==digest(HERE/'provenance/run_cell.py')
    results=read(HERE/'provenance/results.json')
    generator=torch.Generator().manual_seed(7)
    inputs=tuple(torch.rand(shape,generator=generator) for shape in ((1,1,64,64),(1,4,64,64),(1,4,16,16)))
    stats={};checks=[];histories={};all_images=[]
    data=read(HERE/'provenance/data_manifest.json')
    for arm in ('B0','B0_DS'):
        run_id=f'{arm}_QB_{arm}_k4_s42';result=results[run_id]
        history=read(HERE/'results'/arm/'history.json');histories[arm]=history
        assert len(history)==200 and [r['epoch'] for r in history]==list(range(1,201))
        best_index=min(range(len(history)),key=lambda i:history[i]['val_mse'])
        assert result['best_epoch']==best_index+1==199
        assert result['best_mse']==history[best_index]['val_mse']
        record={'best_epoch':199,'best_validation':history[best_index],'latest_validation':history[-1],
                'means':{},'checkpoint_checks':[]}
        for protocol in ('RR','FR'):
            rows=read(HERE/'results'/arm/f'{protocol}_per_scene.json')
            assert rows==result['metrics'][protocol]['rows']
            assert len(rows)==data[protocol]['shapes']['pan'][0]==20
            assert [r['scene_index'] for r in rows]==list(range(20))
            means={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key!='scene_index'}
            for key,value in means.items():
                assert math.isfinite(value) and abs(value-result['metrics'][protocol]['mean'][key])<1e-12,key
            record['means'][protocol]=means
        folder=HERE/'checkpoints'/f'{arm}_QB_s42'
        for kind in ('best','latest','best_first100'):
            path=folder/f'{kind}.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
            assert ck['config']['variant']==arm and ck['config']['k']==4
            assert ck['epoch']==(199 if kind=='best' else 200 if kind=='latest' else min(history[:100],key=lambda r:r['val_mse'])['epoch'])
            assert ck['best_mse']==min(history[:ck['epoch']],key=lambda r:r['val_mse'])['val_mse']
            assert len(ck['history'])==ck['epoch']
            for a,b in zip(ck['history'],history):
                for key in ('train_mse','val_mse','val_psnr_peak1','val_sam_deg','sample_order_sha256',
                            'val_aux_quarter_l1','val_aux_half_l1','val_aux_full_l1'):
                    assert a[key]==b[key],key
            for key in ('optimizer','scaler','cpu_rng','cuda_rng','loader_rng','python_rng','numpy_rng','lr_state'):
                assert key in ck,key
            if kind=='latest':
                assert hashlib.sha256(ck['best_recovery_bytes']).hexdigest()==digest(folder/'best.pt')
                assert ck['mid_epoch'] is None
            if kind in ('best','latest'):
                assert digest(path)==result['checkpoints'][kind]['sha256']
            net=current.build_model(arm,4,4);net.load_state_dict(ck['model'],strict=True);net.eval()
            original_net=old.build_model(arm,4,4);original_net.load_state_dict(ck['model'],strict=True);original_net.eval()
            with torch.inference_mode():
                prediction=net(*inputs);ref=original_net(*inputs)
            torch.testing.assert_close(prediction,ref,rtol=0,atol=0)
            assert prediction.shape==(1,4,64,64) and torch.isfinite(prediction).all()
            point={'path':str(path.relative_to(HERE)).replace('\\','/'),'epoch':ck['epoch'],
                   'sha256':digest(path),'strict_load':True,'finite_CPU_forward':True,
                   'current_vs_archived_forward_bitwise':True}
            record['checkpoint_checks'].append(point);checks.append(point)
        images=result['images']
        assert [x['scene_index'] for x in images]==selection['zero_based_scene_indices']
        for item in images:
            p=HERE/'results'/arm/item['file']
            with Image.open(p) as im:
                assert im.size==(1024,296);im.verify()
            all_images.append(str(p.relative_to(HERE)).replace('\\','/'))
        stats[arm]=record
    assert [r['sample_order_sha256'] for r in histories['B0']]==[r['sample_order_sha256'] for r in histories['B0_DS']]
    initial=[read(HERE/'results'/arm/'initialization.json')['core_sha256'] for arm in ('B0','B0_DS')]
    assert initial[0]==initial[1]
    for a,b in zip(results['B0_QB_B0_k4_s42']['images'],results['B0_DS_QB_B0_DS_k4_s42']['images']):
        assert a['display']==b['display']
    paired=read(HERE/'provenance/paired_comparison.json')[0]
    summary={};wins={}
    for protocol in ('RR','FR'):
        summary[protocol]={};wins[protocol]={}
        for key,value in stats['B0_DS']['means'][protocol].items():
            delta=value-stats['B0']['means'][protocol][key]
            assert abs(delta-paired['metric_delta_B0_DS_minus_B0'][protocol][key])<1e-12
            summary[protocol][key]=delta
            if key=='inference_ms':continue
            direction=-1 if key in ('PSNR','PSNR_sensor_peak','SCC','Q2n','QNR') else 1
            a=read(HERE/'results/B0'/f'{protocol}_per_scene.json');b=read(HERE/'results/B0_DS'/f'{protocol}_per_scene.json')
            changes=[(y[key]-x[key])*direction for x,y in zip(a,b)]
            wins[protocol][key]={'improved':sum(v<0 for v in changes),'worse':sum(v>0 for v in changes),'equal':sum(v==0 for v in changes)}
    return {'status':'provided evidence consistent; partial improvement; final RR worsened',
        'checked_at_KST':'2026-10-13','experimenter':'김윤지','source_manifest_checked_files':original_count,
        'retained_original_files_checked':len(mapped),'archive':archive_check,
        'embedded_model_worker_match_provided_code':True,'200_epoch_order_and_initialization_match':True,
        'validation_minimum_selected':True,'original_winner_bytes_recoverable_from_latest':True,
        'statistics':stats,'candidate_minus_baseline':summary,'scene_counts':wins,
        'checkpoints':checks,'images':all_images,
        'scope':'Recorded results/means/best/history/hash consistency and six CPU strict-load forwards verified. '
                'No original-H5 metric recomputation, fresh training, T4 rerun or CUDA resume-parity test.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,help='Read-only original extracted output')
    p.add_argument('--archive',type=Path,help='Optional original results ZIP for CRC/SHA256')
    p.add_argument('--output',type=Path,required=True,help='New verification JSON; existing files refused')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=audit(a.source,a.archive)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f"PASS: {len(result['checkpoints'])} original checkpoint loads; 80 scene scores; 200 paired epochs")

if __name__=='__main__':main()
