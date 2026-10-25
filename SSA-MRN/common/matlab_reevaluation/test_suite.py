"""Local checks. Synthetic report fixtures are NOT MATLAB or model performance results."""
import ast
import copy
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
from suite import HERE,REPO,EXP,cases,checkpoint,network,data_files,sha,read,write
from report import summarize,aggregate


def main():
    torch.set_num_threads(1)
    jobs=cases()
    assert len(jobs)==31 and len({x['id'] for x in jobs})==31
    assert len(cases((53,)))==13 and len(cases((54,)))==18
    assert all(x['trained_sensor']=='WV3' for x in jobs if x['sensor']=='WV2')
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),filename=str(p))
    checked=[]
    for case in jobs:
        path,expected=checkpoint(case)
        assert sha(path)==expected,(case['id'],'checkpoint hash')
        model=network(case).eval()
        ck=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model'],strict=True)
        if case['seed']==42:
            bands=8 if case['sensor'] in ('WV3','WV2') else 4
            with torch.inference_mode():
                y=model(torch.rand(1,1,32,32),torch.rand(1,bands,32,32),torch.rand(1,bands,8,8))
            assert y.shape==(1,bands,32,32) and torch.isfinite(y).all()
        checked.append(case['id'])
        del model,ck
    for sensor in ('QB','GF2','WV3','WV2'):
        for protocol,spec in data_files(sensor).items():
            assert len(spec['sha256'])==64 and spec['revision'] in spec['url']
    # Golden expected arithmetic in report plumbing, with clearly synthetic MATLAB inputs.
    with tempfile.TemporaryDirectory(dir=REPO/'tmp') as temp:
        temp=Path(temp);artifacts=temp/'cases';out=temp/'presentation'
        selected=[x for x in jobs if x['experiment']==54 and x['sensor']=='GF2' and x['arm']=='C2MS']
        for case in selected:
            dest=artifacts/case['id'];dest.mkdir(parents=True)
            meta=dict(case=case,dlpan_commit='test_only',checkpoint_sha256='test_only',checkpoint_best_epoch=1,
                      source_commit='synthetic',peak=1023,inference_seconds=0)
            write(dest/'prediction_manifest.json',meta)
            python=[];matlab=[]
            for protocol in ('RR','RR2','FR_WALD','FR'):
                for scene in range(20):
                    metrics=dict(D_lambda=.02,D_s=.03,QNR=.9506) if protocol=='FR' else dict(SAM=3.,ERGAS=4.,SCC=.9,Q2n=.8,PSNR=30.)
                    python.append(dict(protocol=protocol,scene=scene,python_legacy=metrics,FR_OBS_RMSE_DN=2.))
                    new=copy.deepcopy(metrics)
                    if protocol!='FR':
                        new.pop('PSNR');new.update(Q=.8,PSNR_global=23.,PSNR_bandmean=24.)
                    matlab.append(dict(protocol=protocol,scene=scene,metrics=new))
            write(dest/'python_legacy.json',python)
            write(dest/'previous_report.json',read(EXP[54]/'results/evaluation_v2_3.json'))
            write(dest/'matlab_scores.json',dict(case=case,dlpan_commit='test_only',rows=matlab,profile=dict(peak=1023,cut_bounds=False,clip_output=False),matlab_release='SYNTHETIC_TEST_NOT_MATLAB',seconds=0))
            result=summarize(dest)
            assert result['matlab_minus_python']['RR']['PSNR_bandmean_vs_legacy']==-6
            assert result['matlab_mean']['RR']['PSNR_global']==23
        aggregate(artifacts,out,[x['id'] for x in selected])
        assert read(out/'completion.json')['cases']==3
        assert '± 0.0000' in (out/'presentation_table.md').read_text(encoding='utf8')
        try:aggregate(artifacts,out,[x['id'] for x in selected]+['missing'])
        except ValueError:pass
        else:raise AssertionError('Missing case was silently accepted')
    write(HERE/'local_validation.json',dict(status='local CPU checks passed',retained_best_hashes_and_strict_loads=checked,
          synthetic_forward_checked='seed42 all 11 target/arm combinations',matrix_cases=31,
          report_plumbing='synthetic fixtures only; not a MATLAB validation',
          training_executed=False,matlab_executed_locally=False,real_test_reinference_locally=False,
          unverified=['GitHub Actions runtime','actual MATLAB execution','full real-data CPU inference']))
    print('31 retained best hash/strict loads and forward/report tests passed. MATLAB not executed locally.')


if __name__=='__main__':main()
