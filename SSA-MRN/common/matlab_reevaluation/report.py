"""Compact MATLAB-vs-Python evidence and presentation tables; no invented paper scores."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import numpy as np
from suite import read,write,sha,cases


def average(rows,key):
    return {k:float(np.mean([r[key][k] for r in rows])) for k in rows[0][key]}


def previous(meta,old):
    case=meta['case'];arm=case['arm'];seed=case['seed']
    if case['experiment']==53:return old['models'][f'{arm}_s{seed}']['mean']
    return old['models'][f"{arm}_{case['trained_sensor']}_s{seed}"]['targets'][case['sensor']]['mean']


def summarize(out):
    out=Path(out);meta=read(out/'prediction_manifest.json');matlab=read(out/'matlab_scores.json');python=read(out/'python_legacy.json')
    if matlab['case']!=meta['case'] or matlab['dlpan_commit']!=meta['dlpan_commit']:raise ValueError('MATLAB result provenance mismatch')
    expected={(p,i) for p in ('RR','RR2','FR_WALD','FR') for i in range(20)}
    if {(r['protocol'],r['scene']) for r in matlab['rows']}!=expected or len(matlab['rows'])!=80:raise ValueError('Incomplete/duplicate MATLAB scenes')
    old=previous(meta,read(out/'previous_report.json'))
    means,legacy,delta,drift={}, {}, {}, {}
    for protocol in ('RR','RR2','FR_WALD','FR'):
        mrows=[r for r in matlab['rows'] if r['protocol']==protocol]
        prows=[r for r in python if r['protocol']==protocol]
        means[protocol]=average(mrows,'metrics');legacy[protocol]=average(prows,'python_legacy')
        common=set(means[protocol])&set(legacy[protocol])
        delta[protocol]={k:means[protocol][k]-legacy[protocol][k] for k in common}
        if protocol!='FR':
            delta[protocol]['PSNR_bandmean_vs_legacy']=means[protocol]['PSNR_bandmean']-legacy[protocol]['PSNR']
        drift[protocol]={k:legacy[protocol][k]-old[protocol][k] for k in legacy[protocol] if k in old[protocol]}
    obs=float(np.mean([r['FR_OBS_RMSE_DN'] for r in python if r['protocol']=='FR']))
    source=meta['case']
    result=dict(case=source,matlab_mean=means,python_legacy_reinference_mean=legacy,stored_previous_mean=old,
                matlab_minus_python=delta,python_reinference_minus_previous=drift,
                FR_OBS_RMSE_DN=obs,FR_OBS_note='Existing sensor observation operator in Python, not a DLPan/MATLAB metric',
                peak=meta['peak'],profile=matlab['profile'],matlab_release=matlab['matlab_release'],
                checkpoint_sha256=meta['checkpoint_sha256'],checkpoint_best_epoch=meta['checkpoint_best_epoch'],
                dlpan_commit=meta['dlpan_commit'],source_commit=meta['source_commit'],complete=True,
                direct_metric_delta=not matlab['profile']['cut_bounds'] and not matlab['profile']['clip_output'],
                inference_seconds=meta['inference_seconds'],matlab_seconds=matlab['seconds'],
                limits=['CPU FP32 reinference may differ slightly from previous GPU results; drift is reported separately',
                        'Same dataset, crop, clipping, PSNR definition and QNR variant must be verified for each comparison paper',
                        'Matlab execution alone does not establish identical evaluation settings to an unspecified paper'])
    write(out/'summary.json',result)
    with (out/'per_scene.csv').open('w',newline='',encoding='utf8') as f:
        fields=['experiment','sensor','arm','seed','protocol','scene','metric','matlab','python_legacy','difference']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        pmap={(r['protocol'],r['scene']):r['python_legacy'] for r in python}
        for row in matlab['rows']:
            for k,value in row['metrics'].items():
                py=pmap[row['protocol'],row['scene']].get(k)
                writer.writerow(dict(experiment=source['experiment'],sensor=source['sensor'],arm=source['arm'],seed=source['seed'],
                    protocol=row['protocol'],scene=row['scene'],metric=k,matlab=value,python_legacy=py,
                    difference=value-py if py is not None else ''))
    # Hash predictions AND the exact DLPan files, without uploading large input arrays.
    files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    write(out/'artifact_manifest.json',dict(files=files,retained_predictions=False,
        note='Prediction hashes are retained; ephemeral arrays can be regenerated from pinned sources/data/weights'))
    return result


def aggregate(source,out,expected_cases):
    source,out=Path(source),Path(out);out.mkdir(parents=True,exist_ok=True)
    results=[read(p) for p in source.rglob('summary.json')]
    found=[r['case']['id'] for r in results]
    if len(found)!=len(set(found)):raise ValueError('Duplicate case summary')
    missing=set(expected_cases)-set(found)
    if missing:raise ValueError('Missing completed evaluations: '+str(sorted(missing)))
    profiles={json.dumps(r['profile'],sort_keys=True) for r in results}
    # Peak differs by sensor, so compare settings per sensor, not peak across sensors.
    for sensor in ('QB','GF2','WV3','WV2'):
        group=[r for r in results if r['case']['sensor']==sensor]
        if len({json.dumps(r['profile'],sort_keys=True) for r in group})>1:raise ValueError('Mixed evaluation profiles')
    table=[]
    for result in sorted(results,key=lambda r:(r['case']['experiment'],r['case']['sensor'],r['case']['arm'],r['case']['seed'])):
        case=result['case']
        for protocol,metrics in result['matlab_mean'].items():
            for metric,value in metrics.items():
                table.append(dict(**{k:case[k] for k in ('experiment','sensor','arm','seed')},protocol=protocol,metric=metric,value=value))
    with (out/'matlab_per_seed.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=('experiment','sensor','arm','seed','protocol','metric','value'));writer.writeheader();writer.writerows(table)
    groups={}
    for row in table:
        key=tuple(row[k] for k in ('experiment','sensor','arm','protocol','metric'))
        groups.setdefault(key,[]).append(row['value'])
    mean_rows=[dict(zip(('experiment','sensor','arm','protocol','metric'),key),mean=float(np.mean(values)),
                    sample_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,seeds=len(values)) for key,values in groups.items()]
    with (out/'matlab_seed_mean_sd.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=('experiment','sensor','arm','protocol','metric','mean','sample_sd','seeds'));writer.writeheader();writer.writerows(mean_rows)
    lines=['# 실험 53·54 실제 MATLAB 재평가','',
           '재학습 없이 기존 validation best를 CPU FP32로 재추론하고 공식 DLPan MATLAB 함수로 계산했습니다. '
           'GF2 peak는 1023, 나머지는 2047입니다. 기존 GF2 peak2047 결과와 구분합니다. '
           'PSNR은 DLPan 원래 반환값이 아니므로 MATLAB에서 global-MSE와 band-mean을 각각 계산했습니다.','',
           '| 실험 | 센서 | 모델 | 시드 수 | RR SAM ↓ | RR ERGAS ↓ | RR PSNR global ↑ | RR PSNR bandmean ↑ | RR SCC ↑ | RR Q2n ↑ | FR Dλ ↓ | FR Ds ↓ | FR QNR ↑ |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for exp,sensor,arm in sorted({(r['case']['experiment'],r['case']['sensor'],r['case']['arm']) for r in results}):
        rows=[r for r in mean_rows if (r['experiment'],r['sensor'],r['arm'])==(exp,sensor,arm)]
        lookup={(r['protocol'],r['metric']):r for r in rows}
        def cell(protocol,key):
            r=lookup[protocol,key]
            return f"{r['mean']:.4f}"+(f" ± {r['sample_sd']:.4f}" if r['sample_sd'] is not None else '')
        n=rows[0]['seeds']
        cols=[cell('RR',k) for k in ('SAM','ERGAS','PSNR_global','PSNR_bandmean','SCC','Q2n')]+[cell('FR',k) for k in ('D_lambda','D_s','QNR')]
        lines.append('| '+' | '.join([str(exp),sensor,arm,str(n),*cols])+' |')
    lines+=['','## 논문 비교 조건','',
        '이 표는 새 MATLAB 실측값입니다. 비교할 논문의 데이터·test20장·공간 배율·crop·clipping·PSNR 정의·QNR 변형이 같은지 확인한 뒤 함께 사용해야 합니다. '
        '특정 논문이 지정되지 않았으므로 논문 수치를 복사하거나 동등 비교라고 표시하지 않았습니다.','',
        '기존 Python 재추론과 과거 보고값의 차이, MATLAB과 Python의 차이는 각 case summary.json에 분리했습니다. '
        'FR_OBS는 기존 Python 관측 오차로 별도 보존하며 MATLAB 지표라고 표시하지 않습니다. '
        '각 시드의 20장 평균을 먼저 구한 뒤 시드 평균±표본 SD를 계산했습니다.','',
        '재평가 옵션·MATLAB release·source/data/checkpoint SHA-256은 case별 manifest에 있습니다. 과거 보고서는 덮어쓰지 않았습니다.']
    (out/'presentation_table.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    write(out/'all_cases.json',results)
    write(out/'completion.json',dict(complete=True,cases=len(results),expected=len(expected_cases),missing=[],training=False))
    summary=os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary,'a',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('case');s.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('aggregate');a.add_argument('--input',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--matrix',type=Path,required=True)
    args=p.parse_args()
    if args.mode=='case':summarize(args.out)
    else:aggregate(args.input,args.out,[r['id'] for r in read(args.matrix)['include']])


if __name__=='__main__':main()
