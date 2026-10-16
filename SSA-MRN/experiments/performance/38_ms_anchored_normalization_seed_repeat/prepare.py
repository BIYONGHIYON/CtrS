"""Verify retained original bytes and build the repeated-seed report (no training).

Run from anywhere after placing this file in the experiment folder.
Only --register updates the five shared experiment indexes; no Git commands run.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NAME = '38_ms_anchored_normalization_seed_repeat'
BASE = HERE.parent / '33_ms_anchored_reversible_normalization'
METRICS = {'RR': ['PSNR', 'SAM', 'ERGAS', 'SCC', 'Q2n'],
           'FR': ['QNR', 'D_lambda', 'D_s']}
HIGHER = {'PSNR', 'SCC', 'Q2n', 'QNR'}
DETAIL = {'worker.log', 'events.jsonl', 'status.json'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode('utf-8'))


def dump(path, obj):
    write(path, json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def evidence(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def retained_path(seed, name):
    prov = HERE / 'provenance' / f'seed{seed}'
    result = HERE / 'results' / f'seed{seed}'
    if name.startswith('checkpoints/'):
        return HERE / name
    if name.startswith('provenance/initial_'):
        arm = Path(name).stem.split('_')[1]
        return HERE / 'checkpoints' / f'{arm}_QB_s{seed}' / 'initial.pt'
    if name.startswith('provenance/'):
        return prov / name.removeprefix('provenance/')
    if name.startswith('results/'):
        return result / name.removeprefix('results/')
    if name in {'config.json', 'environment.json', 'data_manifest.json',
                'selection.json', 'protocol.json'}:
        return prov / name
    return result / name


def verify_originals():
    counts = {}
    for seed in (43, 44):
        original = read(HERE / 'provenance' / f'seed{seed}' / 'original_manifest.json')
        count = 0
        for name, expected in original['files'].items():
            if Path(name).name in DETAIL:
                continue
            path = retained_path(seed, name)
            require(path.is_file(), f'Missing original: {path}')
            require(evidence(path) == expected, f'Original hash mismatch: {path}')
            count += 1
        counts[str(seed)] = count
    return counts


def collect():
    summaries = {42: read(BASE / 'results' / 'summary.json')}
    configs = {42: read(BASE / 'provenance' / 'config.json')}
    checks = {}
    for seed in (43, 44):
        folder = HERE / 'results' / f'seed{seed}'
        configs[seed] = read(HERE / 'provenance' / f'seed{seed}' / 'config.json')
        summaries[seed] = read(folder / 'summary.json')
        states = []
        for arm in ('N0', 'N1'):
            history = read(folder / arm / 'history.json')
            require([r['epoch'] for r in history] == list(range(101)), 'Incomplete history')
            best = min(history, key=lambda r: (r['val_mse'], r['epoch']))
            complete = read(folder / arm / 'complete.json')
            require(best['epoch'] == complete['best_epoch'], 'Validation best mismatch')
            for kind in ('best', 'latest'):
                path = HERE / 'checkpoints' / f'{arm}_QB_s{seed}' / f'{kind}.pt'
                require(evidence(path)['sha256'] == complete[f'{kind}_sha256'], 'Weight mismatch')
            states.append(read(folder / arm / 'initialization.json')['state_sha256'])
            checks[f'{seed}/{arm}'] = {'best_epoch': best['epoch'],
                                      'best_val_mse': best['val_mse'],
                                      'best_val_psnr': best['val_psnr'],
                                      'best_val_sam': best['val_sam']}
            for index in (1, 8, 11, 13, 19):
                require((folder / arm / 'results' / f'RR_scene_{index:03}.png').is_file(),
                        'Missing fixed preview')
        require(states[0] == states[1], 'Paired initialization differs')
        selection = read(HERE / 'provenance' / f'seed{seed}' / 'selection.json')
        require(selection['indices'] == [1, 8, 11, 13, 19], 'Preview selection differs')
        require(configs[seed]['seed'] == seed and configs[seed]['epochs'] == 100,
                'Unexpected training config')
        for key in ('source_hashes', 'epochs', 'batch', 'micro', 'lr', 'std_floor',
                    'precision', 'selection', 'data_normalization'):
            require(configs[seed][key] == configs[42][key], f'Condition differs: {key}')
    per_seed = {}
    for seed, summary in summaries.items():
        item = {'means': {}, 'deltas': {}, 'improved_scenes': {}}
        for arm in ('N0', 'N1'):
            item['means'][arm] = {}
            for protocol, metrics in METRICS.items():
                obj = summary['metrics'][arm]['protocols'][protocol]
                require([r['index'] for r in obj['rows']] == list(range(20)), 'Scene set differs')
                if seed in (43, 44):
                    raw = read(HERE / 'results' / f'seed{seed}' / arm / 'results' / f'{protocol}.json')
                    require(raw == obj, 'Summary and raw scene metrics differ')
                item['means'][arm][protocol] = {}
                for metric in metrics:
                    value = statistics.mean(r['metrics'][metric] for r in obj['rows'])
                    require(math.isclose(value, obj['mean'][metric], abs_tol=1e-10), 'Mean differs')
                    item['means'][arm][protocol][metric] = value
        for protocol, metrics in METRICS.items():
            for metric in metrics:
                key = f'{protocol}_{metric}'
                delta = item['means']['N1'][protocol][metric] - item['means']['N0'][protocol][metric]
                require(math.isclose(delta, summary['deltas'][key], abs_tol=1e-10), 'Delta differs')
                item['deltas'][key] = delta
                rows0 = summary['metrics']['N0']['protocols'][protocol]['rows']
                rows1 = summary['metrics']['N1']['protocols'][protocol]['rows']
                sign = 1 if metric in HIGHER else -1
                item['improved_scenes'][key] = sum(
                    sign * (b['metrics'][metric] - a['metrics'][metric]) > 0
                    for a, b in zip(rows0, rows1))
        per_seed[str(seed)] = item
    aggregate = {}
    for key in per_seed['42']['deltas']:
        values = [per_seed[str(s)]['deltas'][key] for s in (42, 43, 44)]
        aggregate[key] = {'mean_delta_N1_minus_N0': statistics.mean(values),
                          'sample_sd': statistics.stdev(values)}
    return configs, per_seed, aggregate, checks


def chart(per_seed):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="700" viewBox="0 0 1000 700">',
           '<rect width="1000" height="700" fill="white"/>',
           '<g font-family="sans-serif" fill="#18283b">',
           '<text x="28" y="30" font-size="20">Repeated seeds: delta N1 - N0 (20 scenes per protocol)</text>']
    keys = list(per_seed['42']['deltas'])
    for i, key in enumerate(keys):
        x, y = 20 + (i % 2) * 495, 55 + (i // 2) * 155
        values = [per_seed[str(s)]['deltas'][key] for s in (42, 43, 44)]
        extent = max(abs(v) for v in values) * 1.3 or 1
        center = x + 240
        svg.append(f'<text x="{x+8}" y="{y+16}" font-size="15">{key} ({"higher" if key.split("_",1)[1] in HIGHER else "lower"} is better)</text>')
        svg.append(f'<line x1="{center}" y1="{y+24}" x2="{center}" y2="{y+126}" stroke="#8392a5"/>')
        for j, (seed, value) in enumerate(zip((42, 43, 44), values)):
            yy = y + 40 + j * 29
            width = abs(value) / extent * 145
            start = center if value >= 0 else center - width
            good = value * (1 if key.split('_',1)[1] in HIGHER else -1) > 0
            color = '#187d6b' if good else '#c54b42'
            svg.append(f'<text x="{x+10}" y="{yy+12}" font-size="13">seed {seed}</text>')
            svg.append(f'<rect x="{start:.3f}" y="{yy}" width="{width:.3f}" height="17" fill="{color}"/>')
            svg.append(f'<text x="{x+400}" y="{yy+12}" font-size="12">{value:+.6f}</text>')
    svg.append('</g></svg>')
    write(HERE / 'results' / 'seed_deltas.svg', '\n'.join(svg) + '\n')


def report(per_seed, aggregate, checks):
    out = ['# MS 기준 가역 정규화의 반복 시드 검증', '',
           '**연구 분야:** 기존 성능 개선', '', '**실험자:** 김경찬', '',
           '**변경 내용:** 33번의 LR MS 기준 공유 정규화·역변환을 유지하고 시드43·44로 독립 반복했습니다.', '',
           '| 확인된 결과 | 상태 | 개선 여부 |', '|---|:---:|:---:|',
           '| 세 시드의 RR 평균 5지표가 모두 개선됐으나 seed44의 잠정 FR QNR·Ds가 악화했습니다. | 평가 완료 | 부분 개선 |', '',
           '[실험 목록](../../../research/performance/README.md#experiments) · [선행 실험](../33_ms_anchored_reversible_normalization/README.md)', '',
           '## 1. 목적과 상태', '',
           '33번의 단일 시드 이득이 다른 초기값에서도 반복되는지 확인했습니다. 시드43·44의 N0·N1 학습을 각각100에폭 완료했고 validation best로 RR·FR 각각20장 전체를 평가했습니다. 시드42는33번에 보관된 결과를 참조하며 다시 학습하지 않았습니다.', '',
           '## 2. 변경한 구조와 비교 조건', '',
           'N0는 초기 LRPAN을 포함한 원본 SSA-MRN K4입니다. N1은 LR MS의 C·H·W 전체에서 샘플별 스칼라 평균 μ와 모집단 표준편차 σ를 구하고 s=max(σ,0.01)로 설정합니다. PAN·LMS·MS에 같은 (x−μ)/s를 적용하고 원본 모델의 출력에 s를 곱한 뒤 μ를 더합니다. GT는 입력 통계 계산에 사용하지 않습니다. 추가 학습 파라미터는 없습니다.', '',
           '| 항목 | 실제 조건 |', '|---|---|',
           '| 데이터 | QuickBird; train17,139·validation1,905패치; RR20·FR20장 |',
           '| 내부 특징 차원·공간 배율 | K=4·4배 |',
           '| 시드 | 선행42·추가43,44; 시드별 N0/N1 공통 초기 모델·동일 학습 순서 |',
           '| 학습 | fresh initialization; 100에폭; effective batch64·micro batch32 |',
           '| 최적화·손실 | Adam·고정 LR0.0001·원래 DN/2047 단위 MSE |',
           '| 입력 범위 | DN/2047·clipping 없음 |',
           '| 장치 | Kaggle Tesla T4 두 장; GPU0 N0·GPU1 N1 독립 프로세스 |',
           '| 정밀도 | 학습 FP16 AMP; 통계·역변환·손실·validation·test FP32; TF32 off |',
           '| 가중치 선택 | epoch0 포함 최소 validation MSE; 정확한 동률이면 이른 에폭 |', '',
           '<a id="results"></a>', '', '## 3. 정량 결과', '',
           'RR은 정답이 있는 축소 해상도 평가이며 PSNR·SCC·Q2n은 높을수록, SAM·ERGAS는 낮을수록 좋습니다. FR은 정답 없는 원해상도 평가이며 QNR은 높을수록, Dλ·Ds는 낮을수록 좋습니다. QNR·Q2n·SCC는 MATLAB 수치 대응이 완전히 확인되지 않은 잠정 지표입니다. 아래 차이는 N1−N0입니다.', '',
           '| 시드 | 군 | best 에폭 | validation MSE | validation PSNR | validation SAM |',
           '|---|---|---:|---:|---:|---:|']
    for seed in (43, 44):
        for arm in ('N0', 'N1'):
            c = checks[f'{seed}/{arm}']
            out.append(f"| {seed} | {arm} | {c['best_epoch']} | {c['best_val_mse']:.10f} | {c['best_val_psnr']:.6f} | {c['best_val_sam']:.6f} |")
    for protocol, metrics in METRICS.items():
        out += ['', f'### {protocol} 전체 장면 평균', '',
                '| 시드 | 군 | ' + ' | '.join(metrics) + ' |',
                '|---|---|' + '---:|' * len(metrics)]
        for seed in (42, 43, 44):
            item = per_seed[str(seed)]
            for arm in ('N0', 'N1'):
                out.append(f'| {seed} | {arm} | ' + ' | '.join(f'{item["means"][arm][protocol][m]:.6f}' for m in metrics) + ' |')
            out.append(f'| {seed} | 차이 | ' + ' | '.join(f'{item["deltas"][protocol+"_"+m]:+.6f}' for m in metrics) + ' |')
    out += ['', '### 세 시드의 paired 차이', '',
            '장면별 지표를 각 시드 안에서 평균한 후 동일 시드의 N1−N0를 구했습니다. 아래 표준편차는 세 시드 차이의 표본 표준편차이며 신뢰구간이나 유의성 검정 결과가 아닙니다.', '',
            '| 지표 | 평균 차이 | 시드 간 표본 표준편차 |', '|---|---:|---:|']
    for key, value in aggregate.items():
        out.append(f'| {key} | {value["mean_delta_N1_minus_N0"]:+.6f} | {value["sample_sd"]:.6f} |')
    out += ['', '## 4. 기준선 대비 차이와 해석', '',
            '세 시드 모두 RR의 평균5지표가 개선했습니다. 다만 seed44는 개별 장면에서 PSNR17/20·ERGAS18/20·SCC16/20장만 개선했으므로 모든 장면의 일관된 개선으로 해석하지 않습니다. seed43은 잠정 FR도 개선했지만 seed44는 QNR−0.000938·Ds+0.002439로 악화했습니다. 세 시드의 평균 잠정 QNR 차이는 양수이지만 FR 이득의 일관성은 확보하지 못했습니다.', '',
            '시드42 단독 결과의 기록은33번에 유지합니다. 이번 반복 검증은 같은 QB test를 사용한 탐색 결과이며 독립 holdout 검증이나 다른 센서의 개선을 의미하지 않습니다.', '',
            '<a id="graphs"></a>', '', '## 5. 학습·평가 그래프', '',
            '원본 실측 train MSE·validation PSNR·SAM 곡선과 각 시드의 baseline 비교 그래프를 표시합니다. 통합 그래프의 막대는 원래 지표 단위의 N1−N0이며 녹색이 개선 방향입니다.', '',
            '![세 시드 지표 차이](results/seed_deltas.svg)', '']
    for seed in (43, 44):
        out += [f'### 시드{seed}', '', f'![시드{seed} 학습 곡선](results/seed{seed}/curves.png)', '',
                f'![시드{seed} 기준선 비교](results/seed{seed}/comparison.png)', '']
    out += ['<a id="images"></a>', '', '## 6. 결과 이미지', '',
            '학습 전에 고정한 0부터 시작하는 RR 장면1·8·11·13·19를 각 반복 시드에서 표시합니다. 패널 순서는 LR MS nearest 확대·PAN·N1 예측·GT입니다. RGB 밴드[2,1,0]를 사용하고 MS 패널은 해당 장면 GT의1·99백분위 대비를 공유합니다. 전체 장면을 표시하며 별도 crop을 선택하지 않았습니다. FR에는 GT가 없습니다. N0의 같은 장면 이미지는 각 결과 폴더에 함께 보관했습니다.', '']
    for seed in (43, 44):
        out += [f'### 시드{seed} N1', '']
        for index in (1, 8, 11, 13, 19):
            out += [f'![시드{seed} RR 장면{index}](results/seed{seed}/N1/results/RR_scene_{index:03}.png)', '']
    out += ['<a id="evidence"></a>', '', '## 7. 재현 자료', '',
            '[집계 설정](config.json) · [검증·집계 코드](prepare.py) · [통합 숫자](results/summary.json) · [보존 검증](results/retention_verification.json) · [manifest](manifest.json)', '',
            '실제 학습된 초기·best·latest 가중치12개를 [checkpoints](checkpoints/)에 보관했습니다. 시드42 가중치는 선행33번을 참조합니다. 각 시드의 원본 설정·데이터 해시·고정 장면 선택·소스 snapshot·실행 코드는 [seed43](provenance/seed43/)과 [seed44](provenance/seed44/)에, 전체 장면별 지표와 compact epoch 기록은 [seed43 결과](results/seed43/)와 [seed44 결과](results/seed44/)에 보관했습니다.', '',
            '원본 실행 코드의 학습 당시 설정을 유지했습니다. prepare.py는 Python 표준 라이브러리만 사용해 원본 해시·validation best·장면 평균을 검증하고 보고서를 다시 만듭니다. --register를 지정하면 공용 실험 목록5개도 갱신합니다. 학습이나 Git 커밋·푸시는 수행하지 않습니다.', '',
            '```bash', f'python SSA-MRN/experiments/performance/{NAME}/prepare.py',
            'python SSA-MRN/common/verify.py --research-track performance --experiment 38', '```', '',
            '이번 보관 검증은 원본 manifest와 복사된 파일의 바이트 일치, 기록된 validation best, RR·FR 전체 장면 평균을 확인했습니다. 원본 H5로 추론·지표를 다시 계산하거나 반복 가중치를 strict load한 검증은 수행하지 않았습니다. 학습 당시 preflight 기록은 별도로 보존했습니다.', '',
            '## 8. 개선 여부와 다음 단계', '',
            '**부분 개선·최종 채택 보류**로 판정합니다. RR 이득은 세 시드에서 반복됐으나 FR 개선은 일관되지 않습니다. 다음에는 지표의 수치 대응을 확인하고, 동일 조건의 추가 센서·독립 holdout에서 검증합니다. 이번 test 결과에 맞춰 사후 선택한 설정을 독립 검증 성능으로 보고하지 않습니다.', '']
    write(HERE / 'README.md', '\n'.join(out))


def register():
    key = f'performance/{NAME}'
    prefix = f'experiments/performance/{NAME}'
    entry = {'report': prefix + '/README.md', 'research_track': 'performance',
             'experiment': 38, 'display_name': '38. MS 기준 가역 정규화의 반복 시드 검증',
             'experimenter': '김경찬', 'status': '평가 완료', 'improvement': '부분 개선',
             'checkpoints': prefix + '/checkpoints', 'results': prefix + '/results',
             'manifest': prefix + '/manifest.json', 'config': prefix + '/config.json',
             'entrypoint': prefix + '/prepare.py', 'evaluated_seeds': [43, 44],
             'aggregate_seeds': [42, 43, 44], 'sensor': 'QB', 'reference_experiment': 33,
             'limits': '세 시드 RR 평균5지표 개선; seed44 잠정 FR QNR·Ds 악화. 부분 개선·최종 채택 보류. H5 재평가·반복 가중치 strict load·MATLAB 수치 대응 미검증.'}
    pending = {}
    for relative in ('SSA-MRN/experiments/catalog.json', 'SSA-MRN/research/experiments.json'):
        path = ROOT / relative
        obj = read(path)
        collisions = [k for k, v in obj.items() if v.get('research_track') == 'performance'
                      and v.get('experiment') == 38 and k != key]
        require(not collisions, f'Experiment38 already used: {collisions}')
        require(key not in obj or obj[key].get('report') == entry['report'], 'Existing registration differs')
        obj[key] = entry
        pending[path] = json.dumps(obj, ensure_ascii=False, indent=2) + '\n'
    for relative, link in (
        ('README.md', f'SSA-MRN/{prefix}/README.md'),
        ('SSA-MRN/README.md', prefix + '/README.md'),
        ('SSA-MRN/research/performance/README.md', f'../../{prefix}/README.md')):
        path = ROOT / relative
        text = path.read_text(encoding='utf-8-sig')
        lines = text.splitlines()
        row = f'| [38. MS 기준 가역 정규화의 반복 시드 검증]({link})<br>[김경찬] | 33번의 정규화·역변환을 유지하고 seed43·44로 반복했습니다. | 세 시드 평균 PSNR +0.1800dB·잠정 QNR +0.00761; seed44 QNR·Ds 악화, 채택 보류 | 평가&nbsp;완료 | 부분&nbsp;개선 |'
        existing = [i for i, line in enumerate(lines) if f'/{NAME}/README.md)' in line]
        if existing:
            require(len(existing) == 1, 'Duplicate experiment row')
            lines[existing[0]] = row
        else:
            matches = [i for i, line in enumerate(lines) if line.startswith('| [')
                       and ('performance/' in line or relative.endswith('performance/README.md'))]
            require(bool(matches), f'Cannot locate performance table: {relative}')
            lines.insert(matches[-1] + 1, row)
        pending[path] = '\n'.join(lines) + '\n'
    for path, text in pending.items():
        write(path, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    require(HERE.name == NAME and (ROOT / 'SSA-MRN').is_dir(), 'Put prepare.py in the experiment38 folder')
    counts = verify_originals()
    configs, per_seed, aggregate, checks = collect()
    config = {'research_track': 'performance', 'experiment': 38, 'experimenter': '김경찬',
              'purpose': 'archive and aggregate the completed experiment33 seed repeats',
              'new_training_seeds': [43, 44], 'aggregate_seeds': [42, 43, 44],
              'reference_experiment': 33, 'status': '평가 완료', 'improvement': '부분 개선',
              'training_configs': {str(s): f'provenance/seed{s}/config.json' for s in (43, 44)},
              'training_sources': {str(s): f'provenance/seed{s}/run_cell.py' for s in (43, 44)},
              'reference_summary': '../33_ms_anchored_reversible_normalization/results/summary.json'}
    dump(HERE / 'config.json', config)
    dump(HERE / 'results' / 'summary.json', {'per_seed': per_seed, 'aggregate_paired_deltas': aggregate,
         'validation_selection': checks, 'status': 'partial improvement; final adoption deferred',
         'aggregation': '20-scene mean per seed, then paired N1-N0; sample SD across three seeds'})
    dump(HERE / 'results' / 'retention_verification.json', {
        'original_manifest_verified_files': counts, 'checkpoint_count': 12,
        'rr_fr_scene_means_recomputed': True, 'validation_selection_checked': True,
        'paired_initial_state_digest_checked': True, 'cross_seed_source_hashes_checked': True,
        'h5_re_evaluation_performed': False, 'repeat_checkpoint_strict_load_performed': False,
        'excluded_completed_runtime_files': sorted(DETAIL),
        'external_reference_files': {str(p.relative_to(ROOT)): evidence(p) for p in
            (BASE / 'results' / 'summary.json', BASE / 'provenance' / 'config.json')}})
    chart(per_seed)
    report(per_seed, aggregate, checks)
    if args.register:
        register()
    files = {}
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or path == HERE / 'manifest.json':
            continue
        if path.name in DETAIL or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        files[path.relative_to(HERE).as_posix()] = evidence(path)
    require(sum(n.endswith('.pt') for n in files) == 12, 'Expected12 checkpoints')
    dump(HERE / 'manifest.json', {'research_track': 'performance', 'experiment': 38,
         'experimenter': '김경찬', 'files': files,
         'scope': 'retained bytes and stored metric arithmetic; no H5 re-evaluation'})
    print(f'PASS: original retained files {counts}; 12 checkpoints; RR/FR means; validation best')
    print('Report, aggregate metrics, chart and manifest generated.')
    print('Five experiment indexes updated.' if args.register else 'Shared indexes unchanged (use --register).')


if __name__ == '__main__':
    main()
