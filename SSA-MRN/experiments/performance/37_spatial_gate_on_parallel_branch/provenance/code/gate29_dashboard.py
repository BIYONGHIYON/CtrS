"""Notebook presentation only: never imports torch or changes worker/config state."""
import html
import json
import math
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

STAGES = {'starting': '시작 준비', 'load_gpu_data': '데이터 → GPU', 'train': '학습',
          'validation': '검증', 'epoch_complete': '에폭 완료', 'paused': '체크포인트 저장 · 일시정지',
          'training_complete_no_test': '학습 완료 · test 미실행', 'evaluate_RR': 'RR 평가',
          'evaluate_FR': 'FR 평가', 'evaluation_complete': '평가 완료', 'error': '오류',
          'interrupted': '중단', 'idle': '대기'}
PHASES = {'calibrate': '손실 크기 확인 · λ 후보 고정', 'smoke': '소량 실행 · 저장/재개 점검',
          'sweep': 'λ 후보 비교 · validation만 사용', 'select': 'λ 선택 · 최종 설정 고정',
          'final_train': '본 학습 · 현재 seed의 두 군', 'final_test': '최종 RR/FR 평가'}

def esc(value):
    return html.escape(str(value), quote=True)

def number(value, digits=4):
    try:
        value = float(value)
        return f'{value:.{digits}g}' if math.isfinite(value) else '—'
    except (TypeError, ValueError):
        return '—'

def progress(value):
    try:
        value = float(value)
        if not math.isfinite(value): return '<span class="muted">측정 대기</span>'
    except (TypeError, ValueError):
        return '<span class="muted">—</span>'
    value = max(0., min(100., value))
    return f'<div class="bar"><i style="width:{value:.1f}%"></i></div><small>{value:.1f}%</small>'

def render(phase, rows, note='', data_rows=(), gpu_stats=None, updated=''):
    gpu_stats = gpu_stats or {}
    body, metric_body = [], []
    for gpu in (0, 1):
        row = rows.get(gpu, {})
        v = row.get('status', {})
        stage = v.get('stage', 'idle')
        epoch, epochs = v.get('epoch'), row.get('epochs')
        batch, batches = v.get('batch'), v.get('batches')
        fraction = None
        if stage == 'train' and batches:
            fraction = 100 * batch / batches if batch is not None else None
        elif stage in ('validation', 'epoch_complete', 'training_complete_no_test'):
            fraction = 100
        elif stage.startswith('evaluate_') and v.get('scenes'):
            fraction = 100 * v.get('scene', 0) / v['scenes']
        elif stage == 'evaluation_complete':
            fraction = 100
        unit = '장면' if stage.startswith('evaluate_') else '배치'
        count = (f"{v.get('scene', '—')}/{v.get('scenes', '—')}" if unit == '장면'
                 else f'{batch}/{batches}' if stage == 'train' and batches else '—')
        epoch_text = f'{epoch}/{epochs}' if epoch is not None and epochs else '—'
        # On validation/terminal reports, loss/metrics may refer to the last observed report.
        metrics = row.get('metrics', {})
        allocated = number(v.get('allocated_gib'), 3)
        peak = number(v.get('peak_gib'), 3)
        stats = gpu_stats.get(gpu, {})
        device = f"{number(stats.get('used_gib'), 3)} / {number(stats.get('total_gib'), 3)} GiB"
        util = number(stats.get('utilization'), 3)
        eta = v.get('train_eta_kst')
        try: eta = datetime.fromisoformat(eta).strftime('%m/%d %H:%M:%S') if eta else '측정 대기'
        except (ValueError, TypeError): eta = '측정 대기'
        if stage in ('training_complete_no_test', 'evaluation_complete'): eta = '완료'
        elif stage in ('paused', 'error', 'interrupted', 'idle') or stage.startswith('evaluate_'): eta = '—'
        badge = 'bad' if stage in ('error', 'interrupted') else 'good' if stage in ('training_complete_no_test', 'evaluation_complete') else ''
        error = row.get('error') or '—'
        body.append(f'''<tr><td><b>GPU {gpu} · {esc(row.get('label', 'baseline' if gpu == 0 else 'gated'))}</b>
          <small>{esc(row.get('context', '작업 대기'))}</small><small>대기 작업 {esc(row.get('queued', 0))}개</small></td>
          <td><span class="badge {badge}">{esc(STAGES.get(stage, stage))}</span></td>
          <td>{esc(epoch_text)}</td><td>{progress(fraction)}<small>{unit} {esc(count)}</small></td>
          <td>{number(row.get('throughput'), 4)}</td>
          <td>{allocated} / {peak}<small>장치 {device}</small><small>사용률 {util}%</small></td>
          <td class="error">{esc(error)}</td></tr>''')
        metric_body.append(f'<tr><td><b>GPU {gpu} · {esc(row.get("label", "baseline" if gpu == 0 else "spectral"))}</b></td>' +
                           ''.join(f'<td>{number(metrics.get(key), 5)}</td>' for key in
                           ('train_mse', 'val_mse', 'val_psnr', 'val_sam', 'gate_mean')) +
                           f'<td>{esc(eta)}<small>KST · 학습/검증/저장</small></td></tr>')
    dataset = ''
    if data_rows:
        dataset = '<details><summary>데이터 확인 결과 · 상세 수치/해시는 data_manifest.json</summary><table><tr><th>분할</th><th>전체 / 사용 장면·패치</th><th>LMS 크기 (NCHW)</th><th>검사</th></tr>'
        for split, info in data_rows:
            shape = info['arrays']['lms']['shape']
            dataset += f'<tr><td>{esc(split)}</td><td>{shape[0]:,} / {info["used_samples"]:,}</td><td>{esc(" × ".join(map(str, shape)))}</td><td>유한값 · 범위 · SHA-256 확인</td></tr>'
        dataset += '</table></details>'
    return '''<style>
      .spectral-dashboard {font-family:system-ui,-apple-system,"Malgun Gothic",sans-serif;color:#172b4d;background:#fff;border:1px solid #dce4ef;border-radius:12px;padding:18px;margin:12px 0;font-size:13px;line-height:1.5}
      .spectral-dashboard h3 {font-size:18px;margin:0 0 5px;color:#172b4d}.spectral-dashboard h4 {font-size:13px;margin:16px 0 0}.spectral-dashboard .muted,.spectral-dashboard small{color:#66758b}.spectral-dashboard small{display:block;font-size:11px}
      .spectral-dashboard .scroll {overflow-x:auto}.spectral-dashboard table{width:100%;border-collapse:collapse;margin:12px 0;white-space:nowrap}
      .spectral-dashboard th{background:#edf2f8;color:#34465e;font-weight:600;text-align:left;padding:10px 8px}
      .spectral-dashboard td{padding:12px 8px;border-bottom:1px solid #e5ebf3;vertical-align:top}.spectral-dashboard tr:nth-child(even) td{background:#f8fafc}
      .spectral-dashboard .badge{display:inline-block;padding:3px 7px;background:#eaf2ff;color:#2455a4;border-radius:5px}.spectral-dashboard .good{background:#e8f6ed;color:#237746}.spectral-dashboard .bad{background:#ffebeb;color:#a52626}
      .spectral-dashboard .bar{width:92px;height:6px;background:#e1e8f2;border-radius:3px;overflow:hidden;margin:6px 0}.spectral-dashboard .bar i{display:block;height:100%;background:#427ee8}
      .spectral-dashboard .notice{padding:9px 12px;background:#f0f5fb;border-radius:7px;margin:10px 0;white-space:pre-wrap;overflow-wrap:anywhere}
      .spectral-dashboard .error{max-width:200px;min-width:40px;white-space:normal;overflow-wrap:anywhere;color:#a52626}.spectral-dashboard summary{cursor:pointer}
      </style>''' + f'''<section class="spectral-dashboard"><h3>W2SpatialGate_1st · {esc(PHASES.get(phase, phase))}</h3>
      <span class="muted">K=4 · QuickBird · GPU0 W2 / GPU1 gated W2 · 갱신 {esc(updated or '—')}</span>
      <div class="notice">{esc(note or '준비 중')}</div>{dataset}<div class="scroll"><table><thead><tr>
      <th>실험</th><th>상태</th><th>에폭</th><th>현재 에폭 / 평가 진행</th>
      <th>패치/초</th><th>모델 할당 / 최대 GiB</th><th>오류</th>
      </tr></thead><tbody>{''.join(body)}</tbody></table></div>
      <h4>손실 · 최근 완료 Validation · 학습 종료 예상</h4><div class="scroll"><table><thead><tr>
      <th>실험</th><th>Train MSE</th><th>Val MSE</th><th>Val PSNR</th><th>Val SAM</th><th>게이트 평균</th><th>종료 예상</th>
      </tr></thead><tbody>{''.join(metric_body)}</tbody></table></div>
      <small>손실은 최근 worker 보고값입니다. Validation은 최근 완료 에폭 값이며 best 값과 다를 수 있습니다. ERGAS·SCC·Q2n은 고정 validation 32개 패치의 보조 지표입니다. 전체 validation MSE로 best를 선택합니다.</small>
      <small>패치/초는 같은 에폭의 두 학습 보고 사이에서 실측합니다. 첫 보고 전에는 —로 표시합니다. 장치 메모리는 다른 프로세스도 포함합니다.</small>
      <small>종료 예상에는 대기 작업·test 평가·그림·ZIP 시간이 포함되지 않습니다. 상태 표는 약 5초마다, worker 손실/진행 보고는 약 20초마다 갱신됩니다.</small>
      </section>'''

class Dashboard:
    def __init__(self, phase, root):
        self.phase, self.root = phase, Path(root)
        self.rows = {0: {}, 1: {}}
        self.data_rows, self.note, self.gpu_stats = [], '데이터 검사 준비 중', {}
        self.handle = None
        self.last_gpu = self.last_plain = 0.

    def data_checked(self, split, info):
        self.data_rows.append((split, info))
        self.note = f'{split} 검사 완료 · 전체 {info["arrays"]["lms"]["shape"][0]:,} / 사용 {info["used_samples"]:,}개'
        self.refresh()

    def starting(self, job, queued):
        g, config = job['gpu'], job['config']
        self.rows[g] = {'label': job['label'], 'epochs': config['epochs'], 'queued': queued,
                        'samples': config['data']['train']['used_samples'], 'batch_size': config['batch'],
                        'context': f'{job["folder"].parent.name} · λ={config["weight"]:g} · {job["folder"].parent.parent.name}',
                        'status': {'stage': 'starting'}, 'metrics': {}, 'throughput': None}

    def observe(self, gpu, value):
        row = self.rows[gpu]
        previous = row.get('status', {})
        row['throughput'] = None
        if value.get('stage') == previous.get('stage') == 'train' and value.get('epoch') == previous.get('epoch'):
            try:
                seconds = (datetime.fromisoformat(value['updated_kst']) - datetime.fromisoformat(previous['updated_kst'])).total_seconds()
                # Report batch is an effective group; cap the final partial group's sample count.
                count = row['samples']
                delta = min(value['batch'] * row['batch_size'], count) - min(previous['batch'] * row['batch_size'], count)
                if seconds > 0 and delta > 0: row['throughput'] = delta / seconds
            except (ValueError, KeyError, TypeError): pass
        if isinstance(value.get('gate'),dict): value={**value,'gate_mean':value['gate'].get('mean')}
        row['status'] = value
        for key in ('train_mse', 'val_mse', 'val_psnr', 'val_sam', 'gate_mean'):
            if key in value: row['metrics'][key] = value[key]

    def finish(self, job, returncode=0, skipped=False):
        g = job['gpu']
        row = self.rows[g]
        if returncode:
            row['status'] = {**row.get('status', {}), 'stage': 'error'}
            row['error'] = f'종료 코드 {returncode} · {job["folder"].relative_to(self.root)}/worker.log 확인'
        else:
            marker = 'evaluation_complete.json' if job['config']['job_type'] == 'test' else 'training_complete.json'
            complete = (job['folder'] / marker).is_file()
            name = ('evaluation_complete' if job['config']['job_type'] == 'test' else 'training_complete_no_test') if complete else 'paused'
            row['status'] = {**row.get('status', {}), 'stage': name}
            if complete and job['config']['job_type'] != 'test': row['status']['epoch'] = job['config']['epochs']
            if skipped: row['context'] += ' · 저장된 완료 결과 확인'
        row['throughput'] = None

    def refresh(self, note=None, persist=True):
        if note is not None: self.note = note
        now = time.monotonic()
        if now - self.last_gpu >= 5:
            self.last_gpu = now
            try:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL, timeout=2)
                self.gpu_stats = {}
                for line in raw.splitlines():
                    gpu, used, total, utilization = map(float, line.split(','))
                    self.gpu_stats[int(gpu)] = {'used_gib': used / 1024, 'total_gib': total / 1024, 'utilization': utilization}
            except (OSError, ValueError, subprocess.SubprocessError): self.gpu_stats = {}
        value = render(self.phase, self.rows, self.note, self.data_rows, self.gpu_stats,
                       datetime.now(timezone(timedelta(hours=9))).strftime('%m/%d %H:%M:%S KST'))
        # Snapshot file is supplementary; a presentation failure must not interrupt training.
        if persist:
            try: (self.root / 'live_dashboard.html').write_text(value, encoding='utf-8')
            except OSError: pass
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
            if get_ipython() is None: raise ImportError('plain Python')
            if self.handle is None:
                self.handle = display(HTML(value), display_id=True)
                if self.handle is None: raise ImportError('no display update handle')
            else: self.handle.update(HTML(value))
        except Exception:
            if now - self.last_plain >= 20:
                self.last_plain = now
                print(self.note + ' | ' + ' | '.join(f'GPU{g} {r.get("label", "대기")}: {STAGES.get(r.get("status", {}).get("stage", "idle"), "대기")}' for g, r in self.rows.items()), flush=True)
