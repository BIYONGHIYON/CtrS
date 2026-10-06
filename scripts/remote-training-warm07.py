"""Disconnect-safe RGB07 weights-only warm control; separate run ownership."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

spec = importlib.util.spec_from_file_location('ctrs_detail_controller', Path(__file__).with_name('remote-training.py'))
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
controller.BASE = controller.ROOT / 'SSA-MRN/experiments/logs/remote-control-warm07'
controller.RUNS = controller.ROOT / 'SSA-MRN/experiments/checkpoints/remote-runs-warm07'
controller.CONFIG = controller.ROOT / 'SSA-MRN/configs/lib_rgb_hsi_joint17_warm10.json'
original_prepare = controller.prepare


def prepare(request):
    cfg = controller.read(Path(request['config']))
    if (cfg.get('model_type') != 'rgb_triple_joint17_23tap' or
            bool(cfg.get('pilot_options', {})) or
            cfg.get('pilot_label') or cfg.get('pilot_train_scenes') is not None):
        raise RuntimeError('RGB07 control requires the original joint17 model and all scenes')
    source = Path(cfg['init_checkpoint']) if cfg.get('init_checkpoint') else None
    full = False
    warm = cfg.get('epochs') == 10 and cfg.get('learning_rate') == 1e-5
    if not (full or warm) or (request['mode'] == 'start' and warm and source is None):
        raise RuntimeError('Use full100 at 1e-4, or weights-only fine-tune10 at 1e-5')
    if source and not source.is_file():
        raise RuntimeError('Initialization checkpoint missing: ' + str(source))
    meta = original_prepare(request)
    snapshot = Path(meta['log']).parent / 'config.json'
    cfg = controller.read(snapshot)
    if source and request['mode'] == 'start':
        frozen = Path(meta['output']) / 'init-source.pt'
        shutil.copy2(source, frozen)
        cfg['init_checkpoint'] = str(frozen)
        meta['initialization_source'] = str(source)
        meta['initialization_sha256'] = hashlib.sha256(frozen.read_bytes()).hexdigest()
    if request['mode'].startswith('resume-'):
        import torch
        previous = Path(request['checkpoint']).parent
        state = torch.load(Path(meta['output']) / 'resume-source.pt', map_location='cpu', weights_only=True)
        history = previous / 'history.jsonl'
        if history.exists():
            rows = [line for line in history.read_text(encoding='utf-8').splitlines()
                    if line.strip() and json.loads(line)['epoch'] <= state['epoch']]
            (Path(meta['output']) / 'history.jsonl').write_text('\n'.join(rows) + '\n', encoding='utf-8')
        metrics = previous / 'best_validation_metrics.json'
        if metrics.exists() and (Path(meta['output']) / 'best.pt').exists():
            shutil.copy2(metrics, Path(meta['output']) / metrics.name)
    controller.write(snapshot, cfg)
    controller.write(Path(meta['log']).parent / 'run.json', meta)
    return meta


def busy_processes():
    command = r"""$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(train_lib|train|diagnose_rgb_pilot)\.py' } | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"""
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError('Cannot verify shared GPU jobs: ' + result.stderr.strip())
    return json.loads(result.stdout) if result.stdout.strip() else []


controller.prepare = prepare
controller.legacy_processes = busy_processes
if __name__ == '__main__':
    controller.main()
