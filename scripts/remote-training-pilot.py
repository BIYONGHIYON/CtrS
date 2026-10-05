"""Independent disconnect-safe controller for the six-stage pilot suite."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess

script = Path(__file__).with_name('remote-training.py')
spec = importlib.util.spec_from_file_location('ctrs_pilot_controller', script)
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
controller.BASE = controller.ROOT / 'SSA-MRN/experiments/logs/remote-control-pilot'
controller.RUNS = controller.ROOT / 'SSA-MRN/experiments/checkpoints/remote-runs-pilot'
controller.CONFIG = controller.ROOT / 'SSA-MRN/configs/pilot/00_baseline10.json'
original_prepare = controller.prepare


def prepare(request):
    cfg = controller.read(Path(request['config']))
    task = cfg.get('pilot_task', 'train')
    if task not in ('train', 'diagnostic'):
        raise RuntimeError('Unknown pilot task')
    if cfg.get('eval_layout') != 'tiles' or cfg.get('degradation') != 'area':
        raise RuntimeError('Pilot suite requires the fixed tiles/area protocol')
    if task == 'diagnostic' and request['mode'] != 'start':
        raise RuntimeError('Diagnostics must use start, not resume')
    # Preflight before creating run files.
    key = 'diagnostic_checkpoint' if task == 'diagnostic' else 'init_checkpoint'
    source = Path(cfg[key]) if cfg.get(key) else None
    if source and not source.is_file():
        raise RuntimeError('Initialization checkpoint missing: ' + str(source))
    meta = original_prepare(request)
    snapshot = Path(meta['log']).parent / 'config.json'
    cfg = controller.read(snapshot)
    if request['mode'].startswith('resume-'):
        import json
        import torch
        previous = Path(request['checkpoint']).parent
        state = torch.load(Path(meta['output']) / 'resume-source.pt', map_location='cpu', weights_only=True)
        history = previous / 'history.jsonl'
        if history.exists():
            # Best-resume must not inherit records from epochs after the chosen weight.
            lines = [line for line in history.read_text(encoding='utf-8').splitlines()
                     if line.strip() and json.loads(line)['epoch'] <= state['epoch']]
            (Path(meta['output']) / 'history.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        metrics = previous / 'best_validation_metrics.json'
        if metrics.exists() and (Path(meta['output']) / 'best.pt').exists():
            shutil.copy2(metrics, Path(meta['output']) / metrics.name)
        del state
    if source and request['mode'] == 'start':
        frozen = Path(meta['output']) / ('diagnostic-source.pt' if task == 'diagnostic' else 'init-source.pt')
        shutil.copy2(source, frozen)
        cfg[key] = str(frozen)
        meta['initialization_source'] = str(source)
        meta['initialization_sha256'] = hashlib.sha256(frozen.read_bytes()).hexdigest()
    if task == 'diagnostic':
        meta['command'] = [str(controller.PYTHON), '-u',
            str(controller.ROOT / 'SSA-MRN/scripts/diagnose_rgb_pilot.py'),
            '--config', str(snapshot), '--output-dir', meta['output']]
    meta['pilot_label'] = cfg['pilot_label']
    meta['task'] = task
    controller.write(snapshot, cfg)
    controller.write(Path(meta['log']).parent / 'run.json', meta)
    return meta


def busy_processes():
    # Detect other SSA-MRN, ECRformer and diagnostic jobs; never stop them.
    command = r"""$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(train_lib|train|diagnose_rgb_pilot)\.py' } | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"""
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                            capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise RuntimeError('Cannot verify shared GPU jobs: ' + result.stderr.strip())
    import json
    return json.loads(result.stdout) if result.stdout.strip() else []


controller.prepare = prepare
controller.legacy_processes = busy_processes
if __name__ == '__main__':
    controller.main()
