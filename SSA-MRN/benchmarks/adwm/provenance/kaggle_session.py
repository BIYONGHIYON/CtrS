# ADWM Kaggle session. Paste into one notebook cell, set MODE below, then "Save Version" > "Save & Run All (Commit)".
# Notebook settings: accelerator GPU T4 x2, internet on, input dataset "CtrS PAN Sharpening Dataset".
#   MODE = 'measure' : first short session. Batch-64 GPU memory and seconds per iteration for QB (GPU 0) and GF2 (GPU 1),
#                      with and without layer recomputation, then the WV3 validation choice of the author LayerNorm handling.
#                      Outputs: /kaggle/working/adwm/measure_QB.json, measure_GF2.json, author_WV3/layernorm_selection.json
#   MODE = 'train'   : QB on GPU 0 and GF2 on GPU 1 at the same time, each stopping cleanly before DEADLINE_HOURS.
#                      To continue, add the previous version's output as an input ("Add Input" > Your Work > this notebook);
#                      the script copies adwm/QB and adwm/GF2 from it and resumes from latest.pt with the same code commit.
# No API token is used. Outputs stay in /kaggle/working/adwm/ (download history.json, status.json, best.pt from Output).
import json, os, shutil, subprocess, sys, time
from pathlib import Path

MODE = 'measure'            # 'measure' or 'train'
DEADLINE_HOURS = 5.5        # training time per session (setup and data loading come on top)
RECOMPUTE = False           # set True only if measure_*.json shows that batch 64 does not fit without it
CODE_COMMIT = None          # None on the first session = newest commit of branch adwm; resumed runs reuse the recorded one
SENSORS = ('QB', 'GF2')

REPO, BRANCH = 'https://github.com/BIYONGHIYON/CtrS.git', 'adwm'
CODE = Path('/kaggle/working/CtrS'); OUT = Path('/kaggle/working/adwm'); INPUT = Path('/kaggle/input')
FOLDER = 'SSA-MRN/benchmarks/adwm'


def sh(*cmd, **kw):
    print('$', ' '.join(map(str, cmd)), flush=True); return subprocess.run(list(map(str, cmd)), check=True, **kw)


def previous_run(sensor):
    hits = [p.parent for p in INPUT.rglob('latest.pt') if p.parent.name == sensor and 'adwm' in p.parts]
    if len(hits) > 1: raise SystemExit(f'several earlier {sensor} runs in the inputs: {hits}')
    return hits[0] if hits else None


# 1. recorded code commit of an earlier session, if any
recorded = set()
for s in SENSORS:
    prev = previous_run(s)
    if prev:
        import torch
        recorded.add(torch.load(prev / 'latest.pt', map_location='cpu', weights_only=False)['config']['code_commit'])
if len(recorded) > 1: raise SystemExit(f'QB and GF2 were trained with different code commits: {recorded}')
commit = CODE_COMMIT or (recorded.pop() if recorded else None)

# 2. code: sparse checkout of the ADWM folder and its official submodule only (the full repository is large)
if CODE.exists(): shutil.rmtree(CODE)
sh('git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse', '-b', BRANCH, REPO, CODE)
if commit:
    sh('git', '-C', CODE, 'fetch', '--depth', '1', 'origin', commit); sh('git', '-C', CODE, 'checkout', '-q', commit)
sh('git', '-C', CODE, 'sparse-checkout', 'set', FOLDER)
sh('git', '-C', CODE, 'submodule', 'update', '--init', '--depth', '1', f'{FOLDER}/upstream')
commit = subprocess.run(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
print('code commit', commit, flush=True)
try: import einops  # noqa: F401
except ImportError: sh(sys.executable, '-m', 'pip', 'install', '-q', 'einops')
TRAIN = CODE / FOLDER / 'train.py'
OUT.mkdir(parents=True, exist_ok=True)


def launch(sensor, gpu, args, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    return subprocess.Popen([sys.executable, str(TRAIN), *map(str, args)], env=env, stdout=open(log, 'w'), stderr=subprocess.STDOUT)


def wait(procs):
    while any(p.poll() is None for p, _ in procs):
        time.sleep(300)
        for p, log in procs: print(log.name, '|', (log.read_text().strip().splitlines() or [''])[-1], flush=True)
    for p, log in procs:
        print(f'--- {log.name} (exit {p.returncode})\n' + '\n'.join(log.read_text().splitlines()[-15:]), flush=True)


if MODE == 'measure':
    wait([(launch(s, g, ['measure', '--sensor', s, '--device', 'cuda:0', '--out', OUT], OUT / f'measure_{s}.log'), OUT / f'measure_{s}.log')
          for g, s in enumerate(SENSORS)])
    log = OUT / 'author_select.log'
    wait([(launch('WV3', 0, ['author-select', '--data', INPUT, '--device', 'cuda:0', '--out', OUT / 'author_WV3'], log), log)])
elif MODE == 'train':
    procs = []
    for g, s in enumerate(SENSORS):
        run = OUT / s; prev = previous_run(s); args = ['train', '--sensor', s, '--data', INPUT, '--out', run, '--device', 'cuda:0',
                                                      '--deadline-hours', DEADLINE_HOURS, '--code-commit', commit]
        if prev:
            if run.exists(): shutil.rmtree(run)
            shutil.copytree(prev, run); args += ['--resume', run / 'latest.pt']
        if RECOMPUTE: args.append('--recompute')
        log = OUT / f'train_{s}_{time.strftime("%Y%m%d_%H%M")}.log'
        procs.append((launch(s, g, args, log), log))
    wait(procs)
    for s in SENSORS:
        status = OUT / s / 'status.json'
        print(s, status.read_text() if status.exists() else 'no status', flush=True)
else:
    raise SystemExit(f'unknown MODE {MODE!r}')
