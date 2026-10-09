"""Build a standalone cell from the preserved MTF trainer, without editing it."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'SSA-MRN/scripts'


def build():
    original = (SCRIPTS / 'kaggle_mtf_pair_cell.py').read_text()
    tree = ast.parse(original)
    assignments = {n.targets[0].id: n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    worker = ast.literal_eval(assignments['WORKER_SOURCE'].value)
    # Remove MTF-only functions; no augmentation enters this comparison.
    wt = ast.parse(worker)
    lines = worker.splitlines(keepends=True)
    for node in reversed(wt.body):
        if isinstance(node, ast.FunctionDef) and node.name in ('bank_choice', 'kernels_for_factor', 'observe', 'make_delta', 'build_banks'):
            start = min([node.lineno]+[x.lineno for x in node.decorator_list])-1
            del lines[start:node.end_lineno]
    worker = ''.join(lines)
    # Strip unreachable augmentation branches as well as their helpers.
    wt = ast.parse(worker)
    ranges = []
    for node in (n for c in wt.body if isinstance(c,ast.ClassDef) for n in ast.walk(c)):
        if isinstance(node, ast.If) and 'mtf_aug' in ast.unparse(node.test):
            ranges.append((node.lineno-1,node.end_lineno))
        elif isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bank' for t in node.targets):
            ranges.append((node.lineno-1,node.end_lineno))
        elif isinstance(node, ast.If) and ast.unparse(node.test)=='bank':
            ranges.append((node.lineno-1,node.end_lineno))
    lines = worker.splitlines(keepends=True)
    for start,end in sorted(ranges,reverse=True):
        del lines[start:end]
    worker = ''.join(lines)
    worker = worker.replace('import numpy as np', 'import random\nimport numpy as np')
    worker = worker.replace("    if variant == 'mtf_aug':\n        profile", "    if False:\n        profile")
    a = worker.index('    if False:\n        profile')
    b = worker.index('    tr = CachedPairs', a)
    worker = worker[:a]+worker[b:]
    worker = worker.replace('from ssamrn.models.ssa_mrn import RestoredPansharpeningNet', 'from ssamrn.models.ssa_fusion import FusionPansharpeningNet')
    worker = worker.replace("RestoredPansharpeningNet(4, config['k'])", "FusionPansharpeningNet(4, config['k'], variant)")
    worker = worker.replace("choices=['baseline','mtf_aug']", "choices=['A0','A1','A2','A3']")
    worker = worker.replace("torch.manual_seed(config['seed'])", "random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])")
    worker = worker.replace("torch.set_rng_state(ck['rng'])", "torch.set_rng_state(ck['rng']); random.setstate(ck['python_rng']); np.random.set_state(('MT19937', np.asarray(ck['numpy_rng'][1], dtype=np.uint32), *ck['numpy_rng'][2:]))")
    worker = worker.replace("rng=torch.get_rng_state(),", "python_rng=random.getstate(), numpy_rng=(np.random.get_state()[0], np.random.get_state()[1].tolist(), *np.random.get_state()[2:]), rng=torch.get_rng_state(),")
    worker = worker.replace("variant='baseline', training_variant=variant", "variant=variant, training_variant=variant")
    worker = worker.replace("provenance = dict(config=config,", "provenance = dict(parameters_total=sum(p.numel() for p in model.parameters()), parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad), config=config,")
    # Replace snapshots from end to beginning, preserving all unrelated source files.
    sources = {p: (ROOT/p).read_text() for p in ('SSA-MRN/src/ssamrn/models/ssa_mrn.py', 'SSA-MRN/src/ssamrn/models/ssa_fusion.py', 'SSA-MRN/references/upstream/network.py')}
    text = original
    for key, value in sorted([('WORKER_SOURCE',worker), ('EMBEDDED_SOURCES',sources)], key=lambda pair: assignments[pair[0]].lineno, reverse=True):
        n = assignments[key]
        ls = text.splitlines(keepends=True)
        ls[n.lineno-1:n.end_lineno] = [key+' = '+repr(value)+'\n']
        text = ''.join(ls)
    text = text.replace("RUN_TAG = 'mtf_pair'", "RUN_TAG = 'ssa_fusion'")
    text = text.replace("import os, sys, json", "PAIR = ('A0', 'A1')            # second round: ('A2', 'A3')\nSTART_TRAINING = False         # set True only after the user requests new training\nassert PAIR in (('A0','A1'), ('A2','A3'))\nif not START_TRAINING: raise RuntimeError('실행 준비 완료. 신규 학습 요청 후 START_TRAINING=True로 변경하세요.')\n\nimport os, sys, json",1)
    text = text.replace("f'{RUN_TAG}_qb_k{K}_s{SEED}'", "f'{RUN_TAG}_{PAIR[0]}_{PAIR[1]}_qb_k{K}_s{SEED}'")
    text = text.replace("('baseline','mtf_aug')", 'PAIR')
    text = text.replace("protocol='qb_mtf_interior_delta_v1'", "protocol='qb_ssa_fusion_v1', pair=list(PAIR), window=5, chunk_rows=32")
    text = text.replace("smoke=SMOKE, mtf_factors=[.9,1.1]", "smoke=SMOKE, augmentation='none'")
    start = text.index("profile = json.loads(EMBEDDED_SOURCES['qb_profile.json'])")
    end = text.index('conditions = dict',start)
    text = text[:start]+text[end:]
    start = text.index('# Only this reviewed original worker')
    end = text.index("CACHE = Path('/tmp')",start)
    text = text[:start]+"def compatible(previous):\n    return {k: previous.get(k) for k in conditions} == conditions\naccepted_fingerprints = [fingerprint]\n\n"+text[end:]
    text = text.replace("    required += 2*(np.prod(data_info['train']['shapes']['lms'])+np.prod(data_info['train']['shapes']['ms']))*4\n", '')
    text = text.replace('pair_config.json', 'fusion_config.json')
    text = text.replace('# Kaggle 한 셀: GPU 0 = 원본 기준선 / GPU 1 = MTF 증강', '# Kaggle 한 셀: A0/A1 또는 A2/A3 · MTF 없이 SSA 연산만 비교')
    # Refuse any live compute PID, including a low-memory/idle existing training process.
    needle = "    used = [int(x)"
    pos = text.index(needle)
    text = text[:pos]+"    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()\n    if active: raise RuntimeError('GPU 프로세스가 남아 있습니다. 기존 MTF 학습을 유지하고 별도 세션을 사용하세요.')\n"+text[pos:]
    # Verify imported Output files before deserializing its checkpoints.
    needle = '    for variant in PAIR:\n        source = previous/variant'
    text = text.replace(needle, "    manifest = json.loads((previous/'artifact_manifest.json').read_text())\n    for relative, info in manifest.items():\n        path = previous/relative\n        if not path.resolve().is_relative_to(previous.resolve()): raise ValueError('잘못된 manifest 경로')\n        if not path.is_file() or path.stat().st_size != info['bytes'] or file_hash(path) != info['sha256']: raise ValueError('Output 복구 해시 불일치: '+relative)\n"+needle)
    text = text.replace("    return choices[0]", "    return choices[0]")
    ast.parse(text); ast.parse(worker)
    (SCRIPTS/'kaggle_ssa_fusion_cell.py').write_text(text)
    notebook = dict(cells=[dict(id='ssa-fusion-cell', cell_type='code', execution_count=None, metadata={}, outputs=[], source=text.splitlines(keepends=True))], metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')), nbformat=4, nbformat_minor=5)
    (SCRIPTS/'kaggle_ssa_fusion.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    print('Built identical single cell:', hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    build()
