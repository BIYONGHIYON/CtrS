"""Prepare the single-study Pages folder from preserved offline band viewers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
STUDY = 'rgb11_gradient_suite'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Dedicated Pages checkout or staging directory')
    args = parser.parse_args()
    source = ROOT / 'SSA-MRN/docs/assets' / STUDY / 'band_viewer'
    target = args.output / 'ssa-mrn'
    target.mkdir(parents=True, exist_ok=True)
    rows = json.loads((source / 'manifest.json').read_text())
    assert len(rows) == 5 and all(r['bands_count'] == 204 for r in rows)
    hashes = {r['checkpoint_sha256'] for r in rows}
    assert len(hashes) == 1
    for name in ['index.html', 'manifest.json'] + [r['file'] for r in rows]:
        shutil.copy2(source / name, target / name)
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    metadata = {
        'repository': 'BIYONGHIYON/CtrS', 'source_commit': source_commit,
        'model': STUDY, 'weight_epoch': rows[0]['epoch'],
        'checkpoint_sha256': next(iter(hashes)),
        'selection': 'Validation MSE within gradient suite; latest representative by PSNR and SAM among matching tile evaluations. RGB09 has lower mean MSE.',
        'evaluation': 'LIB-HSI synthetic area x4; 75 test scenes; 300 tiles; LR consistency',
        'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.iterdir()) if p.is_file()},
    }
    (args.output / 'source.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(f'Prepared {STUDY}: 5 scenes, 204 bands at {target}')

if __name__ == '__main__':
    main()
