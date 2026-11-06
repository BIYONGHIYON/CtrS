"""Experiment 77: check the extracted source against the local manifest before any run."""
import hashlib, json, sys
from pathlib import Path
root = Path('C:/CtrS-p77'); man = json.loads((root / 'ctrs77-source-manifest.json').read_text(encoding='utf-8'))
bad = [f for f, h in man['files'].items() if not (root / f).is_file() or hashlib.sha256((root / f).read_bytes()).hexdigest() != h]
(root / 'source_verification.json').write_text(json.dumps(dict(commit=man['commit'], files=len(man['files']), mismatched=bad), indent=1), encoding='utf-8')
print('SOURCE', man['commit'], len(man['files']), 'files, mismatched', len(bad)); sys.exit(1 if bad else 0)
