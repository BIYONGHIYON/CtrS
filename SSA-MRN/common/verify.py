"""Verify current retained file hashes; this does not recompute test metrics."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def verify(folder):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        path=folder/name
        if not path.is_file() or path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']:
            raise ValueError(f'File mismatch: {path}')
    return len(manifest['files']),sum(name.endswith('.pt') for name in manifest['files'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',default='all',help='Number, numbered folder name, or all')
    p.add_argument('--artifact-root',type=Path,help='Verify a new export report_manifest.json')
    a=p.parse_args()
    if a.artifact_root:
        data=json.loads((a.artifact_root/'report_manifest.json').read_text(encoding='utf-8'))
        files=data.get('artifacts',data.get('files'))
        if not files:raise ValueError('No export files')
        for name,e in files.items():
            path=a.artifact_root/name
            if path.stat().st_size!=e['bytes'] or digest(path)!=e['sha256']:raise ValueError(name)
        print(f'PASS: {len(files)} export hashes');return
    folders=sorted((ROOT/'research').glob('*/[0-9]*_*'),key=lambda x:int(x.name.split('_')[0]))
    if a.experiment!='all':folders=[f for f in folders if a.experiment in (f.name,f.name.split('_',1)[0],f.name.split('_',1)[1])]
    if not folders:p.error('Unknown experiment')
    total=weights=0
    for folder in folders:
        count,pts=verify(folder);total+=count;weights+=pts
    print(f'PASS: {len(folders)} experiments, {total} files, {weights} checkpoints; hashes only')
if __name__=='__main__':main()
