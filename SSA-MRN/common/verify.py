"""Verify current retained file hashes; this does not recompute test metrics.

Files kept in the repository are listed under manifest 'files'. Resume and initial weights (latest.pt, initial.pt) are kept
as GitHub Release assets and listed under 'release_assets' with their release URL; --releases downloads and checks them.
"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def remote_digest(url):
    h=hashlib.sha256();size=0
    with urllib.request.urlopen(url) as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block);size+=len(block)
    return size,h.hexdigest()

def verify(folder,releases=False):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if isinstance(manifest['files'],list):manifest['files']={e['path']:dict(bytes=e['size'],sha256=e['sha256']) for e in manifest['files']}
    for name,expected in manifest['files'].items():
        path=folder/name
        if not path.is_file() or path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']:
            raise ValueError(f'File mismatch: {path}')
    assets=manifest.get('release_assets',{})
    if releases:
        for name,expected in assets.items():
            if remote_digest(expected['url'])!=(expected['bytes'],expected['sha256']):raise ValueError(f'Release asset mismatch: {folder/name}')
    return len(manifest['files']),sum(name.endswith('.pt') for name in manifest['files']),len(assets)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',default='all',help='Number, numbered folder name, or all')
    p.add_argument('--research-track', choices=('ssai','variable_scale','performance'), help='Disambiguate independent research numbering')
    p.add_argument('--artifact-root',type=Path,help='Verify a new export report_manifest.json')
    p.add_argument('--releases',action='store_true',help='Also download and hash the release assets (latest.pt, initial.pt)')
    a=p.parse_args()
    if a.artifact_root:
        data=json.loads((a.artifact_root/'report_manifest.json').read_text(encoding='utf-8'))
        files=data.get('artifacts',data.get('files'))
        if not files:raise ValueError('No export files')
        for name,e in files.items():
            path=a.artifact_root/name
            if path.stat().st_size!=e['bytes'] or digest(path)!=e['sha256']:raise ValueError(name)
        print(f'PASS: {len(files)} export hashes');return
    folders=sorted(list((ROOT/'research').glob('*/[0-9]*_*'))+list((ROOT/'experiments').glob('*/[0-9]*_*')),key=lambda x:(x.parent.name,int(x.name.split('_')[0])))
    if a.research_track:folders=[f for f in folders if f.parent.name==a.research_track]
    if a.experiment!='all':folders=[f for f in folders if a.experiment in (f.name,f.name.split('_',1)[0],f.name.split('_',1)[1])]
    if not folders:p.error('Unknown experiment')
    if a.experiment!='all' and a.experiment.isdigit() and len(folders)>1:p.error('Ambiguous number: specify --research-track')
    total=weights=released=0
    for folder in folders:
        count,pts,assets=verify(folder,a.releases);total+=count;weights+=pts;released+=assets
    print(f'PASS: {len(folders)} experiments, {total} files, {weights} checkpoints; release assets {released} ({"hashes checked" if a.releases else "listed, use --releases to check"}); hashes only')
if __name__=='__main__':main()
