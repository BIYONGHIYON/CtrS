"""Upload an experiment's resume and initial weights (latest.pt, initial.pt) to its GitHub Release and record them in manifest.json.

Git keeps best.pt only; latest.pt and initial.pt are ignored by .gitignore and kept as Release assets:
  release   weights-<experiments|research>-<track>-<number>, pointing at the empty weights-anchor commit
  asset     the path inside the experiment folder with '/' replaced by '__'
  manifest  release_assets[path] = release, asset, url, bytes, sha256 (the entry is removed from files)
Existing assets with the same name are kept and must have the same sha256. Every upload is checked against the sha256
GitHub reports for the asset. Requires the GitHub CLI (gh) logged in with write access.

python SSA-MRN/common/release_weights.py SSA-MRN/experiments/performance/54_multi_sensor_scale_mixed_training
python SSA-MRN/common/release_weights.py <folder> --dry-run
"""
import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

REPO = 'BIYONGHIYON/CtrS'
ANCHOR = 'weights-anchor'
URL = 'https://github.com/{}/releases/download/{}/{}'
WEIGHTS = re.compile(r'(^|/)([^/]*latest\.pt|initial\.pt)$')
FOLDER = re.compile(r'(?:^|/)SSA-MRN/(experiments|research)/([^/]+)/(\d+)_[^/]+$')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def gh(*args):
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def assets(tag):
    return {a['name']: a for a in json.loads(gh('api', f'repos/{REPO}/releases/tags/{tag}'))['assets']}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', type=Path, help='experiment folder, e.g. SSA-MRN/experiments/performance/54_...')
    p.add_argument('--dry-run', action='store_true', help='list what would be uploaded without changing anything')
    a = p.parse_args()
    folder = a.folder.resolve(); m = FOLDER.search(folder.as_posix())
    if not m: p.error('not an experiment folder: SSA-MRN/<experiments|research>/<track>/<number>_<name>')
    tag = 'weights-{}-{}-{}'.format(*m.groups())
    files = sorted(f for f in folder.rglob('*.pt') if f.is_file() and WEIGHTS.search(f.relative_to(folder).as_posix()))
    if not files: print('no latest.pt / initial.pt in', folder); return
    local = {f.relative_to(folder).as_posix(): dict(file=f, bytes=f.stat().st_size, sha256=digest(f)) for f in files}
    for rel, v in local.items(): print(f"{rel}  {v['bytes']:,} bytes  {v['sha256'][:12]}")
    if a.dry_run: print('release', tag, '(dry run, nothing uploaded)'); return
    if not json.loads(gh('release', 'list', '-R', REPO, '--limit', '1000', '--json', 'tagName', '-q', f'[.[]|select(.tagName=="{tag}")]')):
        anchor = json.loads(gh('api', f'repos/{REPO}/git/ref/tags/{ANCHOR}'))['object']['sha']
        gh('release', 'create', tag, '-R', REPO, '--target', anchor, '--title', tag,
           '--notes', f'`{folder.relative_to(folder.parents[3]).as_posix()}`의 재개용(latest)·초기(initial) 가중치입니다. 경로·크기·SHA-256은 해당 실험의 manifest.json `release_assets`에 있습니다.')
    have = assets(tag)
    with tempfile.TemporaryDirectory() as stage:
        todo = []
        for rel, v in local.items():
            name = rel.replace('/', '__')
            if name in have:
                if have[name]['digest'] != 'sha256:' + v['sha256']: raise SystemExit(f'{tag}/{name} already exists with different content')
                continue
            link = Path(stage) / name; link.symlink_to(v['file']); todo.append(str(link))
        for i in range(0, len(todo), 10): gh('release', 'upload', tag, '-R', REPO, *todo[i:i + 10])
    have = assets(tag)
    for rel, v in local.items():
        got = have.get(rel.replace('/', '__'))
        if not got or got['size'] != v['bytes'] or got['digest'] != 'sha256:' + v['sha256']: raise SystemExit(f'upload check failed: {rel}')
    mpath = folder / 'manifest.json'
    manifest = json.loads(mpath.read_text(encoding='utf-8')) if mpath.exists() else {'files': {}}
    manifest['files'] = {k: e for k, e in manifest.get('files', {}).items() if k not in local}
    manifest.setdefault('release_assets', {}).update({rel: dict(release=tag, asset=rel.replace('/', '__'), url=URL.format(REPO, tag, rel.replace('/', '__')),
                                                                 bytes=v['bytes'], sha256=v['sha256']) for rel, v in sorted(local.items())})
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(local)} assets in release {tag} match GitHub sha256; manifest release_assets updated')


if __name__ == '__main__':
    main()
