"""Pack viewer/data/ into one archive, upload it to the viewer-data GitHub Release and pin it in data.sha256.

The prediction arrays (about 250MB) stay out of Git. The Pages workflow (.github/workflows/pages.yml) downloads the
asset named in data.sha256, checks its SHA-256 and deploys it with index.html and app.js. The archive is reproducible
(sorted entries, fixed time and owner), so the same data always gives the same name and hash. Each data version keeps
its own asset (viewer-data-<sha12>.tar.gz); existing assets are never replaced. Requires gh logged in with write access.

python SSA-MRN/viewer/export.py                 # make viewer/data/
python SSA-MRN/viewer/release_data.py           # upload and write data.sha256, then commit data.sha256 and open a PR
python SSA-MRN/viewer/release_data.py --dry-run
"""
import argparse
import gzip
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

REPO = 'BIYONGHIYON/CtrS'
TAG = 'viewer-data'
ANCHOR = 'weights-anchor'
HERE = Path(__file__).resolve().parent


def gh(*args):
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def pack(data, out):
    with open(out, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as tar:
        for f in sorted(p for p in data.rglob('*') if p.is_file()):
            info = tarfile.TarInfo(f.relative_to(data).as_posix())
            info.size, info.mode, info.mtime = f.stat().st_size, 0o644, 0
            with open(f, 'rb') as src: tar.addfile(info, src)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=HERE / 'data')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not (a.data / 'index.json').is_file(): raise SystemExit(f'{a.data}/index.json not found; run export.py first')
    with tempfile.TemporaryDirectory() as stage:
        tmp = Path(stage) / 'data.tar.gz'
        pack(a.data, tmp)
        sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
        name = f'{TAG}-{sha[:12]}.tar.gz'
        size = tmp.stat().st_size
        print(f'{name}  {size:,} bytes  {sum(1 for p in a.data.rglob("*") if p.is_file())} files  sha256 {sha}')
        if a.dry_run: print('dry run, nothing uploaded'); return
        if not json.loads(gh('release', 'list', '-R', REPO, '--limit', '1000', '--json', 'tagName', '-q', f'[.[]|select(.tagName=="{TAG}")]')):
            anchor = json.loads(gh('api', f'repos/{REPO}/git/ref/tags/{ANCHOR}'))['object']['sha']
            gh('release', 'create', TAG, '-R', REPO, '--target', anchor, '--title', TAG, '--latest=false',
               '--notes', '결과 비교 사이트(`SSA-MRN/viewer`)의 예측 데이터입니다. 배포에 쓰는 자산 이름과 SHA-256은 `SSA-MRN/viewer/data.sha256`에 있습니다.')
        have = {x['name']: x for x in json.loads(gh('api', f'repos/{REPO}/releases/tags/{TAG}'))['assets']}
        if name in have:
            print('already uploaded')
        else:
            final = Path(stage) / name
            tmp.rename(final)
            gh('release', 'upload', TAG, '-R', REPO, str(final))
            have = {x['name']: x for x in json.loads(gh('api', f'repos/{REPO}/releases/tags/{TAG}'))['assets']}
        got = have.get(name)
        if not got or got['size'] != size or got['digest'] != 'sha256:' + sha: raise SystemExit(f'upload check failed: {name}')
    (HERE / 'data.sha256').write_text(f'{sha}  {name}\n')
    print('verified on GitHub; wrote', (HERE / 'data.sha256').relative_to(HERE.parents[1]))


if __name__ == '__main__':
    main()
