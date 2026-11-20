"""Inputs, restoring and checks for pansharpening with generative image models (issues #209-212, zero-shot, RGB only).

Generative image models (GPT Image 2.6 Sunburst and Nano Banana 2.1 of the Dongguk AI CHAT API Gateway) take 8-bit pictures and return one
8-bit RGB picture, so they are tested on the RGB bands only (QB/GF2 bands [2,1,0], WV3/WV2 [4,2,1]) and scored with the RGB
mode of the MATLAB workflow (benchmark_prepare.py --bands rgb). The ground truth is never used to build inputs or to stretch.

inputs    RR test H5 -> per scene two 1024x1024 PNGs and meta.json, zipped per sensor (genimage_inputs_<SENSOR>.zip)
            image 1 (colour): the dataset's own LMS, RGB bands, per-band 0.1-99.9% stretch of that LMS, 8 bit, bicubic x4
            image 2 (detail): PAN, 0.1-99.9% stretch, 8 bit, bicubic x4 (grey)
restore   20 returned pictures -> 256x256 by area averaging -> DN by the inverse of image 1's stretch -> output_mulExm_<i>.mat
            (key sr, H x W x 3, DN, bands in R,G,B order) zipped as <model>_zeroshot_<SENSOR>_rr.zip, plus the untouched
            pictures as <model>_zeroshot_<SENSOR>_raw.zip and a record of sizes, hashes and stretch ranges
identity  the conversion-limit control: image 1 itself sent through restore (interpolation + 8-bit round trip, no model)
checks    per-scene checks that need no ground truth, for the returned pictures and for the ground truth as a reference:
            observation error  RMSE (DN) between the RGB bands of the input MS and the sensor MTF x4 decimation of the output
            PAN detail corr.   correlation of the high-pass of the output intensity (RGB mean) and of the PAN
            shift              phase-correlation shift (rows, cols) of the output intensity against the PAN, in pixels
upload    put files on a GitHub Release (created at weights-anchor if missing); an existing asset is never replaced
generate  make the pictures through the Dongguk AI CHAT (BAZE) API Gateway with the fixed prompts and settings of #209, so
          nobody has to click 60 times: reads the key from the BAZE_API_KEY environment variable (never stored or printed),
          estimates the cost first, stops before --max-credits would be passed, saves <model>_<SENSOR>_<scene>.<ext> and a
          log of every attempt (time, size, credits charged, error), and skips scenes that are already saved (resume)

python SSA-MRN/common/genimage_io.py inputs --out runtime/genimage
python SSA-MRN/common/genimage_io.py restore --sensor QB --model nano_banana_2_1 --images <folder> --out runtime/genimage
python SSA-MRN/common/genimage_io.py checks --sensor QB --predictions runtime/genimage/nano_banana_2_1_zeroshot_QB_rr.zip
python SSA-MRN/common/genimage_io.py upload --release genimage-inputs runtime/genimage/genimage_inputs_*.zip
python SSA-MRN/common/genimage_io.py generate --sensor QB --model nano_banana_2_1 --inputs genimage_inputs_QB.zip --out pictures --estimate-only
python SSA-MRN/common/genimage_io.py generate --sensor QB --model nano_banana_2_1 --inputs genimage_inputs_QB.zip --out pictures --max-credits 800

Returned pictures are named <anything>_<SENSOR>_<scene two digits>.<png|jpg|jpeg|webp>, e.g. nano_banana_2_1_QB_07.png.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parents[1]
REPO = 'BIYONGHIYON/CtrS'
ANCHOR = 'weights-anchor'
SENSORS = {'QB': ('test_qb_multiExm1.h5', [2, 1, 0]), 'GF2': ('test_gf2_multiExm1.h5', [2, 1, 0]),
           'WV3': ('test_wv3_multiExm1.h5', [4, 2, 1]), 'WV2': ('test_wv2_multiExm1.h5', [4, 2, 1])}
SIZE, BIG, SCENES = 256, 1024, 20
LOW, HIGH = 0.1, 99.9          # stretch percentiles (image 1 from the LMS, image 2 from the PAN)
ZIP_TIME = (2026, 1, 1, 0, 0, 0)
PICTURE = re.compile(r'_(QB|GF2|WV3|WV2)_(\d{2})\.(png|jpe?g|webp)$', re.I)


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else Path(data).read_bytes()).hexdigest()


def rr_scenes(data, sensor):
    found = list(Path(data).rglob(SENSORS[sensor][0]))
    if len(found) != 1: raise FileNotFoundError(f'{SENSORS[sensor][0]} below {data}: {found}')
    with h5py.File(found[0], 'r') as f:
        scenes = [{k: np.asarray(f[k][i], dtype=np.float64) for k in ('gt', 'pan', 'lms', 'ms')} for i in range(len(f['gt']))]
    if len(scenes) != SCENES: raise ValueError(f'{found[0]}: {len(scenes)} scenes, expected {SCENES}')
    return scenes, dict(file=found[0].name, sha256=sha(found[0]))


def to8(x, lo, hi):
    return np.rint(np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)


def png(array):
    buf = io.BytesIO(); Image.fromarray(array).save(buf, format='PNG', optimize=False); return buf.getvalue()


def write_zip(path, files):
    """Deterministic zip (sorted names, fixed time) so that the same content always gives the same SHA-256."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, ZIP_TIME); info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            z.writestr(info, files[name])
    return path


def make_inputs(scenes, sensor):
    """{name: png bytes} and per-scene stretch ranges for one sensor."""
    bands = SENSORS[sensor][1]; files, meta = {}, []
    for i, s in enumerate(scenes):
        lms = s['lms'][bands]
        lo = np.percentile(lms, LOW, axis=(1, 2)); hi = np.percentile(lms, HIGH, axis=(1, 2))
        rgb = np.stack([to8(lms[k], lo[k], hi[k]) for k in range(3)], -1)
        plo, phi = np.percentile(s['pan'][0], [LOW, HIGH])
        pan = to8(s['pan'][0], plo, phi)
        one = np.asarray(Image.fromarray(rgb).resize((BIG, BIG), Image.Resampling.BICUBIC))
        two = np.asarray(Image.fromarray(pan).resize((BIG, BIG), Image.Resampling.BICUBIC).convert('RGB'))
        files[f'{sensor}_{i:02}_1_ms.png'] = png(one); files[f'{sensor}_{i:02}_2_pan.png'] = png(two)
        meta.append(dict(scene=i, ms_low=lo.tolist(), ms_high=hi.tolist(), pan_low=float(plo), pan_high=float(phi)))
    return files, meta


def cmd_inputs(a):
    out = Path(a.out)
    for sensor in a.sensors:
        scenes, data = rr_scenes(a.data, sensor)
        files, meta = make_inputs(scenes, sensor)
        record = dict(sensor=sensor, rgb_bands=SENSORS[sensor][1], data=data, size=BIG, source_size=SIZE, percentiles=[LOW, HIGH],
                      image1='LMS RGB bands, per-band stretch of the LMS itself, 8 bit, bicubic x4', image2='PAN stretch, 8 bit, bicubic x4, grey as RGB',
                      ground_truth_used=False, scenes=meta, files={n: sha(b) for n, b in files.items()})
        files['meta.json'] = (json.dumps(record, indent=1) + '\n').encode()
        path = write_zip(out / f'genimage_inputs_{sensor}.zip', files)
        print(f'{path}  {path.stat().st_size:,} bytes  sha256 {sha(path)}', flush=True)


def read_meta(inputs, sensor):
    p = Path(inputs)
    if p.is_dir(): p = p / f'genimage_inputs_{sensor}.zip'
    with zipfile.ZipFile(p) as z: meta = json.loads(z.read('meta.json'))
    if meta['sensor'] != sensor: raise ValueError(f'{p}: inputs of {meta["sensor"]}, not {sensor}')
    return meta, sha(p)


def to_256(picture):
    """Returned picture -> float RGB 256x256 (area average). Square pictures only."""
    img = Image.open(io.BytesIO(picture)); img.load()
    if img.width != img.height: raise ValueError(f'not square: {img.width}x{img.height}')
    x = np.asarray(img.convert('RGB'), dtype=np.float64)
    if img.width % SIZE == 0:
        f = img.width // SIZE
        return x.reshape(SIZE, f, SIZE, f, 3).mean((1, 3)), (img.width, img.height)
    return np.stack([np.asarray(Image.fromarray(x[..., k].astype(np.float32), 'F').resize((SIZE, SIZE), Image.Resampling.BOX)) for k in range(3)], -1), (img.width, img.height)


def to_dn(v, scene):
    lo, hi = np.asarray(scene['ms_low']), np.asarray(scene['ms_high'])
    return lo + v / 255.0 * (hi - lo)


def restore(pictures, meta, model, sensor, out, extra=None):
    """pictures: {scene: (name, bytes)} for all 20 scenes -> rr zip, raw zip and record."""
    if sorted(pictures) != list(range(SCENES)): raise ValueError(f'{model} {sensor}: scenes {sorted(pictures)}, expected 0-{SCENES - 1}')
    mats, raw, rows = {}, {}, []
    for i in range(SCENES):
        name, data = pictures[i]
        v, size = to_256(data)
        sr = to_dn(v, meta['scenes'][i])
        buf = io.BytesIO(); savemat(buf, dict(sr=sr), do_compression=False)
        mats[f'output_mulExm_{i}.mat'] = buf.getvalue(); raw[name] = data
        rows.append(dict(scene=i, picture=name, picture_sha256=sha(data), size=list(size)))
    out = Path(out)
    rr = write_zip(out / f'{model}_zeroshot_{sensor}_rr.zip', mats)
    record = dict(model=model, sensor=sensor, rgb_bands=meta['rgb_bands'], method='area average to 256x256, inverse of the image 1 stretch (genimage_io.py)',
                  rr_zip=rr.name, rr_zip_sha256=sha(rr), scenes=rows, **(extra or {}))
    paths = [rr]
    if raw:
        rz = write_zip(out / f'{model}_zeroshot_{sensor}_raw.zip', raw); record.update(raw_zip=rz.name, raw_zip_sha256=sha(rz)); paths.append(rz)
    rec = out / f'{model}_zeroshot_{sensor}_restore.json'; rec.write_text(json.dumps(record, indent=1) + '\n'); paths.append(rec)
    for p in paths: print(f'{p}  {p.stat().st_size:,} bytes', flush=True)
    return record


def cmd_restore(a):
    meta, inputs_sha = read_meta(a.inputs, a.sensor)
    pictures = {}
    for p in sorted(Path(a.images).iterdir()):
        m = PICTURE.search(p.name)
        if not m or m.group(1).upper() != a.sensor or not p.name.startswith(a.model + '_'): continue   # one model per call; folders may hold several
        i = int(m.group(2))
        if i in pictures: raise ValueError(f'two pictures for scene {i}: {pictures[i][0]} and {p.name}')
        pictures[i] = (p.name, p.read_bytes())
    restore(pictures, meta, a.model, a.sensor, a.out, dict(inputs_zip_sha256=inputs_sha))


def cmd_identity(a):
    for sensor in a.sensors:
        p = Path(a.inputs); p = p / f'genimage_inputs_{sensor}.zip' if p.is_dir() else p
        meta, inputs_sha = read_meta(p, sensor)
        with zipfile.ZipFile(p) as z:
            pictures = {i: (f'{sensor}_{i:02}_1_ms.png', z.read(f'{sensor}_{i:02}_1_ms.png')) for i in range(SCENES)}
        rec = restore(pictures, meta, a.model, sensor, a.out, dict(inputs_zip_sha256=inputs_sha, control='image 1 sent back unchanged: interpolation and 8-bit round trip, no model'))
        (Path(a.out) / rec['raw_zip']).unlink()      # the raw pictures are the inputs themselves


def load_predictions(path):
    with zipfile.ZipFile(path) as z:
        names = {int(re.search(r'(\d+)\.mat$', n).group(1)): n for n in z.namelist() if n.endswith('.mat')}
        if sorted(names) != list(range(SCENES)): raise ValueError(f'{path}: scenes {sorted(names)}')
        return [np.moveaxis(np.asarray(loadmat(io.BytesIO(z.read(names[i])))['sr'], dtype=np.float64), -1, 0) for i in range(SCENES)]


def highpass(x):
    """x minus its 5x5 box mean (edge replicated)."""
    from scipy.ndimage import uniform_filter
    return x - uniform_filter(x, 5, mode='nearest')


def scene_checks(rgb, s, sensor, wald, blur, torch):
    """rgb: (3,256,256) DN in R,G,B order."""
    from skimage.registration import phase_cross_correlation
    bands = SENSORS[sensor][1]
    with torch.no_grad():
        k = wald.ms_kernel[bands]
        f = torch.from_numpy(rgb)[None]
        low = blur(f, k)[..., 2::4, 2::4][0].numpy()
    e = low - s['ms'][bands]
    obs = float(np.sqrt(np.mean(e[..., 5:-5, 5:-5] ** 2)))
    inten, pan = rgb.mean(0), s['pan'][0]
    a, b = highpass(inten).ravel(), highpass(pan).ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    shift, _, _ = phase_cross_correlation(pan, inten, upsample_factor=10, normalization=None)
    return dict(observation_rmse_dn=obs, pan_detail_corr=corr, shift_rows=float(shift[0]), shift_cols=float(shift[1]))


def cmd_checks(a):
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments/performance/54_multi_sensor_scale_mixed_training'))
    from prepare import Wald, _blur  # x4 sensor MTF operator of experiment 54 (DLPan genMTF values)
    scenes, data = rr_scenes(a.data, a.sensor); wald = Wald(a.sensor); bands = SENSORS[a.sensor][1]
    result = dict(sensor=a.sensor, data=data, rgb_bands=bands, margin_obs=5, highpass='x minus 5x5 box mean', shift='phase correlation against PAN, upsample 10',
                  reference_ground_truth=[scene_checks(s['gt'][bands], s, a.sensor, wald, _blur, torch) for s in scenes])
    if a.predictions:
        preds = load_predictions(a.predictions)
        result.update(predictions=Path(a.predictions).name, predictions_sha256=sha(a.predictions),
                      scenes=[scene_checks(p, s, a.sensor, wald, _blur, torch) for p, s in zip(preds, scenes)])
    mean = lambda rows: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    result['mean_ground_truth'] = mean(result['reference_ground_truth'])
    if 'scenes' in result: result['mean'] = mean(result['scenes'])
    out = Path(a.out) if a.out else Path(a.predictions or '.').with_name(f'{Path(a.predictions).stem if a.predictions else "ground_truth_" + a.sensor}_checks.json')
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(out, json.dumps({k: result[k] for k in ('mean', 'mean_ground_truth') if k in result}), flush=True)


def gh(*args):
    return subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout


def cmd_upload(a):
    have = json.loads(gh('release', 'list', '-R', REPO, '--limit', '1000', '--json', 'tagName'))
    if not any(r['tagName'] == a.release for r in have):
        anchor = json.loads(gh('api', f'repos/{REPO}/git/ref/tags/{ANCHOR}'))['object']['sha']
        gh('release', 'create', a.release, '-R', REPO, '--target', anchor, '--title', a.title or a.release, '--latest=false', '--prerelease',
           '--notes', a.notes or '생성형 이미지 모델 팬샤프닝 시험(#209-212) 자료입니다. 만든 방법은 `SSA-MRN/common/genimage_io.py`에 있습니다.')
    assets = {x['name']: x for x in json.loads(gh('api', f'repos/{REPO}/releases/tags/{a.release}'))['assets']}
    for f in map(Path, a.files):
        digest = 'sha256:' + sha(f)
        if f.name in assets:
            if assets[f.name]['digest'] != digest: raise SystemExit(f'{f.name} already on {a.release} with other content; not replaced')
            print('already uploaded', f.name); continue
        gh('release', 'upload', a.release, '-R', REPO, str(f))
    assets = {x['name']: x for x in json.loads(gh('api', f'repos/{REPO}/releases/tags/{a.release}'))['assets']}
    for f in map(Path, a.files):
        if assets.get(f.name, {}).get('digest') != 'sha256:' + sha(f): raise SystemExit(f'upload check failed: {f.name}')
        print(f'verified {a.release}/{f.name} {assets[f.name]["digest"]}')


# ---------- generation through the API Gateway (issues #209-212) ----------
API = 'https://factchat-cloud.mindlogic.ai/v1/gateway'
RULES = """- Keep the exact framing, scale and geometry. Do not crop, rotate, shift, zoom, mirror or reframe. Every object must stay at the same pixel position as in both inputs.
- Take edges, textures and fine structures only from {pan}. Do not add, remove or invent any building, road, vehicle, tree, boat, field pattern or texture that is not visible in {pan}.
- Take colors only from {ms}. Keep its overall brightness and color balance. Do not increase saturation or contrast, and do not apply any artistic style, haze removal, color grading or sharpening halos.
- Output a single square (1:1) top-down satellite photograph, 1024×1024, with no text, labels, borders or watermarks."""
PROMPTS = {
    'gpt_image_2_6': """Edit the first image. Both images show the same satellite scene and are exactly aligned pixel to pixel.
The first image is a low-resolution multispectral color image: blurry, but its colors are correct.
The second image is a high-resolution panchromatic (grayscale) image of the same area: sharp, but without color.

Task: pansharpening. Make the first image as sharp as the second image while keeping the colors of the first image.

Rules:
""" + RULES.format(pan='the second image', ms='the first image'),
    'nano_banana_2_1': """Do not change the composition. I am giving you two images of the same satellite scene, exactly aligned pixel to pixel.
The first image is a low-resolution multispectral color image of the area: it is blurry, but its colors are correct.
The second image is a high-resolution panchromatic grayscale image of the same area: it is sharp, but it has no color.

Create one image that combines them (pansharpening): use the second image as the guide for all spatial detail and the first image as the guide for all color.

Rules:
""" + RULES.format(pan='the second image', ms='the first image'),
}
GEN_MODELS = {   # folder name -> API model id and fixed parameters (same as the image menu settings of #209)
    'gpt_image_2_6': ('gpt-image-2.6-sunburst', lambda q: dict(size='1024x1024', quality=q, background='opaque')),
    'nano_banana_2_1': ('gemini-nano-banana-2.1', lambda q: dict(aspect_ratio='1:1', image_size='1K')),
}
RETRIES = 2      # extra attempts after a failure (refusal, no picture, not square); never for a picture we merely dislike


def api(path, key, body=None, timeout=600):
    req = urllib.request.Request(API + path, data=None if body is None else json.dumps(body).encode(), method='GET' if body is None else 'POST',
                                 headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json', 'User-Agent': 'CtrS-genimage/1'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode(errors="replace")[:500]}') from None


def data_url(png_bytes):
    return 'data:image/png;base64,' + base64.b64encode(png_bytes).decode()


def fetch_picture(url):
    if url.startswith('data:'): return base64.b64decode(url.split(',', 1)[1])
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'CtrS-genimage/1'}), timeout=300) as r: return r.read()


def generate_one(key, model_id, request):
    """One API call (polls Fal operations); returns (picture bytes, credits charged, response id)."""
    resp = api('/images/generate/', key, request)
    credits = float(resp.get('credits_charged') or 0)
    waited = 0
    while not resp.get('data'):
        if 'operation_id' not in resp: raise RuntimeError(f'no picture in response: {json.dumps(resp)[:300]}')
        if waited > 900: raise RuntimeError('operation still processing after 15 minutes')
        time.sleep(5); waited += 5
        resp = {**api(f'/images/generate/{resp["operation_id"]}/?model={urllib.parse.quote(model_id, safe="")}', key), 'operation_id': resp['operation_id']}
        if resp.get('status') == 'failed': raise RuntimeError(f'operation failed: {json.dumps(resp)[:300]}')
    return fetch_picture(resp['data'][0]['url']), credits, resp.get('id') or resp.get('operation_id')


def cmd_generate(a):
    model_id, params = GEN_MODELS[a.model]
    if a.model == 'gpt_image_2_6' and not a.quality: a.quality = 'high'   # fixed for all four sensors in #209
    key = os.environ.get('BAZE_API_KEY')
    if not key and not a.dry_run: raise SystemExit('set the BAZE_API_KEY environment variable to your own API Gateway key first')
    meta, inputs_sha = read_meta(a.inputs, a.sensor)
    p = Path(a.inputs); p = p / f'genimage_inputs_{a.sensor}.zip' if p.is_dir() else p
    with zipfile.ZipFile(p) as z:
        pair = {i: (z.read(f'{a.sensor}_{i:02}_1_ms.png'), z.read(f'{a.sensor}_{i:02}_2_pan.png')) for i in range(SCENES)}
    fixed = dict(model=model_id, prompt=PROMPTS[a.model], number_of_images=1, **params(a.quality))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log_path = out / f'{a.model}_{a.sensor}_generation.json'
    log = json.loads(log_path.read_text()) if log_path.exists() else dict(model=a.model, model_id=model_id, sensor=a.sensor, inputs_zip_sha256=inputs_sha,
                                                                          settings={k: v for k, v in fixed.items() if k != 'prompt'}, prompt=PROMPTS[a.model], attempts=[])
    if log['settings'] != {k: v for k, v in fixed.items() if k != 'prompt'} or log['prompt'] != PROMPTS[a.model]:
        raise SystemExit(f'{log_path} was made with other settings or another prompt; use a new --out folder')
    done = {int(m.group(2)) for f in out.iterdir() if (m := PICTURE.search(f.name)) and f.name.startswith(a.model + '_') and m.group(1).upper() == a.sensor}
    todo = [i for i in (a.scenes if a.scenes is not None else range(SCENES)) if i not in done]
    print(f'{a.model} ({model_id}) {a.sensor}: {len(done)} saved, {len(todo)} to make; settings {log["settings"]}', flush=True)
    if a.dry_run:
        body = dict(fixed, input_images=[data_url(x) for x in pair[todo[0] if todo else 0]])
        print(f'request body {len(json.dumps(body)) / 1e6:.2f} MB, keys {sorted(body)}'); return
    est = api('/estimate/', key, dict(kind='image', **{k: v for k, v in fixed.items() if k != 'prompt'}, input_images=[data_url(x) for x in pair[0]]))
    print(f'estimate per picture: {est.get("credits")} credits ({est.get("bound")}); for {len(todo)} scenes about {float(est.get("credits") or 0) * len(todo):.0f}', flush=True)
    if a.estimate_only: return
    if a.max_credits is None: raise SystemExit('give --max-credits (the most this run may spend)')
    spent, per = 0.0, float(est.get('credits') or 0)   # per: reserve for the next picture (estimate, raised to the most actually charged)
    for i in todo:
        for attempt in range(RETRIES + 1):
            if spent + per > a.max_credits:
                print(f'stopping before scene {i}: {spent:.1f} spent, next would pass --max-credits {a.max_credits}'); log_path.write_text(json.dumps(log, indent=1) + '\n'); return
            row = dict(scene=i, attempt=attempt, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
            try:
                picture, credits, rid = generate_one(key, model_id, dict(fixed, input_images=[data_url(x) for x in pair[i]]))
                spent += credits; per = max(per, credits); row.update(credits_charged=credits, response_id=rid)
                img = Image.open(io.BytesIO(picture)); row.update(size=[img.width, img.height], format=img.format)
                if img.width != img.height: raise RuntimeError(f'not square: {img.width}x{img.height}')
                name = f'{a.model}_{a.sensor}_{i:02}.{dict(JPEG="jpg", WEBP="webp").get(img.format, "png")}'
                (out / name).write_bytes(picture); row.update(status='ok', picture=name, picture_sha256=sha(picture))
            except Exception as e:  # noqa: BLE001  record every failure, then retry under the fixed rule
                row.update(status='failed', error=str(e)[:500])
            log['attempts'].append(row); log_path.write_text(json.dumps(log, indent=1) + '\n')
            print(f'scene {i:02} attempt {attempt}: {row["status"]} {row.get("size", "")} credits {row.get("credits_charged", 0)} (total {spent:.1f}) {row.get("error", "")}', flush=True)
            if row['status'] == 'ok': break
    print(f'done: {spent:.1f} credits this run; log {log_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    data = dict(type=Path, default=ROOT / 'data', help='folder with the PanCollection RR test H5 files')
    i = sub.add_parser('inputs'); i.add_argument('--data', **data); i.add_argument('--sensors', nargs='+', default=list(SENSORS), choices=list(SENSORS)); i.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('restore'); r.add_argument('--sensor', choices=list(SENSORS), required=True); r.add_argument('--model', required=True)
    r.add_argument('--images', type=Path, required=True); r.add_argument('--inputs', type=Path, required=True, help='genimage_inputs_<SENSOR>.zip or its folder'); r.add_argument('--out', type=Path, required=True)
    d = sub.add_parser('identity'); d.add_argument('--inputs', type=Path, required=True); d.add_argument('--sensors', nargs='+', default=list(SENSORS), choices=list(SENSORS))
    d.add_argument('--model', default='genimage_identity'); d.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('checks'); c.add_argument('--data', **data); c.add_argument('--sensor', choices=list(SENSORS), required=True)
    c.add_argument('--predictions', type=Path, help='<model>_zeroshot_<SENSOR>_rr.zip (omit for the ground-truth reference only)'); c.add_argument('--out', type=Path)
    g = sub.add_parser('generate'); g.add_argument('--sensor', choices=list(SENSORS), required=True); g.add_argument('--model', choices=list(GEN_MODELS), required=True)
    g.add_argument('--inputs', type=Path, required=True, help='genimage_inputs_<SENSOR>.zip or its folder'); g.add_argument('--out', type=Path, required=True)
    g.add_argument('--quality', choices=('low', 'medium', 'high'), help='GPT Image only; default high, the value fixed for all sensors in #209')
    g.add_argument('--scenes', type=int, nargs='+', help='only these scenes (e.g. 0 for the first check); default all 20')
    g.add_argument('--max-credits', type=float, help='stop before this run would spend more'); g.add_argument('--estimate-only', action='store_true')
    g.add_argument('--dry-run', action='store_true', help='check the inputs and the request without the key or any call')
    u = sub.add_parser('upload'); u.add_argument('--release', required=True); u.add_argument('--title'); u.add_argument('--notes'); u.add_argument('files', nargs='+')
    a = p.parse_args()
    dict(inputs=cmd_inputs, restore=cmd_restore, identity=cmd_identity, checks=cmd_checks, upload=cmd_upload, generate=cmd_generate)[a.command](a)


if __name__ == '__main__':
    main()
