"""Prepare a local CAS500-1 FR transfer pool; never modify source imagery.

Requires numpy, Pillow, scipy and h5py. Run with the original TIFF directory
as --source. Output contains normalized pan/ms/lms, coordinates and provenance;
it deliberately has no GT, train/validation split or synthetic RR degradation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve1d

Image.MAX_IMAGE_PIXELS = None
BANDS = ('B', 'G', 'R', 'N')
SCALE = np.float32(4095)
RATIO = 4
MS_PATCH = 128
HALO = 16  # > support of the two-stage 23-tap interpolator (8.25 MS pixels).
HALF = [.5, .305334091185, 0, -.072698593239, 0, .021809577942,
        0, -.005192756653, 0, .000807762146, 0, -.000060081482]
KERNEL = np.asarray(HALF[:0:-1] + HALF, dtype=np.float64) * 2


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class Raster:
    """Bounded reads of the uncompressed, one-row TIFF strips in this delivery."""
    def __init__(self, path):
        self.path = path
        with path.open('rb') as f:
            if f.read(2) != b'II':
                raise ValueError('Expected little-endian delivery TIFF')
        with Image.open(path) as im:
            t = im.tag_v2
            assert im.mode == 'I;16' and t[259] == 1 and t[278] == 1
            assert t.get(274, 1) == 1 and t.get(277, 1) == 1
            self.w, self.h = im.size
            self.offsets = np.asarray(t[273], dtype=np.int64)
            counts = np.asarray(t[279], dtype=np.int64)
            assert len(counts) == self.h and np.all(counts == self.w * 2)
            assert np.all(self.offsets + counts <= path.stat().st_size)
            order = np.argsort(self.offsets)
            assert np.all(self.offsets[order][1:] >= (self.offsets + counts)[order][:-1])
        self.mm = np.memmap(path, mode='r', dtype='u1')

    def crop(self, y, x, h, w):
        assert 0 <= y < y+h <= self.h and 0 <= x < x+w <= self.w
        return np.stack([np.frombuffer(self.mm, dtype='<u2', count=w,
                                      offset=int(self.offsets[j]) + 2*x)
                         for j in range(y, y+h)])


def interpolate(ms):
    # Same coefficients, insertion phases and wrap boundary as common/interpolation.py.
    a = ms.astype(np.float64)
    for stage in range(2):
        c, h, w = a.shape
        up = np.zeros((c, h*2, w*2), dtype=np.float64)
        phase = 1 if stage == 0 else 0
        up[:, phase::2, phase::2] = a
        a = convolve1d(convolve1d(up, KERNEL, axis=1, mode='wrap'),
                       KERNEL, axis=2, mode='wrap')
    return a.astype(np.float32)


def stretch(a):
    lo, hi = np.percentile(a, [2, 98])
    return (np.clip((a-lo)/max(hi-lo, 1e-12), 0, 1)*255).astype('u1')


def audit_alignment(rasters):
    checks = []
    for y, x in [(2200, 2500), (3200, 4100), (4400, 4700), (5400, 2600)]:
        n = 384
        g = rasters['G'].crop(y, x, n, n).astype('f4')
        p = rasters['P'].crop(y*4, x*4, n*4, n*4).astype('f4')
        p = p.reshape(n, 4, n, 4).mean((1, 3))
        def grad(a):
            return np.hypot((a[1:-1, 2:]-a[1:-1, :-2])/2,
                            (a[2:, 1:-1]-a[:-2, 1:-1])/2)
        a, b = grad(g), grad(p)
        margin = 8
        a = a[margin:-margin, margin:-margin]
        a = a-a.mean()
        scores = []
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                bb = b[margin+dy:b.shape[0]-margin+dy,
                       margin+dx:b.shape[1]-margin+dx]
                bb = bb-bb.mean()
                scores.append((float(np.sum(a*bb)/np.sqrt(np.sum(a*a)*np.sum(bb*bb))), dy, dx))
        score, dy, dx = max(scores)
        if (dy, dx) != (0, 0):
            raise ValueError('New integer displacement detected; review geometry before extraction')
        checks.append(dict(ms_yx=[y, x], ms_size=n, best_dy_dx_ms=[dy, dx],
                           gradient_correlation=score, search_radius_ms=6))
    return dict(method='G gradient magnitude vs 4x4 area-averaged PAN; integer NCC',
                checks=checks, applied_translation_pan_yx=[0, 0],
                subpixel_registration='not certified; no subpixel warp applied')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    final = out/'fr_patches.h5'
    partial = out/'fr_patches.partial.h5'
    if final.exists() or partial.exists():
        raise FileExistsError('Refusing to overwrite an existing patch collection')
    rasters = {b: Raster(next(args.source.glob('*_'+b+'.tif'))) for b in BANDS+('P',)}
    xml_path = next(args.source.glob('*_Aux.xml'))
    root = ET.parse(xml_path).getroot()
    assert root.findtext('General/Satellite') == 'CAS500-1'
    assert root.findtext('General/ProductLevel') == 'Level2G'
    h, w = rasters['B'].h, rasters['B'].w
    assert all((rasters[b].h, rasters[b].w) == (h, w) for b in BANDS)
    assert (rasters['P'].h, rasters['P'].w) == (h*4+1, w*4+1)
    for b in BANDS+('P',):
        entry = next(e for e in root.find('Image') if e.findtext('ImageFileName') == rasters[b].path.name)
        assert int(entry.findtext('ImageSize/Width')) == rasters[b].w
        assert int(entry.findtext('ImageSize/Height')) == rasters[b].h
    sources = [dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=digest(p))
               for p in [xml_path]+[rasters[b].path for b in BANDS+('P',)]]
    align = audit_alignment(rasters)
    p = rasters['P']
    removed_edge = dict(bottom_row_nonzero=int(np.count_nonzero(p.crop(p.h-1, 0, 1, p.w))),
                        right_column_nonzero=int(np.count_nonzero(p.crop(0, p.w-1, p.h, 1))))
    coords, rejected = [], []
    start = time.monotonic()
    # Entire scene scanned on a predeclared nonoverlapping grid. No quality/ranking selection.
    for y in range(0, h-MS_PATCH+1, MS_PATCH):
        for x in range(0, w-MS_PATCH+1, MS_PATCH):
            if y < HALO or x < HALO or y+MS_PATCH+HALO > h or x+MS_PATCH+HALO > w:
                rejected.append([y, x, 'scene_edge_context']); continue
            n = MS_PATCH+2*HALO
            ctx = np.stack([rasters[b].crop(y-HALO, x-HALO, n, n) for b in BANDS])
            pc = p.crop((y-HALO)*4, (x-HALO)*4, n*4, n*4)
            # Conservative mask: zero DN is excluded in every band and the full context halo.
            if np.any(ctx == 0) or np.any(pc == 0):
                rejected.append([y, x, 'zero_dn_in_patch_or_context']); continue
            coords.append((y, x))
    if not coords:
        raise ValueError('No valid patches')
    print(json.dumps(dict(selected=len(coords), rejected=len(rejected), alignment=align,
                          removed_pan_edge=removed_edge)), flush=True)
    ranges = {k: [float('inf'), float('-inf')] for k in ('pan', 'ms', 'lms')}
    with h5py.File(partial, 'x') as f:
        f.attrs.update(sensor='CAS500-1', product_level='L2G', task='FR_transfer',
                       layout='NCHW', band_order='B,G,R,NIR', normalization='DN / 4095; no clipping',
                       scale_dn=4095., ratio=4, has_ground_truth=False,
                       registration='original grid; trim bottom/right PAN edge; no warp',
                       lms_interpolation='two-stage interp23tap with 16-MS-pixel source halo',
                       split='unassigned single-scene transfer pool')
        ds = {}
        for key, shape in [('pan', (1, 512, 512)), ('ms', (4, 128, 128)), ('lms', (4, 512, 512))]:
            ds[key] = f.create_dataset(key, shape=(len(coords),)+shape, dtype='f4',
                                       chunks=(1,)+shape, compression='lzf', shuffle=True, fletcher32=True)
        f.create_dataset('ms_yx', data=np.asarray(coords, dtype='i4'))
        f.create_dataset('pan_yx', data=np.asarray(coords, dtype='i4')*4)
        for i, (y, x) in enumerate(coords):
            n = MS_PATCH+2*HALO
            ctx = np.stack([rasters[b].crop(y-HALO, x-HALO, n, n) for b in BANDS]).astype('f4')/SCALE
            arrays = dict(ms=ctx[:, HALO:HALO+MS_PATCH, HALO:HALO+MS_PATCH],
                          pan=p.crop(y*4, x*4, 512, 512)[None].astype('f4')/SCALE,
                          lms=interpolate(ctx)[:, HALO*4:(HALO+MS_PATCH)*4, HALO*4:(HALO+MS_PATCH)*4])
            for key, a in arrays.items():
                assert np.isfinite(a).all()
                ranges[key][0] = min(ranges[key][0], float(a.min()))
                ranges[key][1] = max(ranges[key][1], float(a.max()))
                ds[key][i] = a
            if (i+1) % 100 == 0 or i+1 == len(coords):
                print(f'written {i+1}/{len(coords)} elapsed={time.monotonic()-start:.1f}s', flush=True)
        f.flush()
    # Read every stored chunk back (including Fletcher32), compare normalized DN to sources,
    # and check complete indexing/band order. Keep original TIFFs as the canonical DN source.
    with h5py.File(partial, 'r') as f:
        assert 'gt' not in f and len(set(coords)) == len(coords)
        for i, (y, x) in enumerate(coords):
            assert np.array_equal(f['pan_yx'][i], [y*4, x*4])
            for key in ('pan', 'ms', 'lms'):
                assert np.isfinite(f[key][i]).all()
            raw_ms = np.stack([rasters[b].crop(y, x, 128, 128) for b in BANDS])
            raw_pan = p.crop(y*4, x*4, 512, 512)[None]
            assert np.array_equal(np.rint(f['ms'][i]*SCALE).astype('u2'), raw_ms)
            assert np.array_equal(np.rint(f['pan'][i]*SCALE).astype('u2'), raw_pan)
        # Larger context must give the same LMS crop: patch boundaries cannot wrap into content.
        for i in [0, len(coords)//2, len(coords)-1]:
            y, x = coords[i]
            halo = 24
            if y < halo or x < halo or y+128+halo > h or x+128+halo > w:
                continue
            ctx = np.stack([rasters[b].crop(y-halo, x-halo, 128+2*halo, 128+2*halo)
                            for b in BANDS]).astype('f4')/SCALE
            expected = interpolate(ctx)[:, halo*4:(halo+128)*4, halo*4:(halo+128)*4]
            assert np.array_equal(expected, f['lms'][i])
        examples = [0, len(coords)//2, len(coords)-1]
        preview = Image.new('RGB', (512*2, 3*538), 'white')
        draw = ImageDraw.Draw(preview)
        for row, i in enumerate(examples):
            lms, pan = f['lms'][i], f['pan'][i, 0]
            rgb = np.stack([stretch(lms[c]) for c in [2, 1, 0]], axis=-1)
            preview.paste(Image.fromarray(rgb), (0, row*538+26))
            preview.paste(Image.fromarray(stretch(pan)).convert('RGB'), (512, row*538+26))
            draw.text((6, row*538+6), f'patch {i} | MS y,x={coords[i]} | LMS RGB / PAN', fill='black')
        preview.save(out/'patch_preview.png')
    partial.rename(final)
    np.savetxt(out/'patch_coordinates.csv', np.asarray([[i,y,x,y*4,x*4] for i,(y,x) in enumerate(coords)]),
               delimiter=',', fmt='%d', header='patch_id,ms_y,ms_x,pan_y,pan_x', comments='')
    (out/'rejected_patches.json').write_text(json.dumps(rejected, indent=1))
    manifest = dict(schema_version=1, source_scene=xml_path.name.removesuffix('_Aux.xml'),
                    sources=sources, bands=list(BANDS), normalization=dict(divisor=4095, clip=False,
                    per_band_minmax=False, normalized_ranges=ranges,
                    inverse='round(normalized * 4095) reconstructs original uint16 PAN/MS'),
                    geometry=dict(ms_shape_hw=[h,w], source_pan_shape_hw=[p.h,p.w],
                    aligned_pan_shape_hw=[h*4,w*4], crop_pan_top_left_bottom_right=[0,0,1,1],
                    removed_pan_edge=removed_edge, alignment=align,
                    georeferencing='TIFFs have no georeferencing tags; pixel coordinates only',
                    metadata_ms_gsd_xy=[2.111,2.189], metadata_pan_gsd_xy=[.5,.5],
                    gsd_note='XML MS GSD differs from nominal 2 m; not used to resample image grid'),
                    patches=dict(count=len(coords), ms_size=128, pan_size=512, stride_ms=128,
                    context_halo_ms=HALO, selection='all grid patches positive in all PAN/MS context pixels',
                    rejected=len(rejected), spatial_overlap=False, split='unassigned', independent_scenes=1),
                    lms=dict(method='interp23tap', coefficients=KERNEL.tolist(),
                    insertion_phases=[1,0], calculation_dtype='float64', stored_dtype='float32',
                    boundary='wrap at context boundary; cropped halo removes boundary influence'),
                    ground_truth=False, rr_degradation=None, inference_run=False,
                    verification=dict(all_chunks_read_back=True, all_pan_ms_dn_roundtrips_exact=True,
                    lms_larger_halo_equality_checked=True, source_files_modified=False),
                    output=dict(path=final.name, bytes=final.stat().st_size, sha256=digest(final)),
                    preparation=dict(script=Path(__file__).name, sha256=digest(Path(__file__)),
                    numpy=np.__version__, h5py=h5py.__version__, elapsed_seconds=time.monotonic()-start))
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(dict(output=str(final), count=len(coords), size_bytes=final.stat().st_size,
                          ranges=ranges, elapsed_seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__':
    main()
