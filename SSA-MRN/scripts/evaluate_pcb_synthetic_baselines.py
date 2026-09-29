"""Small HSI-only synthetic 4x benchmark; no real RGB-HSI registration assumed."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

from inspect_pcb_pairs import DEFAULT_DATASET, open_cube, read_header


def center_patch(cube, header, size):
    h, w, bands = (header[key] for key in ("lines", "samples", "bands"))
    if min(h, w) < size:
        raise ValueError(f"Cube {w}x{h} is smaller than {size}")
    y, x = (h - size) // 2, (w - size) // 2
    layout = header["interleave"]
    if layout == "bip":
        patch = cube[y:y + size, x:x + size, :]
    elif layout == "bil":
        patch = cube[y:y + size, :, x:x + size].transpose(0, 2, 1)
    else:
        patch = cube[:, y:y + size, x:x + size].transpose(1, 2, 0)
    return np.asarray(patch, dtype=np.float32).copy().reshape(size, size, bands)


def metrics(reference, prediction):
    valid = np.isfinite(reference).all(axis=2) & np.isfinite(prediction).all(axis=2)
    ref, pred = reference[valid], prediction[valid]
    mse = float(np.mean((ref - pred) ** 2))
    dot = np.sum(ref * pred, axis=1)
    norms = np.linalg.norm(ref, axis=1) * np.linalg.norm(pred, axis=1)
    spectral_valid = norms > 1e-8
    sam = np.degrees(np.arccos(np.clip(dot[spectral_valid] / norms[spectral_valid], -1, 1)))
    return {"psnr_db_peak_1": float(-10 * np.log10(mse)) if mse > 0 else None,
            "sam_deg": float(np.mean(sam)) if len(sam) else None,
            "valid_pixels": int(valid.sum())}


def evaluate(dataset, scene_id, size, scale):
    path = dataset / "HSI" / f"pcb{scene_id}" / f"pcb{scene_id}"
    header = read_header(path.with_suffix(".hdr"))
    reference = center_patch(open_cube(path, header), header, size)
    # Fixed, disclosed synthetic degradation. This is not the true FX10 optics.
    blurred = gaussian_filter(reference, sigma=(1.2, 1.2, 0), mode="reflect")
    low = resize(blurred, (size // scale, size // scale, header["bands"]), order=1,
                 anti_aliasing=False, preserve_range=True).astype(np.float32)
    output = {"pcb": scene_id, "reference_shape_hwb": list(reference.shape),
              "synthetic_lr_shape_hwb": list(low.shape),
              "reference_outside_0_1_fraction": float(np.mean((reference < 0) | (reference > 1))),
              "baselines": {}}
    for name, order in (("nearest", 0), ("bicubic", 3)):
        prediction = resize(low, reference.shape, order=order, anti_aliasing=False,
                            preserve_range=True).astype(np.float32)
        output["baselines"][name] = metrics(reference, prediction)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ids", type=int, nargs="+", default=[1, 13, 27, 40, 53])
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/pcb_pilot/synthetic_baselines.json"))
    args = parser.parse_args()
    if args.size % args.scale:
        parser.error("--size must be divisible by --scale")
    result = {"protocol": "center HSI crop; Gaussian sigma=1.2; linear resize to 1/4; upsample",
              "note": "HSI-only synthetic reduction, not real RGB-HSI ground truth",
              "scenes": [evaluate(args.dataset, scene_id, args.size, args.scale) for scene_id in args.ids]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
