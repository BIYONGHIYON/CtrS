"""Read-only audit of a few PCB-Vision RGB/ENVI pairs.

Only small, strided views of each HSI cube are materialized. Reports are
written under experiments/results, never into the source dataset.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage import color, feature, measure, transform


DEFAULT_DATASET = Path("/Volumes/T7 Shield/윤지/병현/SSA-MRN Data/RGB-HSI/PCBDataset")


def read_header(path):
    text = path.read_text(encoding="latin-1")
    def scalar(key):
        match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([^\r\n]+)", text)
        if not match:
            raise ValueError(f"Missing ENVI field {key}: {path}")
        return match.group(1).strip()
    wavelengths = re.search(r"(?is)^\s*wavelength\s*=\s*\{([^}]+)\}", text, re.M)
    return {
        "samples": int(scalar("samples")), "lines": int(scalar("lines")),
        "bands": int(scalar("bands")), "interleave": scalar("interleave").lower(),
        "data_type": int(scalar("data type")), "byte_order": int(scalar("byte order")),
        "header_offset": int(scalar("header offset")),
        "wavelengths_nm": ([float(x) for x in wavelengths.group(1).split(",") if x.strip()]
                           if wavelengths else []),
    }


def open_cube(path, header):
    if header["data_type"] != 4 or header["byte_order"] != 0:
        raise ValueError(f"Unsupported ENVI dtype/endian: {path}")
    h, w, b = (header[k] for k in ("lines", "samples", "bands"))
    expected = header["header_offset"] + h * w * b * 4
    if path.stat().st_size != expected:
        raise ValueError(f"Size mismatch for {path}: {path.stat().st_size} != {expected}")
    layout = header["interleave"]
    shapes = {"bip": (h, w, b), "bil": (h, b, w), "bsq": (b, h, w)}
    if layout not in shapes:
        raise ValueError(f"Unsupported interleave: {layout}")
    return np.memmap(path, dtype="<f4", mode="r", offset=header["header_offset"], shape=shapes[layout])


def band(cube, layout, index, step=1):
    if layout == "bip":
        return np.asarray(cube[::step, ::step, index])
    if layout == "bil":
        return np.asarray(cube[::step, index, ::step])
    return np.asarray(cube[index, ::step, ::step])


def display_band(values):
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1, 99])
    return np.uint8(np.round(255 * np.clip((values - low) / max(high - low, 1e-12), 0, 1)))


def match_features(rgb, hsi):
    """Exploratory match score; never treated as proof of physical registration."""
    gray_rgb = color.rgb2gray(rgb)
    gray_hsi = color.rgb2gray(hsi)
    detectors = []
    for gray in (gray_hsi, gray_rgb):
        detector = feature.ORB(n_keypoints=800, fast_threshold=0.04)
        try:
            detector.detect_and_extract(gray)
        except RuntimeError:
            return {"matches": 0, "inliers": 0, "status": "no_features"}
        detectors.append(detector)
    matches = feature.match_descriptors(detectors[0].descriptors, detectors[1].descriptors,
                                        cross_check=True, max_ratio=0.8)
    result = {"matches": int(len(matches)), "inliers": 0, "status": "insufficient_matches"}
    if len(matches) < 8:
        return result
    src = detectors[0].keypoints[matches[:, 0]][:, ::-1]
    dst = detectors[1].keypoints[matches[:, 1]][:, ::-1]
    try:
        model, inliers = measure.ransac((src, dst), transform.ProjectiveTransform,
                                        min_samples=4, residual_threshold=5, max_trials=1000,
                                        rng=np.random.default_rng(42))
    except (ValueError, np.linalg.LinAlgError):
        return result
    if model is None:
        return result
    result.update(inliers=int(inliers.sum()), status="estimated",
                  median_reprojection_px=float(np.median(np.linalg.norm(model(src[inliers]) - dst[inliers], axis=1)))
                  if inliers.any() else None)
    return result


def inspect_pair(dataset, scene_id, output_dir):
    rgb_path = dataset / "RGB" / f"{scene_id}.jpg"
    hsi_path = dataset / "HSI" / f"pcb{scene_id}" / f"pcb{scene_id}"
    header = read_header(hsi_path.with_suffix(".hdr"))
    cube = open_cube(hsi_path, header)
    wavelengths = np.asarray(header["wavelengths_nm"])
    wavelength_count_matches = len(wavelengths) == header["bands"]
    step = max(1, int(np.ceil(max(header["lines"], header["samples"]) / 700)))
    indices = ([int(np.argmin(abs(wavelengths - target))) for target in (650, 550, 450)]
               if wavelength_count_matches else [89, 64, 28])
    channels = [band(cube, header["interleave"], index, step) for index in indices]
    hsi_preview = np.stack([display_band(channel) for channel in channels], axis=2)
    sample = np.stack([band(cube, header["interleave"], index, max(step, 8)) for index in range(header["bands"])])
    finite = sample[np.isfinite(sample)]
    with Image.open(rgb_path) as source:
        rgb_size = source.size
        rgb = source.convert("RGB")
        rgb.thumbnail((700, 700))
    hsi_image = Image.fromarray(hsi_preview)
    aligned_rgb = np.asarray(rgb.resize(hsi_image.size), dtype=np.float32) / 255.0
    aligned_hsi = np.asarray(hsi_image, dtype=np.float32) / 255.0
    registration = match_features(aligned_rgb, aligned_hsi)
    panel = Image.new("RGB", (1400, 760), "white")
    draw = ImageDraw.Draw(panel)
    for image, x, label in ((rgb, 0, "Camera RGB"), (hsi_image, 700, f"HSI display band indices {indices}")):
        image = image.copy()
        image.thumbnail((690, 700))
        panel.paste(image, (x, 40))
        draw.text((x + 8, 10), label, fill="black")
    panel.save(output_dir / f"pcb{scene_id}_pair.png")
    return {
        "pcb": scene_id, "rgb_size_wh": list(rgb_size),
        "hsi_size_wh": [header["samples"], header["lines"]],
        "hsi_bands": header["bands"], "interleave": header["interleave"],
        "header_wavelength_count": len(wavelengths),
        "wavelength_count_matches_bands": wavelength_count_matches,
        "header_wavelength_min_max_nm": ([float(wavelengths.min()), float(wavelengths.max())]
                                         if len(wavelengths) else None),
        "display_band_indices_zero_based": indices,
        "sampled_values": {"min": float(finite.min()), "max": float(finite.max()),
                           "p01": float(np.percentile(finite, 1)), "p50": float(np.median(finite)),
                           "p99": float(np.percentile(finite, 99)),
                           "nonfinite": int(sample.size - finite.size), "sample_count": int(sample.size)},
        "registration_probe": registration,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--ids", type=int, nargs="+", default=[1, 13, 27, 40, 53])
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/pcb_pilot"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = [inspect_pair(args.dataset, scene_id, args.output_dir) for scene_id in args.ids]
    (args.output_dir / "inspection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
