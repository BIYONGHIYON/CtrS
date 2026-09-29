"""Evaluate one reduced-resolution PanCollection sample with trained weights."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssamrn.data.pancollection import PanCollectionH5  # noqa: E402
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sensor", choices=("QB", "GF2", "WV3"), required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    dataset = PanCollectionH5(args.data, sensor=args.sensor)
    sample = dataset[args.sample_index]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["sensor"] != args.sensor or checkpoint["channels"] != dataset.channels:
        raise ValueError("Checkpoint sensor/channels do not match the test data")
    dimension = checkpoint.get("ssai_dimension", 4)
    model = RestoredPansharpeningNet(channels=dataset.channels, ssai_dimension=dimension)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model(*(sample[key].unsqueeze(0) for key in ("pan", "lms", "ms")))[0]
    elapsed = time.perf_counter() - started
    if not torch.isfinite(prediction).all():
        raise ValueError("Prediction contains non-finite values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "prediction_bands.npy", prediction.numpy())
    # PanCollection GF2/QB: B,G,R,NIR; WV3: Coastal,B,G,Yellow,R,RedEdge,NIR1,NIR2.
    bands = [4, 2, 1] if args.sensor == "WV3" else [2, 1, 0]
    gt_rgb = np.moveaxis(sample["gt"].numpy()[bands], 0, -1)
    lo = np.percentile(gt_rgb, 1, axis=(0, 1))
    hi = np.percentile(gt_rgb, 99, axis=(0, 1))

    def rgb(tensor):
        array = np.moveaxis(tensor.numpy()[bands], 0, -1)
        array = np.clip((array - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
        return Image.fromarray((array * 255).astype(np.uint8))

    pan = sample["pan"][0].numpy()
    p_lo, p_hi = np.percentile(pan, [1, 99])
    pan_image = Image.fromarray(
        (np.clip((pan - p_lo) / max(p_hi - p_lo, 1e-8), 0, 1) * 255).astype(np.uint8)
    ).convert("RGB")
    panels = (
        ("MS input", rgb(sample["ms"]), "input_ms.png"),
        ("PAN input", pan_image, "input_pan.png"),
        ("LMS baseline", rgb(sample["lms"]), "input_lms.png"),
        ("Trained prediction", rgb(prediction), "prediction_rgb.png"),
        ("Ground truth", rgb(sample["gt"]), "ground_truth_rgb.png"),
    )
    width, height = sample["gt"].shape[-2:][::-1]
    canvas = Image.new("RGB", (width * len(panels), height + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, picture, filename) in enumerate(panels):
        picture.save(args.output_dir / filename)
        canvas.paste(picture.resize((width, height), Image.Resampling.NEAREST), (width * index, 24))
        draw.text((width * index + 8, 6), label, fill="black")
    canvas.save(args.output_dir / "comparison.png")

    def metrics(tensor):
        mse = float(torch.mean((tensor - sample["gt"]) ** 2))
        return {"mse": mse, "psnr_db_peak_1": float(-10 * np.log10(max(mse, 1e-30)))}

    report = {
        "sensor": args.sensor,
        "checkpoint_file": args.checkpoint.name,
        "data_file": args.data.name,
        "epoch": checkpoint["epoch"],
        "ssai_dimension": dimension,
        "sample_index_zero_based": args.sample_index,
        "sample_count": len(dataset),
        "device": "cpu",
        "inference_seconds": elapsed,
        "input_shapes": {key: list(sample[key].shape) for key in ("pan", "lms", "ms")},
        "prediction_shape": list(prediction.shape),
        "prediction": metrics(prediction),
        "lms_baseline": metrics(sample["lms"]),
        "notes": "Single-sample sanity check, not official paper evaluation. RGB band mapping is assumed; shared GT-based 1-99 percentile stretch. Full sample, no inference crop/downsampling.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
