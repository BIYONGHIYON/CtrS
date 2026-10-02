"""Render five comparable panels from saved LIB weights without additional training."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SSA-MRN/src"))
from ssamrn.data.lib_hsi import LIBHSI
from ssamrn.models.rgb_hsi import RGBHSISsaMRN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "SSA-MRN/experiments/checkpoints/lib_rgb_hsi/smoke/best.pt")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sample-index", type=int, default=0, help="Validation tile index (zero based)")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "SSA-MRN/experiments/results/lib_rgb_hsi_smoke")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    data = LIBHSI(args.data_root or config["data_root"], args.split, config["patch_size"],
                  alignment_manifest=(ROOT/config['alignment_manifest']) if config.get('alignment_manifest') else None)
    sample = data[args.sample_index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(6)
    model = RGBHSISsaMRN(204, config["latent_channels"], config["ssai_dimension"]).to(device).eval()
    model.load_state_dict(state["model"])
    with torch.inference_mode(), torch.autocast(device.type, enabled=config["amp"] and device.type == "cuda"):
        prediction = model(sample["rgb"].unsqueeze(0).to(device), sample["lr_hsi"].unsqueeze(0).to(device))[0].float().cpu()
    baseline = F.interpolate(sample["lr_hsi"].unsqueeze(0), size=sample["gt"].shape[-2:],
                             mode="bicubic", align_corners=False)[0]
    bands = [69, 52, 18]
    gt_rgb = sample["gt"][bands].permute(1, 2, 0).numpy()
    lo, hi = np.percentile(gt_rgb, [1, 99], axis=(0, 1))

    def hsi_display(tensor):
        array = tensor[bands].permute(1, 2, 0).numpy()
        rgb = np.clip((array - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
        return Image.fromarray((rgb * 255).astype(np.uint8))

    rgb = Image.fromarray((sample["rgb"].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    size = config["patch_size"]
    panels = [(f"LR HSI input ({size//4}x{size//4})", hsi_display(sample["lr_hsi"]), "input_lr_hsi.png"),
              (f"RGB guide ({size}x{size})", rgb, "input_rgb.png"),
              ("Bicubic baseline", hsi_display(baseline), "bicubic.png"),
              ("Prediction: best.pt", hsi_display(prediction), "prediction.png"),
              ("HSI ground truth", hsi_display(sample["gt"]), "ground_truth.png")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    width, top, bottom = 256, 70, 42
    canvas = Image.new("RGB", (width * 5, width + top + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    title_font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else font
    steps = int(next(iter(state["optimizer"]["state"].values()))["step"])
    scope = "smoke" if args.checkpoint.parent.name == "smoke" else "checkpoint"
    scene_tile = args.sample_index % data.per_scene
    draw.text((10, 8), f"LIB-HSI | epoch {state['epoch']} | {scope}: {steps} steps | {args.split} {sample['scene']} | tile {scene_tile} (index {args.sample_index})", fill="black", font=title_font)
    for i, (label, picture, filename) in enumerate(panels):
        picture.save(args.output_dir / filename)
        draw.text((i * width + 9, 43), label, fill="black", font=font)
        canvas.paste(picture.resize((width, width), Image.Resampling.NEAREST), (i * width, top))
    draw.text((10, width + top + 10), "HSI display: bands 69/52/18; same GT-based 1-99% stretch for all HSI panels. RGB guide uses original color.", fill="black", font=font)
    canvas.save(args.output_dir / "comparison.png")

    def metrics(image):
        error = (image - sample['gt']).square()
        mse = error[:,sample['valid_mask']].mean().item() if 'valid_mask' in sample else error.mean().item()
        return {"mse": mse, "psnr_db_range_1": float(-10 * np.log10(max(mse, 1e-12)))}

    report = {"checkpoint": str(args.checkpoint), "epoch": state["epoch"], "scene": sample["scene"],
              "sample_index": args.sample_index, "patch_size": config["patch_size"],
              "prediction": metrics(prediction), "bicubic": metrics(baseline),
              "split": args.split,
              "scene_tile": scene_tile,
              "notes": "Single tile illustration; all 204 bands for metrics; not a full-split benchmark."}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(args.output_dir / "comparison.png")


if __name__ == "__main__":
    main()
