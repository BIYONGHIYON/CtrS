"""Render four comparable panels from saved LIB weights without additional training."""
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
from ssamrn.models.rgb_grouped import build_rgb_hsi_model
from ssamrn.models.interp23 import interp23tap
from ssamrn.models.area_consistency import project_area_consistency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--alignment-manifest", type=Path, help="Exact registration file required by the checkpoint")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--sample-index", type=int, default=0, help="Split sample index (zero based)")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--area-consistency", action="store_true", help="Apply x4 area LR consistency to prediction and baseline")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = state["config"]
    if args.area_consistency and config.get('degradation') != 'area':
        parser.error('--area-consistency requires area degradation')
    manifest = args.alignment_manifest.resolve() if args.alignment_manifest else (ROOT/config['alignment_manifest']) if config.get('alignment_manifest') else None
    if manifest and config.get('alignment_sha256'):
        import hashlib
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != config['alignment_sha256']:
            parser.error('Alignment manifest differs from the checkpoint')
    data = LIBHSI(ROOT/(args.data_root or config["data_root"]), args.split, config["patch_size"],
                  alignment_manifest=manifest, eval_layout=config.get('eval_layout','tiles'),
                  degradation=config.get('degradation','bicubic'))
    sample = data[args.sample_index]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_num_threads(6)
    model = build_rgb_hsi_model(config).to(device).eval()
    model.load_state_dict(state["model"])
    with torch.inference_mode():
        lr_device = sample["lr_hsi"].unsqueeze(0).to(device)
        with torch.autocast(device.type, enabled=config["amp"] and device.type == "cuda"):
            prediction = model(sample["rgb"].unsqueeze(0).to(device), lr_device)
        prediction = prediction.float()
        if args.area_consistency:
            mask = sample.get('valid_mask')
            prediction = project_area_consistency(prediction, lr_device, mask.unsqueeze(0).to(device) if mask is not None else None)
        prediction = prediction[0].cpu()
    use_23tap = config.get('upsampler') == '23tap'
    baseline = (interp23tap(sample['lr_hsi'].unsqueeze(0),4) if use_23tap else
                F.interpolate(sample["lr_hsi"].unsqueeze(0), size=sample["gt"].shape[-2:],
                              mode="bicubic", align_corners=False))[0]
    baseline_key = 'interp23' if use_23tap else 'bicubic'
    if args.area_consistency:
        mask = sample.get('valid_mask')
        baseline = project_area_consistency(baseline.unsqueeze(0), sample['lr_hsi'].unsqueeze(0),
                                            mask.unsqueeze(0) if mask is not None else None)[0]
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
              (f"Prediction: {args.checkpoint.name}", hsi_display(prediction), "prediction.png"),
              ("HSI ground truth", hsi_display(sample["gt"]), "ground_truth.png")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    width, top, bottom = 256, 90, 50
    canvas = Image.new("RGB", (width * 4, width + top + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    title_font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else font
    steps = int(next(iter(state.get("optimizer", {}).get("state", {}).values()), {"step": 0})["step"])
    scope = "smoke" if args.checkpoint.parent.name == "smoke" else "checkpoint"
    scene_tile = args.sample_index % data.per_scene
    sample_label = 'full scene' if config.get('eval_layout')=='full256' else f'tile {scene_tile}'
    draw.text((10, 8), f"LIB-HSI | epoch {state['epoch']} | {scope}: {steps} steps | {args.split}", fill="black", font=title_font)
    draw.text((10, 32), f"{sample['scene']} | {sample_label} (index {args.sample_index})", fill="black", font=font)
    for i, (label, picture, filename) in enumerate(panels):
        picture.save(args.output_dir / filename)
        draw.text((i * width + 9, 64), label, fill="black", font=font)
        canvas.paste(picture.resize((width, width), Image.Resampling.NEAREST), (i * width, top))
    draw.text((10, width + top + 10), "HSI: bands 69/52/18; same GT-based 1-99% stretch for all HSI panels.", fill="black", font=font)
    draw.text((10, width + top + 29), "LR is displayed with nearest-neighbor enlargement; RGB guide uses original color.", fill="black", font=font)
    canvas.save(args.output_dir / "comparison.png")

    def metrics(image):
        error = (image - sample['gt']).square()
        mse = error[:,sample['valid_mask']].mean().item() if 'valid_mask' in sample else error.mean().item()
        return {"mse": mse, "psnr_db_range_1": float(-10 * np.log10(max(mse, 1e-12)))}

    report = {"checkpoint": str(args.checkpoint), "epoch": state["epoch"], "scene": sample["scene"],
              "sample_index": args.sample_index, "patch_size": config["patch_size"],
              "prediction": metrics(prediction), baseline_key: metrics(baseline),
              "split": args.split,
              "scene_tile": scene_tile,
              "eval_layout": config.get('eval_layout','tiles'),
              "area_consistency": args.area_consistency,
              "panel_order": ["LR HSI", "RGB guide", "Prediction", "Ground truth"],
              "notes": "Single sample illustration; all 204 bands for metrics; not a full-split benchmark."}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(args.output_dir / "comparison.png")


if __name__ == "__main__":
    main()
