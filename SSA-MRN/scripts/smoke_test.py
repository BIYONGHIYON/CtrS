"""Run the repaired official model on a resized full QuickBird test sample."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssamrn.data.pancollection import PanCollectionH5
from ssamrn.data.resize import prepare_full_sample
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
from ssamrn.preview import save_ms_rgb_preview, save_pan_preview, save_prediction


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data/raw/QuickBird/test_qb_multiExm1.h5"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "experiments/results/quickbird_smoke"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sensor", default="QB")
    parser.add_argument("--size", "--crop", dest="size", type=int, default=32,
                        help="Output side length from full-sample downsampling; divisible by 4")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0, help="Random initialization seed")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    sample = PanCollectionH5(args.data, sensor=args.sensor)[0]
    try:
        pan, lms, ms = prepare_full_sample(sample, args.size)
    except ValueError as exc:
        parser.error(str(exc))
    pan, lms, ms = (image.to(args.device) for image in (pan, lms, ms))

    model = RestoredPansharpeningNet(channels=lms.shape[1]).to(args.device).eval()
    with torch.inference_mode():
        output = model(pan, lms, ms)
    assert output.shape == lms.shape
    assert torch.isfinite(output).all()
    pan_path = save_pan_preview(pan, args.output_dir, args.size)
    ms_path = save_ms_rgb_preview(ms, args.output_dir, args.size)
    array_path, image_path = save_prediction(output, args.output_dir, args.size)
    print(f"Full sample PAN={tuple(sample['pan'].shape)}, MS={tuple(sample['ms'].shape)}; no crop")
    print(f"forward OK: pan={tuple(pan.shape)}, lms={tuple(lms.shape)}, "
          f"ms={tuple(ms.shape)}, output={tuple(output.shape)}")
    print(f"Random initialization seed: {args.seed}")
    print("Weights are random: this is only a shape/execution check, not a reconstruction result.")
    print(f"Saved 4-band array: {array_path}")
    print(f"Saved output MS RGB preview (bands 3, 2, 1): {image_path}")
    print(f"Saved input MS RGB preview (bands 3, 2, 1): {ms_path}")
    print(f"Saved full-sample PAN input preview: {pan_path}")


if __name__ == "__main__":
    main()
