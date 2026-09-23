"""Run the repaired official model on one aligned QuickBird test crop."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssamrn.data.pancollection import PanCollectionH5
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet
from ssamrn.preview import save_prediction


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data/raw/QuickBird/test_qb_multiExm1.h5"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "experiments/results/quickbird_smoke"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--sensor", default="QB")
    parser.add_argument("--crop", type=int, default=32, help="HR crop width; must be divisible by 4")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0, help="Random initialization seed")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.crop < 4 or args.crop % 4:
        parser.error("--crop must be >= 4 and divisible by 4")

    sample = PanCollectionH5(args.data, sensor=args.sensor)[0]
    if args.crop > min(sample["pan"].shape[-2:]):
        parser.error("--crop is larger than the data")
    pan = sample["pan"][:, : args.crop, : args.crop].unsqueeze(0).to(args.device)
    lms = sample["lms"][:, : args.crop, : args.crop].unsqueeze(0).to(args.device)
    ms = sample["ms"][:, : args.crop // 4, : args.crop // 4].unsqueeze(0).to(args.device)

    model = RestoredPansharpeningNet(channels=lms.shape[1]).to(args.device).eval()
    with torch.inference_mode():
        output = model(pan, lms, ms)
    assert output.shape == lms.shape
    assert torch.isfinite(output).all()
    array_path, image_path = save_prediction(output, args.output_dir, args.crop)
    print(f"forward OK: pan={tuple(pan.shape)}, lms={tuple(lms.shape)}, "
          f"ms={tuple(ms.shape)}, output={tuple(output.shape)}")
    print(f"Random initialization seed: {args.seed}")
    print("Weights are random: this is only a shape/execution check, not a reconstruction result.")
    print(f"Saved 4-band array: {array_path}")
    print(f"Saved display-only preview (first 3 bands): {image_path}")


if __name__ == "__main__":
    main()
