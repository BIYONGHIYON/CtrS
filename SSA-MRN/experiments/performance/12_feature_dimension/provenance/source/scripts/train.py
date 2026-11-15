"""Train the restored SSA-MRN using separate PanCollection train/validation H5 files."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssamrn.data.pancollection import PanCollectionH5
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet

ROOT = Path(__file__).resolve().parents[1]


def cpu_state(value):
    """Save portable checkpoints from GPU backends, including DirectML."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(item) for item in value)
    return value


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            pan, lms, ms, gt = (batch[key].to(device) for key in ("pan", "lms", "ms", "gt"))
            prediction = model(pan, lms, ms)
            total += torch.nn.functional.mse_loss(prediction, gt, reduction="sum").item()
            count += gt.numel()
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--sensor", default="QB")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssai-dimension", type=int, default=6,
                        help="SSAI K (default: 6; use 4 for the upstream baseline)")
    parser.add_argument("--seed", type=int, default=42, help="Working seed; paper seed unconfirmed")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda", "dml"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="Defaults to an ignored local directory for this K and sensor")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.train.resolve() == args.val.resolve():
        parser.error("training and validation must use different H5 files")
    if not args.train.is_file() or not args.val.is_file():
        parser.error("both --train and --val must exist")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.ssai_dimension < 1:
        parser.error("epochs, batch-size, lr, and ssai-dimension must be positive")
    if args.checkpoint_dir is None:
        args.checkpoint_dir = (ROOT / "experiments/checkpoints/local"
                               / f"k{args.ssai_dimension}" / f"{args.sensor.lower()}_full")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")
    if args.device == "dml":
        try:
            import torch_directml
        except ImportError:
            parser.error("Install torch-directml to use --device dml")
        device = torch_directml.device()
    else:
        device = args.device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_set = PanCollectionH5(args.train, args.sensor, limit=args.max_train_samples)
    val_set = PanCollectionH5(args.val, args.sensor, limit=args.max_val_samples)
    if train_set.channels != val_set.channels:
        parser.error("training and validation channel counts differ")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RestoredPansharpeningNet(
        channels=train_set.channels, ssai_dimension=args.ssai_dimension
    )
    if args.device == "dml":
        model.use_directml_prelu()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["sensor"] != args.sensor.upper() or checkpoint["channels"] != train_set.channels:
            parser.error("checkpoint sensor/channels do not match the dataset")
        if checkpoint.get("ssai_dimension", 4) != args.ssai_dimension:
            parser.error("checkpoint SSAI dimension does not match --ssai-dimension")
        if checkpoint.get("learning_rate", args.lr) != args.lr or checkpoint.get("batch_size", args.batch_size) != args.batch_size:
            parser.error("checkpoint learning rate/batch size do not match this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            pan, lms, ms, gt = (batch[key].to(device) for key in ("pan", "lms", "ms", "gt"))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(pan, lms, ms), gt)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_mse = evaluate(model, val_loader, device)
        print(f"epoch={epoch + 1} train_mse={np.mean(losses):.8f} val_mse={val_mse:.8f}", flush=True)
        checkpoint = {"epoch": epoch + 1, "sensor": args.sensor.upper(),
                      "channels": train_set.channels, "model": cpu_state(model.state_dict()),
                      "optimizer": cpu_state(optimizer.state_dict()), "seed": args.seed,
                      "ssai_dimension": args.ssai_dimension, "learning_rate": args.lr,
                      "batch_size": args.batch_size,
                      "torch_rng_state": torch.get_rng_state()}
        torch.save(checkpoint, args.checkpoint_dir / "latest.pt")


if __name__ == "__main__":
    main()
