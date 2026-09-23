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


def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
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
    parser.add_argument("--seed", type=int, default=42, help="Working seed; paper seed unconfirmed")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "experiments/checkpoints")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.train.resolve() == args.val.resolve():
        parser.error("training and validation must use different H5 files")
    if not args.train.is_file() or not args.val.is_file():
        parser.error("both --train and --val must exist")
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_set = PanCollectionH5(args.train, args.sensor, limit=args.max_train_samples)
    val_set = PanCollectionH5(args.val, args.sensor)
    if train_set.channels != val_set.channels:
        parser.error("training and validation channel counts differ")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    model = RestoredPansharpeningNet(channels=train_set.channels).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=True)
        if checkpoint["sensor"] != args.sensor.upper() or checkpoint["channels"] != train_set.channels:
            parser.error("checkpoint sensor/channels do not match the dataset")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            pan, lms, ms, gt = (batch[key].to(args.device) for key in ("pan", "lms", "ms", "gt"))
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(pan, lms, ms), gt)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_mse = evaluate(model, val_loader, args.device)
        print(f"epoch={epoch + 1} train_mse={np.mean(losses):.8f} val_mse={val_mse:.8f}", flush=True)
        checkpoint = {"epoch": epoch + 1, "sensor": args.sensor.upper(),
                      "channels": train_set.channels, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "seed": args.seed}
        torch.save(checkpoint, args.checkpoint_dir / "latest.pt")


if __name__ == "__main__":
    main()
