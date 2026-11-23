"""Train ADWM LAGConv on one PanCollection sensor."""
import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from upstream.model import bwnet_lagconv_sota as official

BATCH = 64
SEED = 42
SENSORS = {
    "qb": ("QuickBird", 2047, 200, 17139, 1905),
    "gf2": ("Gaofen 2", 1023, 500, 19809, 2201),
}


class Patches(Dataset):
    def __init__(self, path, count, scale):
        with h5py.File(path, "r") as data:
            self.arrays = []
            for key, bands in (("ms", 4), ("pan", 1), ("gt", 4)):
                field = data[key]
                if field.ndim != 4 or field.shape[:2] != (count, bands):
                    raise ValueError(f"{path} {key}: unexpected shape {field.shape}")
                self.arrays.append(torch.from_numpy(np.asarray(field, dtype=np.float32)))
        ms, pan, gt = self.arrays
        if gt.shape[2:] != pan.shape[2:] or tuple(4 * x for x in ms.shape[2:]) != gt.shape[2:]:
            raise ValueError(f"{path}: inconsistent spatial dimensions")
        for array in self.arrays:
            array.div_(scale)

    def __len__(self):
        return len(self.arrays[0])

    def __getitem__(self, index):
        return tuple(array[index] for array in self.arrays)


def save_atomic(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=SENSORS, required=True)
    parser.add_argument("--data", type=Path, required=True, help="PanCollection root")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    folder, scale, epochs, train_count, valid_count = SENSORS[args.sensor]
    data = args.data / folder / "Training Dataset"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    model = official.BWNet(ms_dim=4)
    if args.sensor == "qb":
        blocks = model.bw_output.CovBlockList
        if len(blocks) != 6:
            raise ValueError("Unexpected ADWM covariance block count")
        for index in range(len(blocks)):
            blocks[index] = official.CovBlock(48, 1, round(48 * 0.6), 0.05)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "initial_lr": 0.002}], lr=0.002, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=150, gamma=0.5, last_epoch=0
    )
    criterion = nn.L1Loss()
    training = DataLoader(
        Patches(data / f"train_{args.sensor}.h5", train_count, scale),
        batch_size=BATCH, shuffle=True, drop_last=True,
    )
    validation = DataLoader(
        Patches(data / f"valid_{args.sensor}.h5", valid_count, scale),
        batch_size=BATCH, shuffle=False,
    )
    args.out.mkdir(parents=True, exist_ok=False)
    save_atomic(args.out / "initial.pt", {"model": model.state_dict()})
    best = float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        lr = optimizer.param_groups[0]["lr"]
        for ms, pan, gt in training:
            optimizer.zero_grad()
            loss = criterion(model(ms.to(device), pan.to(device)), gt.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()

        model.eval()
        valid_total = 0.0
        with torch.inference_mode():
            for ms, pan, gt in validation:
                prediction = model(ms.to(device), pan.to(device))
                valid_total += criterion(prediction, gt.to(device)).item() * len(gt)
        valid_l1 = valid_total / valid_count
        if not np.isfinite(valid_l1) or not np.isfinite(total):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")
        row = dict(epoch=epoch, lr=lr, train_l1=total / len(training), valid_l1=valid_l1)
        history.append(row)
        if valid_l1 < best:
            best = valid_l1
            save_atomic(args.out / "best.pt", {"model": model.state_dict(), **row})
        save_atomic(args.out / "latest.pt", {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device), "history": history,
        })
        with (args.out / "history.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
        print(f"{args.sensor.upper()} {epoch}/{epochs} {row}", flush=True)


if __name__ == "__main__":
    main()
