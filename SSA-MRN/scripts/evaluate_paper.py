"""Evaluate all PanCollection examples against SSA-MRN Tables I-IV."""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ssamrn.data.pancollection import SENSOR_MAX  # noqa: E402
from ssamrn.metrics import fr_metrics, rr_metrics  # noqa: E402
from ssamrn.models.ssa_mrn import RestoredPansharpeningNet  # noqa: E402

SENSORS = {
    "QB": ("QuickBird", "qb", "qb_full"),
    "GF2": ("Gaofen2", "gf2", "gf2_full"),
    "WV3": ("WorldView3", "wv3", "wv3_full"),
    "WV2": ("WorldView2", "wv2", "wv3_full"),  # paper's WV3 -> WV2 transfer
}
ARCHIVE_FOLDERS = {"QB": "QuickBird", "GF2": "Gaofen 2", "WV3": "WorldView 3", "WV2": "WorldView 2"}
# SSA-MRN paper Tables I-IV: Ours. Order SAM, ERGAS, PSNR, SCC, Q2n, D_lambda, D_s, QNR.
PAPER = {
    "QB": (4.8478, 4.0726, 37.6645, .9702, .9243, .0341, .0360, .9311),
    "GF2": (.9434, .8755, 46.9734, .9898, .9688, .0395, .0487, .9137),
    "WV3": (3.4873, 2.5866, 37.5691, .9735, .8930, .0329, .0617, .9077),
    "WV2": (5.8845, 4.7254, 29.4148, .9217, .8215, .0657, .0549, .8831),
}
METRIC_KEYS = ("SAM", "ERGAS", "PSNR", "SCC", "Q2n", "D_lambda", "D_s", "QNR")


def evaluate(sensor, protocol, limit, checkpoint_root=ROOT / "experiments/checkpoints/k6",
             data_root=ROOT / "data/raw", device=torch.device("cpu")):
    folder, stem, ckpt_dir = SENSORS[sensor]
    filename = f"test_{stem}{'_OrigScale' if protocol == 'FR' else ''}_multiExm1.h5"
    archive_section = ("Testing Dataset (FullData, H5 Format)" if protocol == "FR"
                       else "Testing Dataset (ReducedData, H5 Format)")
    candidates = (data_root / folder / filename,
                  data_root / ARCHIVE_FOLDERS[sensor] / archive_section / filename)
    data_file = next((path for path in candidates if path.is_file()), None)
    if data_file is None:
        raise FileNotFoundError(f"Missing {sensor} {protocol} H5; checked {candidates}")
    checkpoint_file = checkpoint_root / ckpt_dir / "latest.pt"
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    expected_checkpoint_sensor = "WV3" if sensor == "WV2" else sensor
    if checkpoint["sensor"] != expected_checkpoint_sensor:
        raise ValueError(f"Wrong checkpoint for {sensor}")
    model = RestoredPansharpeningNet(channels=checkpoint["channels"],
                                    ssai_dimension=checkpoint.get("ssai_dimension", 4))
    if device.type == "privateuseone":
        model.use_directml_prelu()
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    scale = SENSOR_MAX[sensor]
    rows = []
    started = time.monotonic()
    with h5py.File(data_file, "r") as h5:
        count = min(len(h5["pan"]), limit or len(h5["pan"]))
        required = {"pan", "lms", "ms"} | ({"gt"} if protocol == "RR" else set())
        if set(h5.keys()) != required:
            raise ValueError(f"Unexpected {protocol} H5 keys: {set(h5.keys())}")
        for index in range(count):
            arrays = {key: np.asarray(h5[key][index], dtype=np.float32) for key in required}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError(f"Non-finite source at {sensor} {protocol} {index}")
            inputs = [torch.from_numpy(arrays[key] / scale).unsqueeze(0).to(device)
                      for key in ("pan", "lms", "ms")]
            with torch.no_grad():
                fused = model(*inputs)[0].cpu().numpy().astype(np.float64) * scale
            if protocol == "RR":
                metrics = rr_metrics(arrays["gt"].astype(np.float64), fused)
            else:
                metrics = fr_metrics(fused, arrays["lms"], arrays["ms"], arrays["pan"])
            if not all(np.isfinite(value) for value in metrics.values()):
                raise ValueError(f"Non-finite metric at {sensor} {protocol} {index}: {metrics}")
            rows.append({"index": index, **metrics})
            print(f"{sensor} {protocol} {index + 1}/{count}", flush=True)
    keys = tuple(key for key in METRIC_KEYS if key in rows[0])
    averages = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    paper = {key: PAPER[sensor][METRIC_KEYS.index(key)] for key in keys}
    return {"sensor": sensor, "protocol": protocol, "samples": count,
            "checkpoint": str(checkpoint_file), "checkpoint_epoch": checkpoint["epoch"],
            "ssai_dimension": checkpoint.get("ssai_dimension", 4),
            "data_file": str(data_file), "elapsed_seconds": time.monotonic() - started,
            "metrics_mean": averages, "paper": paper,
            "delta_ours_minus_paper": {key: averages[key] - paper[key] for key in keys},
            "per_sample": rows,
            "parity": "PSNR uses per-band mean at peak 2047, selected because the paper does not disclose its calculation and this is numerically closest; it is not verified as the authors' protocol. FR uses scikit-image antialiased cubic PAN resize, not MATLAB imresize; D_s and QNR are provisional. RR Q2n port and SCC border handling have not yet been cross-tested against MATLAB."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=(*SENSORS, "all"), default="all")
    parser.add_argument("--protocol", choices=("RR", "FR", "both"), default="both")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/results/local_k6_eval")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "experiments/checkpoints/k6")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--device", choices=("cpu", "cuda", "dml"), default="cpu")
    args = parser.parse_args()
    if args.device == "dml":
        try:
            import torch_directml
        except ImportError:
            parser.error("Install torch-directml to use --device dml")
        device = torch_directml.device()
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this PyTorch environment")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.set_num_threads(4)
    sensors = SENSORS if args.sensor == "all" else (args.sensor,)
    protocols = ("RR", "FR") if args.protocol == "both" else (args.protocol,)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sensor in sensors:
        for protocol in protocols:
            report = evaluate(sensor, protocol, args.max_samples, args.checkpoint_root,
                              args.data_root, device)
            target = args.output_dir / f"{sensor.lower()}_{protocol.lower()}.json"
            target.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Saved {target}: {report['metrics_mean']}", flush=True)


if __name__ == "__main__":
    main()
