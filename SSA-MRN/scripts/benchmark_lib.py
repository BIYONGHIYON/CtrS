"""Measure isolated real-data I/O, preprocessing and CUDA optimizer throughput."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SSA-MRN/src"))
from ssamrn.data.lib_hsi import LIBHSI
from ssamrn.models.rgb_hsi import RGBHSISsaMRN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    config = json.loads((ROOT / "SSA-MRN/configs/lib_rgb_hsi.json").read_text(encoding="utf-8"))
    torch.set_num_threads(6)
    dataset = LIBHSI(config["data_root"], "train", limit=3)
    start = time.perf_counter()
    sample = dataset[0]
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for i in (1, 2, 3):
        dataset[i]
    cached_seconds = (time.perf_counter() - start) / 3
    results = {"first_scene_plus_crop_seconds": first_seconds,
               "cached_crop_cpu_degradation_seconds": cached_seconds, "cuda": []}
    print(json.dumps({k:v for k,v in results.items() if k != "cuda"}), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    for batch_size, vectorized, channels_last in ((1, False, False), (4, False, False),
                                                 (4, True, False), (8, True, False),
                                                 (4, True, True), (8, True, True)):
        torch.manual_seed(42)
        model = RGBHSISsaMRN(204, config["latent_channels"], config["ssai_dimension"], vectorized).cuda()
        rgb = sample["rgb"].unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()
        lr = sample["lr_hsi"].unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()
        gt = sample["gt"].unsqueeze(0).repeat(batch_size, 1, 1, 1).cuda()
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
            rgb, lr, gt = (x.contiguous(memory_format=torch.channels_last) for x in (rgb, lr, gt))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, fused=vectorized)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.reset_peak_memory_stats()
        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = F.mse_loss(model(rgb, lr).float(), gt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
            scaler.step(optimizer)
            scaler.update()
            if not vectorized:
                loss.item()  # Existing per-step sync/logging path.
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        result = {"batch_size": batch_size, "vectorized_ssa": vectorized, "channels_last": channels_last,
                  "seconds_per_batch": seconds / args.steps, "patches_per_second": batch_size * args.steps / seconds,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
        results["cuda"].append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer, scaler, rgb, lr, gt
        torch.cuda.empty_cache()
    output = ROOT / "SSA-MRN/experiments/results/lib_optimization"
    output.mkdir(parents=True, exist_ok=True)
    (output / "microbenchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
