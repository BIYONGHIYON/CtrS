"""Windows/local RGB-HSI feasibility experiment. Run with --smoke before training."""
import argparse
import json
import math
import random
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "SSA-MRN/src"))
from ssamrn.data.lib_hsi import LIBHSI, ScenePatchSampler
from ssamrn.losses import reconstruction_loss
from ssamrn.models.rgb_grouped import build_rgb_hsi_model
from ssamrn.models.interp23 import interp23tap
from ssamrn.models.area_consistency import project_area_consistency


def save_checkpoint(state, path):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def worker_init(worker_id):
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def move_batch(batch, device, channels_last=False):
    rgb, gt = (batch[key].to(device, non_blocking=True) for key in ("rgb", "gt"))
    if "lr_hsi" in batch:
        lr = batch["lr_hsi"].to(device, non_blocking=True)
    else:
        # Run outside autocast: preserve float32 antialiased degradation.
        mode=batch.get('degradation_mode','bicubic')
        mode=mode[0] if isinstance(mode,(list,tuple)) else mode
        if mode=='area':
            lr=F.avg_pool2d(gt,4)
        elif mode=='bicubic':
            lr = F.interpolate(gt, scale_factor=0.25, mode="bicubic", align_corners=False,
                           antialias=True).clamp(0, 1)
        else:
            raise ValueError('Unknown degradation mode: '+mode)
    if channels_last:
        rgb, lr, gt = (x.contiguous(memory_format=torch.channels_last) for x in (rgb, lr, gt))
    return rgb, lr, gt


class DevicePrefetch:
    """Overlap pinned host transfers/degradation with work already queued on CUDA."""
    _streams = {}

    def __init__(self, loader, device, channels_last=False):
        self.loader, self.device, self.channels_last = loader, device, channels_last
        index = device.index if device.index is not None else torch.cuda.current_device()
        # CUDA allocator pools are tied to the creation stream. Reuse one transfer
        # stream per device across both train/validation iterators and all epochs.
        if index not in self._streams:
            self._streams[index] = torch.cuda.Stream(device=device)
        self.stream = self._streams[index]

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        stream = self.stream
        for host in self.loader:
            with torch.cuda.stream(stream):
                rgb, lr, gt = move_batch(host, self.device, self.channels_last)
            current = torch.cuda.current_stream(self.device)
            current.wait_stream(stream)
            for tensor in (rgb, lr, gt):
                tensor.record_stream(current)
            batch = {"rgb": rgb, "lr_hsi": lr, "gt": gt, "scene": host["scene"]}
            if 'valid_mask' in host:
                batch['valid_mask']=host['valid_mask'].to(self.device,non_blocking=True)
            if "source_cube_bytes" in host:
                batch["source_cube_bytes"] = host["source_cube_bytes"]
            yield batch


def preview(batch, prediction, baseline, path):
    bands = [69, 52, 18]
    gt = batch["gt"][0, bands].detach().float().cpu()
    lo = torch.quantile(gt.flatten(1), .01, dim=1)[:, None, None]
    hi = torch.quantile(gt.flatten(1), .99, dim=1)[:, None, None]
    lr = batch.get("lr_hsi")
    if lr is None:
        lr = F.interpolate(batch["gt"], scale_factor=.25, mode="area")
    tensors = [lr[0, bands], batch["rgb"][0], prediction[0, bands], gt]
    pictures = []
    for index, tensor in enumerate(tensors):
        tensor = tensor.detach().float().cpu()
        if index != 1:
            tensor = (tensor-lo)/(hi-lo).clamp_min(1e-8)
        array = (tensor.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
        picture = Image.fromarray(array).resize((gt.shape[-1],gt.shape[-2]),Image.Resampling.NEAREST)
        pictures.append(np.asarray(picture))
    Image.fromarray(np.concatenate(pictures, axis=1)).save(path)


@torch.no_grad()
def evaluate(model, loader, device, amp, output, max_batches=None, channels_last=False,
             baseline_mode='bicubic', degradation_mode='bicubic', area_consistency=False):
    if baseline_mode not in ('bicubic', '23tap'):
        raise ValueError('Unknown baseline interpolation')
    if area_consistency and degradation_mode != 'area':
        raise ValueError('Area consistency requires area degradation')
    baseline_label = 'interp23' if baseline_mode == '23tap' else 'bicubic'
    model.eval()
    totals = {}
    read_bytes = 0
    finite = torch.ones((), dtype=torch.bool, device=device)
    batches = DevicePrefetch(loader, device, channels_last) if device.type == "cuda" else loader
    for step, batch in enumerate(tqdm(batches, desc="validation", leave=False, mininterval=2.)):
        if "source_cube_bytes" in batch:
            read_bytes += int(batch["source_cube_bytes"].sum())
        rgb, lr, gt = move_batch(batch, device, channels_last)
        with torch.autocast(device.type, enabled=amp):
            prediction = model(rgb, lr)
        prediction = prediction.float()  # Unclipped model metrics, data range=1.
        baseline = interp23tap(lr,4) if baseline_mode=='23tap' else F.interpolate(lr, size=gt.shape[-2:], mode="bicubic", align_corners=False)
        finite.logical_and_(torch.isfinite(prediction).all())
        valid = torch.linalg.vector_norm(gt, dim=1) > 1e-6
        spatial = batch.get('valid_mask', torch.ones_like(valid)).to(device)
        if area_consistency:
            prediction = project_area_consistency(prediction, lr, spatial)
            baseline = project_area_consistency(baseline, lr, spatial)
        valid = valid & spatial
        gt_norm = torch.linalg.vector_norm(gt, dim=1)
        values = []
        for image in (prediction, baseline):
            sse = ((image - gt).square()*spatial[:,None]).sum(dim=(1, 2, 3))
            cosine = (image * gt).sum(1) / (torch.linalg.vector_norm(image, dim=1) * gt_norm).clamp_min(1e-12)
            sam = (torch.rad2deg(torch.acos(cosine.clamp(-1, 1))) * valid).sum(dim=(1, 2))
            values.extend((sse, sam))
        values.append(valid.sum(dim=(1, 2)))
        packed = torch.stack(values, dim=1)
        for i, scene in enumerate(batch["scene"]):
            if scene not in totals:
                totals[scene] = {"values": torch.zeros(5, dtype=torch.float64, device=device), "count": 0}
            totals[scene]["values"].add_(packed[i])
            totals[scene]["count"] += int(spatial[i].sum())*gt.shape[1] if 'valid_mask' in batch else gt[i].numel()
        if step == 0:
            preview(batch, prediction, baseline, output / "preview_inputs_prediction_gt.png")
        if max_batches is not None and step + 1 >= max_batches:
            break
    if not finite.item():
        raise RuntimeError("Nonfinite validation prediction")
    scenes = {}
    cpu_values = torch.stack([record["values"] for record in totals.values()]).cpu().tolist()
    for (scene, record), numbers in zip(totals.items(), cpu_values):
        scenes[scene] = {}
        for offset, label in ((0, "model"), (2, baseline_label)):
            mse = numbers[offset] / record["count"]
            scenes[scene][label + "_mse"] = mse
            scenes[scene][label + "_psnr_db"] = -10 * math.log10(max(mse, 1e-12))
            scenes[scene][label + "_sam_deg"] = numbers[offset+1] / numbers[4] if numbers[4] else None
    summary = {key: float(np.mean([r[key] for r in scenes.values() if r[key] is not None]))
               for key in next(iter(scenes.values()))
               if any(r[key] is not None for r in scenes.values())}
    return {"scene_mean": summary, "scenes": scenes, "partial": max_batches is not None,
            "alignment_border_masked": 'valid_mask' in batch,
            "source_cube_requested_gib": read_bytes / 2**30,
            "baseline_mode": baseline_mode,
            "area_consistency": area_consistency,
            "protocol": f"x4 {degradation_mode} LR; {baseline_label} baseline; clipped [0,1] reflectance GT; unclipped output; "
                        f"area consistency {'on' if area_consistency else 'off'}; "
                        "PSNR from all-band scene MSE (range=1); SAM ignores zero GT pixels"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "SSA-MRN/configs/lib_rgb_hsi_triple17_k4_spectral_tiles.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--alignment-manifest", type=Path, help="Exact registration file required by the checkpoint")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Two real-data optimizer steps, partial validation")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate test split, no training")
    parser.add_argument("--area-consistency", action="store_true", help="Project evaluated x4 area outputs onto LR HSI")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-scenes", type=int, help="Limited benchmark, not a full training run")
    parser.add_argument("--max-val-scenes", type=int, help="Limited benchmark, not a full validation run")
    parser.add_argument("--profile", action="store_true", help="Record GPU event timing and loader waits")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.alignment_manifest:
        config['alignment_manifest'] = str(args.alignment_manifest.resolve())
    if config.get('alignment_manifest'):
        config['alignment_sha256']=hashlib.sha256((ROOT/config['alignment_manifest']).read_bytes()).hexdigest()
    if args.data_root:
        config["data_root"] = str(args.data_root)
    if args.device:
        config["device"] = args.device
    config['data_root'] = str(ROOT / config['data_root'])
    if args.output_dir:
        config["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.smoke and args.evaluate:
        parser.error("--smoke and --evaluate are mutually exclusive")
    if args.area_consistency and (not args.evaluate or config.get('degradation') != 'area'):
        parser.error('--area-consistency requires --evaluate with area degradation')
    if args.smoke and args.resume:
        parser.error("smoke must not resume a full training run")
    for key in ("epochs", "batch_size", "accumulation_steps", "patches_per_scene", "latent_channels", "ssai_dimension"):
        if config[key] < 1:
            parser.error(f"{key} must be positive")
    if config["learning_rate"] <= 0 or config["workers"] < 0:
        parser.error("Invalid learning_rate or workers")
    for key in ("val_batch_size", "cpu_threads", "prefetch_factor", "log_interval"):
        if config.get(key, 1) < 1:
            parser.error(f"{key} must be positive")
    if config.get("spectral_weight", 0.) < 0 or config.get("spectral_eps", 1e-6) <= 0:
        parser.error("Invalid spectral loss weight or norm threshold")
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable. Select a CUDA-enabled Python interpreter or pass --device cpu")
    amp = config["amp"] and device.type == "cuda"
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 6))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = config.get("cudnn_benchmark", False)
    output = ROOT / config["output_dir"]
    if args.smoke:
        output = output / "smoke"
    if args.evaluate:
        output = output / "test"
    if not args.smoke and not args.evaluate and not args.resume and (output / "latest.pt").exists():
        parser.error("Existing training run: use --resume or choose a new --output-dir")
    output.mkdir(parents=True, exist_ok=True)
    channels_last = config.get("channels_last", False)
    gpu_degradation = config.get("gpu_degradation", False) and device.type == "cuda"
    dataset_args = dict(root=config["data_root"], patch_size=config["patch_size"],
                        patches_per_scene=config["patches_per_scene"],
                        compute_lr=not gpu_degradation, crop_seed=config["seed"],
                        read_mode=config.get("read_mode", "whole"),
                        alignment_manifest=(ROOT/config['alignment_manifest']) if config.get('alignment_manifest') else None,
                        train_layout=config.get('train_layout','random'),eval_layout=config.get('eval_layout','tiles'),
                        degradation=config.get('degradation','bicubic'))
    validation = LIBHSI(split="test" if args.evaluate else "validation", **dataset_args,
                        limit=1 if args.smoke else args.max_val_scenes, allow_incomplete=args.smoke)
    loader_args = dict(batch_size=config["batch_size"], num_workers=config["workers"],
                       pin_memory=device.type == "cuda")
    if config["workers"]:
        loader_args.update(persistent_workers=True, prefetch_factor=config.get("prefetch_factor", 2),
                           worker_init_fn=worker_init)
    val_loader = DataLoader(validation, shuffle=False,
                            **dict(loader_args, batch_size=config.get("val_batch_size", config["batch_size"])))
    model = build_rgb_hsi_model(config).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 fused=config.get("fused_adam", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best = 0, float("inf")
    checkpoint_path = args.resume or args.checkpoint
    if args.evaluate and not checkpoint_path:
        checkpoint_path = ROOT / config["output_dir"] / "best.pt"
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key,default in [('model_type','latent_rgb'),('upsampler','bicubic'),('train_layout','random'),('eval_layout','tiles'),('degradation','bicubic')]:
            if state['config'].get(key,default)!=config.get(key,default):
                parser.error('Checkpoint protocol differs: '+key)
        if state['config'].get('alignment_sha256') != config.get('alignment_sha256'):
            parser.error('Alignment protocol changed: use a new training run and its checkpoint')
        for key in ("patch_size", "latent_channels", "ssai_dimension"):
            if state["config"][key] != config[key]:
                parser.error(f"Checkpoint {key} differs from config")
        model.load_state_dict(state["model"])
        if args.resume:
            if state['config'].get('alignment_manifest') != config.get('alignment_manifest'):
                parser.error('Alignment protocol changed: start a new run rather than resume')
            for key in ("batch_size", "accumulation_steps", "learning_rate", "patches_per_scene", "seed", "amp", "device", "workers"):
                if state["config"][key] != config[key]:
                    parser.error(f"Resume {key} differs from checkpoint")
            for key, default in [("spectral_weight", 0.), ("spectral_eps", 1e-6)]:
                if state["config"].get(key, default) != config.get(key, default):
                    parser.error("Resume loss differs: " + key)
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start_epoch, best = state["epoch"], state["best_mse"]
            torch.set_rng_state(state["rng"])
            if device.type == "cuda" and state.get("cuda_rng"):
                torch.cuda.set_rng_state_all(state["cuda_rng"])
    (output / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"device={device} torch={torch.__version__} amp={amp} parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    print(f"data={config['data_root']} patch={config['patch_size']} RGB=3 HSI=204 latent={config['latent_channels']}", flush=True)
    print(f"model={config.get('model_type','latent_rgb')} K={config['ssai_dimension']} train_layout={config.get('train_layout','random')} eval_layout={config.get('eval_layout','tiles')} degradation={config.get('degradation','bicubic')} upsampler={config.get('upsampler','bicubic')}", flush=True)
    if args.evaluate:
        metrics = evaluate(model, val_loader, device, amp, output, channels_last=channels_last,
                           baseline_mode=config.get('upsampler','bicubic'), degradation_mode=config.get('degradation','bicubic'),
                           area_consistency=args.area_consistency)
        metrics['eval_layout']=config.get('eval_layout','tiles')
        metrics['limited_scenes']=args.max_val_scenes is not None
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics["scene_mean"], indent=2))
        return
    training = LIBHSI(split="train", **dataset_args, limit=2 if args.smoke else args.max_train_scenes,
                      allow_incomplete=args.smoke)
    if {p.stem for p in training.files} & {p.stem for p in validation.files}:
        raise ValueError("Train/validation scene names overlap")
    train_loader = DataLoader(training, sampler=ScenePatchSampler(training), **loader_args)
    epochs = start_epoch + 1 if args.smoke else config["epochs"]
    accumulation = 1 if args.smoke else config["accumulation_steps"]
    steps = min(2, len(train_loader)) if args.smoke else len(train_loader)
    print(f"train_scenes={len(training.files)} val_scenes={len(validation.files)} steps/epoch={steps} effective_batch={config['batch_size'] * accumulation}", flush=True)
    cores = list(model.cores) if hasattr(model, "cores") else [model.core]
    vectorized_ssa = all(core.vectorized for core in cores)
    print(f"workers={config['workers']} gpu_degradation={gpu_degradation} vectorized_ssa={vectorized_ssa} cores={len(cores)} channels_last={channels_last}", flush=True)
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loader.sampler.set_epoch(epoch)
        total = torch.zeros((), device=device)
        finite = torch.ones((), dtype=torch.bool, device=device)
        spectral_total = torch.zeros((), device=device)
        loss_total = torch.zeros((), device=device)
        count, data_wait, timings, read_bytes = 0, 0., [], 0
        batches = DevicePrefetch(train_loader, device, channels_last) if device.type == "cuda" else train_loader
        progress = tqdm(batches, total=steps, desc=f"epoch {epoch+1}/{epochs}", mininterval=2.)
        last_step_finished = time.perf_counter()
        for step, batch in enumerate(progress):
            if "source_cube_bytes" in batch:
                read_bytes += int(batch["source_cube_bytes"].sum())
            data_wait += time.perf_counter() - last_step_finished
            if args.profile and device.type == "cuda":
                event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                event_start.record()
            rgb, lr, gt = move_batch(batch, device, channels_last)
            # Normalize the final incomplete accumulation group correctly.
            group_size = min(accumulation, steps - (step // accumulation) * accumulation)
            with torch.autocast(device.type, enabled=amp):
                prediction = model(rgb, lr)
            loss, mse_loss, spectral_loss = reconstruction_loss(
                prediction, gt, batch.get('valid_mask', None).to(device) if 'valid_mask' in batch else None,
                config.get('spectral_weight', 0.), config.get('spectral_eps', 1e-6))
            finite.logical_and_(torch.isfinite(loss))
            scaler.scale(loss / group_size).backward()
            if (step + 1) % accumulation == 0 or step + 1 == steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=device.type == "cuda")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total.add_(mse_loss.detach() * gt.shape[0])
            spectral_total.add_(spectral_loss.detach() * gt.shape[0])
            loss_total.add_(loss.detach() * gt.shape[0])
            count += gt.shape[0]
            if args.profile and device.type == "cuda":
                event_end.record()
                timings.append((event_start, event_end))
            if (step+1) % config.get("log_interval", 20) == 0 or step+1 == steps:
                progress.set_postfix(mse=f"{mse_loss.item():.6f}", spectral=f"{spectral_loss.item():.6f}", refresh=False)
            last_step_finished = time.perf_counter()
            if step + 1 >= steps:
                break
        if not finite.item():
            raise RuntimeError("Nonfinite training loss")
        train_seconds = time.perf_counter() - started
        validation_started = time.perf_counter()
        metrics = evaluate(model, val_loader, device, amp, output, max_batches=2 if args.smoke else None,
                           channels_last=channels_last,baseline_mode=config.get('upsampler','bicubic'),
                           degradation_mode=config.get('degradation','bicubic'))
        metrics['eval_layout']=config.get('eval_layout','tiles')
        metrics["limited_scenes"] = args.smoke or args.max_val_scenes is not None
        val_mse = metrics["scene_mean"]["model_mse"]
        improved = val_mse < best
        best = min(best, val_mse)
        record = {"epoch": epoch+1, "train_mse": total.item() / count, "train_loss": loss_total.item() / count, "train_spectral": spectral_total.item() / count,
                  "spectral_weight": config.get("spectral_weight", 0.), "seconds": time.perf_counter()-started,
                  "train_seconds": train_seconds, "validation_seconds": time.perf_counter()-validation_started,
                  "loader_wait_seconds": data_wait, "train_patches_per_second": count / train_seconds,
                  "train_cube_requested_gib": read_bytes / 2**30,
                  "validation_cube_requested_gib": metrics["source_cube_requested_gib"],
                  "data_root": config["data_root"], "limited_scenes": args.smoke or args.max_train_scenes is not None,
                  **metrics["scene_mean"]}
        if timings:
            record["train_gpu_stream_seconds"] = sum(first.elapsed_time(last) for first, last in timings) / 1000
        if device.type == "cuda":
            record["peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
            record["peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
            record['current_allocated_mb'] = torch.cuda.memory_allocated() / 2**20
            record['current_reserved_mb'] = torch.cuda.memory_reserved() / 2**20
        with (output / "history.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        state = {"epoch": epoch+1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "best_mse": best, "config": config,
                 "rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_checkpoint(state, output / "latest.pt")
        if improved:
            save_checkpoint(state, output / "best.pt")
        print(json.dumps(record), flush=True)
    print(f"Finished. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
