"""Evaluate the best spring/winter checkpoint and export evidence."""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from argparse import Namespace
import numpy as np
import torch
from PIL import Image, ImageDraw

OFFICIAL = Path(__file__).resolve().parents[2] / "Official_ECRformer"
sys.path.insert(0, str(OFFICIAL))
import test
from data import find_dataset_using_name
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

def plain(value):
    if isinstance(value, Namespace):
        return {k: plain(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [plain(v) for v in value]
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=str(Path(__file__).parent))
    args = parser.parse_args()
    ckpt = Path(args.checkpoint).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / season / "summary.json").exists() for season in ("spring", "winter")):
        raise FileExistsError("Use a new --output directory to preserve previous evaluations.")
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]["config"]
    roots = dict(x.split("=",1) for x in config.dataset.root.split(";"))
    manifest_path = ckpt.parent.parent / "train_subset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dump(out / "training_config.json", plain(config))
    dump(out / "provenance.json", {
        "checkpoint": str(ckpt), "checkpoint_sha256": digest(ckpt),
        "best_epoch": checkpoint["epoch"], "global_step": checkpoint["global_step"],
        "manifest_sha256": digest(manifest_path), "train_samples": manifest["num_samples"],
        "evaluation_dtype": "float32", "batch_size": 1, "num_workers": 2,
        "seed": 42, "rgb_bands_zero_based": [3,2,1], "brightness": 3.0,
        "image_selection": "first two test patches in natural order per season"
    })
    provenance_path = out / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update(python=platform.python_version(), pytorch=torch.__version__,
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                      source_sha256={str(p.relative_to(OFFICIAL)): digest(p) for p in
                          [OFFICIAL / "test.py", OFFICIAL / "train.py",
                           OFFICIAL / "util/util.py", OFFICIAL / "data/sen12mscr_dataset.py"]})
    dump(provenance_path, provenance)
    with (out / "train_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_index", "season", "sample_id", "roi"])
        for index, paths in zip(manifest["indices"], manifest["paths"]):
            path = Path(paths["S2"])
            writer.writerow([index, "spring" if "spring" in str(path) else "winter", path.stem, path.parent.name])
    history = {}
    for path in ckpt.parent.parent.glob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars":0})
        accumulator.Reload()
        for tag in ["epoch","valid_MAE","valid_RMSE","valid_PSNR","valid_SAM","valid_SSIM","valid_LPIPS","valid_loss","learning_rate"]:
            if tag in accumulator.Tags()["scalars"]:
                for event in accumulator.Scalars(tag):
                    key = (event.step, tag)
                    if key not in history or event.wall_time > history[key][0]:
                        history[key] = (event.wall_time, event.value)
    history_rows = []
    for step in sorted({s for s,t in history if t == "valid_loss"}):
        history_rows.append({"step":step, **{tag:val[1] for (s,tag),val in history.items() if s==step}})
    test.write_metrics_csv(out / "validation_history.csv", history_rows)
    del checkpoint
    torch.manual_seed(42)
    np.random.seed(42)
    # Use the exact architecture/config saved in the trusted training checkpoint.
    test.find_config_using_name = lambda name: lambda: config
    data_class = find_dataset_using_name(config.dataset.name)
    train_paths = {Path(p["S2"]).stem for p in manifest["paths"]}
    audit, visuals = {}, []
    for season in ("spring","winter"):
        valid = data_class(roots[season], split="val", data_range=config.dataset.data_range)
        testing = data_class(roots[season], split="test", data_range=config.dataset.data_range)
        val_ids = {Path(p["S2"]).stem for p in valid.dataset.paths}
        test_ids = {Path(p["S2"]).stem for p in testing.dataset.paths}
        train_rois = {str(Path(p["S2"]).parent) for p in manifest["paths"] if season in p["S2"]}
        val_rois = {str(Path(p["S2"]).parent) for p in valid.dataset.paths}
        test_rois = {str(Path(p["S2"]).parent) for p in testing.dataset.paths}
        audit[season] = {
            "train_samples": sum(season in p["S2"] for p in manifest["paths"]),
            "validation_samples": len(valid), "test_samples": len(testing),
            "train_validation_id_overlap": len(train_paths & val_ids),
            "train_test_id_overlap": len(train_paths & test_ids),
            "validation_test_id_overlap": len(val_ids & test_ids),
            "train_validation_roi_overlap": len(train_rois & val_rois),
            "train_test_roi_overlap": len(train_rois & test_rois),
            "validation_test_roi_overlap": len(val_rois & test_rois)
        }
        assert not any(v for k,v in audit[season].items() if "overlap" in k), audit
        evaluation = Namespace(config="checkpoint-config", name="spring_winter6000_best",
            data_root=roots[season], split="test", ckpt_path=str(ckpt),
            output_dir=str(out / season), export_format="none", rgb_bands=[3,2,1],
            png_brightness=3.0, batch_size=1, num_workers=2, max_samples=None, gpu=0)
        test.run_evaluation(evaluation)
        model = test.load_model(config, str(ckpt), torch.device("cuda:0"))
        with torch.inference_mode():
            for index in range(min(2,len(testing))):
                sample = testing[index]
                batch = {k:torch.from_numpy(v).unsqueeze(0).cuda() for k,v in sample.items()}
                _, merged, target = model.fuse_input(batch)
                pred = model(merged)[0][0].cpu().numpy()
                sar = np.clip(sample["SAR"][0]*255,0,255).astype(np.uint8)
                images = [Image.fromarray(sar).convert("RGB")] + [
                    Image.fromarray(test.to_rgb_image(x,[3,2,1],3.0))
                    for x in [sample["cloudy"],pred,sample["target"]]]
                visuals.append((season, test.resolve_sample_name(testing,index), images))
        del model, valid, testing
        torch.cuda.empty_cache()
    audit["scope"] = "Exact sample IDs and season/ROI directories; spatial proximity across distinct ROIs is not audited."
    dump(out / "split_audit.json", audit)
    size, label, margin = 256, 42, 10
    canvas = Image.new("RGB",(4*size+5*margin, len(visuals)*(size+label)+label), "white")
    draw = ImageDraw.Draw(canvas)
    for column,title in enumerate(["SAR (VV)","Cloudy RGB","ECRformer RGB","Target RGB"]):
        draw.text((margin+column*(size+margin),10),title,fill="black")
    for row,(season,sample_id,images) in enumerate(visuals):
        y=label+row*(size+label)
        draw.text((margin,y),season+" / "+sample_id,fill="black")
        for column,im in enumerate(images):
            canvas.paste(im,(margin+column*(size+margin),y+label))
    canvas.save(out / "comparison.png")
    print("EVALUATION_COMPLETE",flush=True)

if __name__ == "__main__":
    main()
