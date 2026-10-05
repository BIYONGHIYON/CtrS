import json
from pathlib import Path
import torch
from torch.utils.data import ConcatDataset, Subset
import train

BASE = Path(__file__).resolve().parent
RUN = BASE / "experiments/ecrformer_spring_winter_3000_each_seed42_20ep/version_1"

def paths_of(dataset):
    if isinstance(dataset, ConcatDataset):
        return [p for part in dataset.datasets for p in paths_of(part)]
    if isinstance(dataset, Subset):
        paths = paths_of(dataset.dataset)
        return [paths[i] for i in dataset.indices]
    return dataset.dataset.paths

def build(config, dataset_class):
    roots = dict(item.split("=", 1) for item in config.dataset.root.split(";"))
    trains, valids = [], []
    for season in ("spring", "winter"):
        trains.append(dataset_class(roots[season], split="train", data_range=config.dataset.data_range, crop_size=config.dataset.crop_size))
        valids.append(dataset_class(roots[season], split="val", data_range=config.dataset.data_range))
    assert [len(d) for d in trains] == [24378, 13995]
    assert [len(d) for d in valids] == [756, 2771]
    return ConcatDataset(trains), ConcatDataset(valids), {"train": "train(spring+winter)", "valid": "val(spring+winter)"}

def select(dataset, max_samples, seed):
    manifest = json.loads((RUN / "train_subset.json").read_text(encoding="utf-8"))
    indices = manifest["indices"]
    all_paths = paths_of(dataset)
    actual = [all_paths[i] for i in indices]
    assert actual == manifest["paths"], "Training paths differ from saved manifest"
    assert seed == manifest["seed"] and max_samples == len(indices) == 6000
    assert sum("spring" in p["S1"] for p in actual) == 3000
    assert sum("winter" in p["S1"] for p in actual) == 3000
    print("Verified original manifest: spring=3000, winter=3000", flush=True)
    return Subset(dataset, indices), indices

if __name__ == "__main__":
    checkpoint = torch.load(RUN / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
    config = checkpoint["hyper_parameters"]["config"]
    print("Resume checkpoint epoch:", checkpoint["epoch"], "step:", checkpoint["global_step"], flush=True)
    config.name = "winter_3000_each_seed42_20ep"
    config.train.max_epoch = 20
    config.train.max_train_samples = 6000
    config.train.save_dir = "./experiments"
    config.no_resume = False
    config.train.ckpt_path = None
    config.train.init_weights_path = None
    train.config_name = "ecrformer_spring"
    train.find_latest_checkpoint = lambda save_dir, log_name: (str(RUN / "checkpoints/last.ckpt"), 1)
    train.build_train_valid_datasets = build
    train.select_fixed_subset = select
    train.get_subset_paths = lambda dataset, indices: list(map(paths_of(dataset).__getitem__, indices))
    del checkpoint
    train.main(config)
