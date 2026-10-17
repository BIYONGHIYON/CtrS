"""Resolve attached QB files or acquire the specified Kaggle dataset once."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HANDLE = "biyonghiyon/ctrs-pan-sharpening-dataset"
ROOT = "/kaggle/input/datasets/biyonghiyon/ctrs-pan-sharpening-dataset"
FILENAMES = {
    "train": "QuickBird__Training_Dataset__train_qb.h5",
    "val": "QuickBird__Training_Dataset__valid_qb.h5",
    "RR": "QuickBird__Testing_Dataset_ReducedData_H5_Format__test_qb_multiExm1.h5",
    "FR": "QuickBird__Testing_Dataset_FullData_H5_Format__test_qb_OrigScale_multiExm1.h5",
}


def valid_h5(path, split):
    import h5py
    try:
        with h5py.File(path, "r") as f:
            keys = ("pan", "ms", "lms") + (() if split == "FR" else ("gt",))
            return all(k in f and len(f[k].shape) == 4 and f[k].shape[0] > 0 for k in keys)
    except (OSError, ValueError, TypeError):
        return False


def locate(root, splits):
    root = Path(root)
    if not root.is_dir():
        return {}
    found = {}
    for split in splits:
        candidates = sorted(root.rglob(FILENAMES[split]))
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous {split} files under {root}; specify an exact paths mapping")
        if candidates and valid_h5(candidates[0], split):
            found[split] = str(candidates[0].resolve())
    return found


def unpack(root, cache):
    """Never extract into read-only Inputs, outside cache, or over another run."""
    from .data import sha_file
    root, cache = Path(root), Path(cache).resolve()
    archives = sorted(root.rglob("*.zip")) if root.is_dir() else [root]
    roots = []
    for archive in archives:
        if archive.suffix.lower() != ".zip":
            continue
        destination = cache / sha_file(archive)
        if not destination.exists():
            # Validate the entire member list before writing any member.
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    name = member.filename.replace("\\", "/")
                    target = (destination / name).resolve()
                    if (not target.is_relative_to(destination.resolve()) or
                            ":" in name or ((member.external_attr >> 16) & 0o170000) == 0o120000):
                        raise ValueError("Unsafe archive member")
                if z.testzip() is not None:
                    raise ValueError("Dataset archive CRC failure")
                destination.mkdir(parents=True, exist_ok=False)
                z.extractall(destination)
            (destination / ".complete").write_text("verified extraction", encoding="utf-8")
        if not (destination / ".complete").is_file():
            raise ValueError(f"Incomplete dataset extraction: {destination}; use a fresh DATA_CACHE")
        roots.append(destination)
    return roots


def resolve_dataset(paths=None, require_fr=False, version=None, cache=None,
                    downloader=None, event=None):
    event = event or (lambda stage, **values: None)
    required = ["train", "val", "RR"] + (["FR"] if require_fr else [])
    paths = dict(paths or {s: str(Path(ROOT) / name) for s, name in FILENAMES.items()})
    found = {s: str(Path(paths[s]).resolve()) for s in required
             if s in paths and valid_h5(paths[s], s)}
    event("dataset_check", required=required, found=list(found))
    source = "existing_paths"
    resolved_root = None
    if len(found) != len(required):
        handle = HANDLE + (f"/versions/{int(version)}" if version else "")
        if downloader is None:
            if importlib.util.find_spec("kagglehub") is None:
                event("dataset_dependency", package="kagglehub")
                subprocess.run([sys.executable, "-m", "pip", "install", "kagglehub>=0.3,<2"], check=True)
            import kagglehub
            downloader = kagglehub.dataset_download
        event("dataset_download", handle=handle)
        try:
            resolved_root = Path(downloader(handle)).resolve()
        except Exception as error:
            # Do not echo request URLs or credential-bearing library messages.
            raise RuntimeError(
                f"Kaggle download failed ({type(error).__name__}). "
                "Check Notebook Internet, dataset access/consent and Kaggle authentication; "
                "or attach the specified Dataset as Input. Workers have not started."
            ) from None
        source = "kagglehub"
        # A download is one coherent dataset; do not silently mix revisions.
        found = locate(resolved_root, required)
        if len(found) != len(required):
            event("dataset_extract", root=str(resolved_root))
            for extracted in unpack(resolved_root, cache or "/kaggle/working/ctrs_data_cache"):
                candidates = locate(extracted, required)
                for split, path in candidates.items():
                    if split in found and found[split] != path:
                        raise ValueError(f"Ambiguous extracted {split}; specify exact paths")
                    found[split] = path
        if len(found) != len(required):
            raise FileNotFoundError(
                f"Specified Dataset is missing valid H5 splits: {sorted(set(required)-set(found))}. "
                "Check its file layout/version; no worker was started."
            )
    event("dataset_ready", paths=found)
    return found, {"handle": HANDLE, "requested_version": version, "source": source,
                   "returned_root": str(resolved_root) if resolved_root else None,
                   "version_policy": "requested version when set; resume always checks all file hashes"}
