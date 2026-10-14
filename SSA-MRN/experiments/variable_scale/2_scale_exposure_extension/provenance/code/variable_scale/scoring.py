"""Normalized QB RR metrics, with explicit ratio and no prediction clipping."""
import numpy as np
from skimage.metrics import structural_similarity

def rr_score(reference, prediction, scale):
    a, b = (np.asarray(x, dtype=np.float64) for x in (reference, prediction))
    if scale not in (2, 3, 4) or a.shape != b.shape or a.ndim != 3:
        raise ValueError("CHW shapes and supported ratio required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite RR input")
    band_mse = np.mean((a-b)**2, axis=(1, 2))
    norms = np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0)
    valid = norms > 1e-15
    cosine = np.divide(np.sum(a*b, axis=0), norms, out=np.ones_like(norms), where=valid)
    sam = float(np.degrees(np.arccos(np.clip(cosine[valid], -1, 1))).mean()) if valid.any() else 0.
    window = min(7, a.shape[-2], a.shape[-1])
    window -= 1 - window % 2
    return {"MSE": float(band_mse.mean()),
            "PSNR": float(np.mean(-10*np.log10(np.maximum(band_mse, 1e-30)))),
            "SAM": sam,
            "ERGAS": float(100/scale*np.sqrt(np.mean(band_mse/np.maximum(a.mean((1,2))**2,1e-30)))),
            "SSIM": float(np.mean([structural_similarity(x, y, data_range=1., win_size=window)
                                     for x, y in zip(a, b)]))}

def scene_means(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["scene_id"], []).append(row["metrics"])
    scenes = [{"scene_id": scene, "metrics": {k: float(np.mean([r[k] for r in items]))
               for k in items[0]}} for scene, items in sorted(groups.items())]
    mean = {k: float(np.mean([s["metrics"][k] for s in scenes])) for k in scenes[0]["metrics"]}
    return {"samples": rows, "scenes": scenes, "scene_mean": mean, "scene_count": len(scenes)}
