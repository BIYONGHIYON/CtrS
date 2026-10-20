"""보존된 v2 측정 요약으로 현재 실험 표용 판정(v2.2, v2.3)을 계산합니다."""

import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


NOISE = {1: 0.06}  # v2.3: RR PSNR noise threshold by seed count (43 C0 seed SD 0.056 dB / sqrt(n), rounded up)
NOISE_MULTI = 0.04


def rejudge(record, version="v2.2"):
    delta = record["delta"]
    rr = delta["RR"]
    components = {"PSNR": rr["PSNR"] > 0, "SAM": rr["SAM"] < 0,
                  "ERGAS": rr["ERGAS"] < 0}
    primary = all(components.values())
    guards = {
        "RR2": delta["RR2"]["PSNR"] >= -0.05 and delta["RR2"]["SAM"] <= 0.06,
        "FR_WALD": delta["FR_WALD"]["PSNR"] >= -0.07 and delta["FR_WALD"]["SAM"] <= 0.04,
        "FR_OBS": record["obs_ratio"] <= 1.10,
    }
    label = ("개선" if primary and all(guards.values()) else
             "부분 개선" if any(components.values()) else "개선 없음")
    out = {
        **record,
        "original_v2_label": record["label"],
        "primary": primary,
        "guards": guards,
        "rr_components_improved": components,
        "label": label,
    }
    if version == "v2.3":
        threshold = NOISE.get(len(record["seeds"]), NOISE_MULTI)
        above = rr["PSNR"] >= threshold
        out["noise_threshold_dB"] = threshold
        out["above_noise"] = above
        if label == "개선" and not above:
            out["label"] = "부분 개선"
            out["note"] = "all items pass but RR dPSNR is below the noise threshold"
    return out


def main(version="v2.2"):
    source = HERE / "results/summary.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    judgements = {group: {name: rejudge(record, version) for name, record in candidates.items()}
                  for group, candidates in original["judgements"].items()}
    counts = Counter(record["label"] for candidates in judgements.values()
                     for record in candidates.values())
    output = {
        "criteria_version": version,
        "criteria_document": "SSA-MRN/docs/judgement_criteria.md",
        "source": "results/summary.json",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "QB; original measured seed means; no training or inference; current table labels only",
        "aggregation": "single seed or all paired seed means; no majority condition; OBS ratio of seed means",
        "thresholds": {"RR": {"PSNR_gt": 0, "SAM_lt": 0, "ERGAS_lt": 0},
                       "RR2": {"PSNR_ge": -0.05, "SAM_le": 0.06},
                       "FR_WALD": {"PSNR_ge": -0.07, "SAM_le": 0.04},
                       "FR_OBS_ratio_le": 1.10,
                       **({"RR_PSNR_noise_ge": {"1_seed": NOISE[1], "2_or_more_seeds": NOISE_MULTI}} if version == "v2.3" else {})},
        "counts": dict(counts),
        "judgements": judgements,
    }
    target = HERE / f"results/summary_{version.replace('.', '_')}.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(dict(counts))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "v2.2")
