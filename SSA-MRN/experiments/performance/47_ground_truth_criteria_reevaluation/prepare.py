"""보존된 v2 측정 요약으로 현재 실험 표용 v2.2 판정만 계산합니다."""

import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def rejudge(record):
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
    return {
        **record,
        "original_v2_label": record["label"],
        "primary": primary,
        "guards": guards,
        "rr_components_improved": components,
        "label": label,
    }


def main():
    source = HERE / "results/summary.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    judgements = {group: {name: rejudge(record) for name, record in candidates.items()}
                  for group, candidates in original["judgements"].items()}
    counts = Counter(record["label"] for candidates in judgements.values()
                     for record in candidates.values())
    output = {
        "criteria_version": "v2.2",
        "criteria_document": "SSA-MRN/docs/judgement_criteria.md",
        "source": "results/summary.json",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "QB; original measured seed means; no training or inference; current table labels only",
        "aggregation": "single seed or all paired seed means; no majority condition; OBS ratio of seed means",
        "thresholds": {"RR": {"PSNR_gt": 0, "SAM_lt": 0, "ERGAS_lt": 0},
                       "RR2": {"PSNR_ge": -0.05, "SAM_le": 0.06},
                       "FR_WALD": {"PSNR_ge": -0.07, "SAM_le": 0.04},
                       "FR_OBS_ratio_le": 1.10},
        "counts": dict(counts),
        "judgements": judgements,
    }
    target = HERE / "results/summary_v2_2.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(dict(counts))


if __name__ == "__main__":
    main()
