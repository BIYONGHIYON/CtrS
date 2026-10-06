# 정합 보정 256 모델 가중치

best.pt (epoch 98), latest.pt (epoch 100)과 checksums.json을 보존합니다.

[실험 보고서](../../../docs/experiments/rgb_hsi/rgb02_aligned256.md). 과거 별도 학습 설정은 폐기했습니다. 기존 가중치의 내장 config를 읽어 `scripts/export_results.py --checkpoint <best.pt> ...`로 평가합니다.
