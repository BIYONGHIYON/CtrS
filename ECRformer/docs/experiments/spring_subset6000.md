# Spring 고정 6,000개 · 발산 분석

[현재 연구](../../README.md) · [실험 이력](../previous_experiments.md)

## 1. 목적과 상태

2026-10-04 실행은 조기 종료됐고 안정화 실학습은 미실행입니다. [기존 상세 로그 분석](../spring_subset6000_log_analysis_20261005.md)은 먼저 main에 병합된 문서입니다. 당시 기록을 요약한 것이며 오늘 서버 상태를 재측정한 보고는 아닙니다.

## 2. 조건

전체 ECRformer 약 11.37M, spring train 24,378개 중 6,000개, validation 756개, seed 42, crop 128, RTX 3060 Ti 8GB. FP16 mixed, 배치 2 × accumulation 8, lr 4e-4, 최대 200 epoch, patience 10, gradient clipping 0.5.

## 3. 결과 (validation)

| epoch(0-based) | 학습 MAE 평균 | 검증 MAE | 검증 PSNR dB | 검증 SSIM |
| ---: | ---: | ---: | ---: | ---: |
| 8(best) | 0.040728 | 0.022640 | 30.030531 | 0.912831 |
| 15 | 0.039295 | 0.031105 | 28.259220 | 0.870403 |
| 16 | 7.999050 | 21.068518 | −26.853853 | −0.000260 |
| 18(last) | 57.833340 | 64.329369 | −36.421108 | 0.000013 |

epoch 16 배치 201~400 구간에서 누적 오차 이상을 확인했습니다. 정확한 최초 샘플·연산은 확정할 수 없습니다. 정상 종료 코드와 성능 안정성은 별개입니다.

## 4. 전후 비교

안정화 결과는 미측정이므로 차이는 N/A. 겨울 test와 계절·분할이 달라 직접 비교하지 않습니다.

## 5. 곡선

처음 문서를 정리할 때는 곡선을 생성하지 않았습니다. 이후 2026-10-05 test 평가에서 실제 epoch 7~18의 compact 기록과 곡선을 추가했습니다. [평가 보고서](spring_subset6000_test.md#5-학습검증-및-test-그래프)에서 볼 수 있으며 없는 0~6은 채우지 않았습니다.

## 6. 이미지

2026-10-05에 spring best 전체 test 3,983패치 평가와 사전 고정 5 ROI 비교 그림을 확보했습니다. [test 보고서](spring_subset6000_test.md)에서 수치·선정 근거와 best/last 그림을 확인합니다. 겨울 그림을 spring 결과로 표시하지 않습니다.

## 7. 근거

서버 `C:\CtrS\ECRformer\Official_ECRformer`의 `spring_subset_resume_detailed.stdout.log`, `.stderr.log` 및 `experiments\ecrformer_spring_spring_subset6000\version_0\training_diagnostics.jsonl`. checkpoint의 `epoch=8-step=3375.ckpt`는 평가 후보이고 `last.ckpt`는 발산 상태입니다. 서버 경로는 GitHub 링크가 아닙니다. 학습 당시 코드 commit은 미확보이며 가중치 SHA-256은 [후속 평가 provenance](../../reproduction/spring_subset6000_test_20261005/provenance.json)에 추가했습니다.

## 8. 다음 판단

[안정화 실행](../stable_training.md)은 lr 1e-4, 정체 기반 감소, FP32 손실·지표와 배치별 진단을 지원합니다. 여러 요인을 동시에 바꾸므로 FP16 단독 원인을 확정할 수 없습니다. 초기 문서 정리에서는 실행하지 않았고, 2026-10-05 후속 작업은 기존 checkpoint 추론만 수행했습니다. 새로운 안정화 학습은 아직 실행하지 않았습니다.
