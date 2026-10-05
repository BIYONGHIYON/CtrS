# Spring 고정 6,000개 · 발산 분석

[현재 연구](../../README.md) · [실험 이력](../previous_experiments.md)

## 1. 목적과 상태

2026-10-04 실행은 조기 종료됐고 안정화 실학습은 미실행입니다. [기존 상세 로그 분석](https://github.com/BIYONGHIYON/CtrS/blob/main/ECRformer/docs/spring_subset6000_log_analysis_20261005.md)은 main에 병합된 문서입니다. 당시 기록을 요약한 것이며 오늘 서버 상태를 재측정한 보고는 아닙니다.

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

상세 로그는 서버에 있지만 이번 정리에서 다운로드·곡선 생성을 하지 않았습니다. 표는 기존 관측값만 요약하며 없는 epoch를 채우지 않습니다.

## 6. 이미지

spring best의 test 그림과 사전 선정 5사례는 미확보입니다. 겨울 그림을 spring 결과로 표시하지 않습니다.

## 7. 근거

서버 `C:\CtrS\ECRformer\Official_ECRformer`의 `spring_subset_resume_detailed.stdout.log`, `.stderr.log` 및 `experiments\ecrformer_spring_spring_subset6000\version_0\training_diagnostics.jsonl`. checkpoint의 `epoch=8-step=3375.ckpt`는 평가 후보이고 `last.ckpt`는 발산 상태입니다. 서버 경로는 GitHub 링크가 아닙니다. 실행 당시 코드 commit·가중치 해시는 미확보입니다.

## 8. 다음 판단

[안정화 실행](../stable_training.md)은 lr 1e-4, 정체 기반 감소, FP32 손실·지표와 배치별 진단을 지원합니다. 여러 요인을 동시에 바꾸므로 FP16 단독 원인을 확정할 수 없습니다. 다음 결과에 실제 곡선·이미지·비교 조건·실패 사례를 기록합니다. 이번 작업은 학습·추론을 실행하지 않습니다.
