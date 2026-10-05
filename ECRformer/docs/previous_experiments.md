# ECRformer 실험 이력

[현재 연구](../README.md) · [재현과 실행 근거](reproduction.md)

test 평가와 validation 학습 기록을 분리합니다. 계절·ROI·패치 목록·분할이 다른 수치로 개선을 주장하지 않습니다.

| 실험 | 상태 / 평가 범위 | PSNR dB ↑ | SAM ° ↓ | 근거 |
| --- | --- | ---: | ---: | --- |
| 겨울 첫 번째 부분 | 완료 · test 783패치 | 29.6068 | 11.2503 | [재현 기록](reproduction.md), [summary](../reproduction/winter_half1/summary.json) |
| 겨울 두 번째 부분 · 전 | 완료 · 같은 test 784패치 | 25.2455 | 11.4777 | [전후 보고서](experiments/winter_half2_finetune.md) |
| 겨울 두 번째 부분 · 후 | 완료 · 같은 test 784패치 | 25.8885 | 10.7794 | [전후 보고서](experiments/winter_half2_finetune.md) |
| spring 고정 6,000개 | 발산 후 종료 · best validation | 30.0305 | 5.6365 | [학습 분석](experiments/spring_subset6000.md) |
| spring 고정 6,000개 · best test | 2026-10-05 완료 · test 3,983패치·5 ROI | 27.1505 | 9.2988 | [전체 test와 실제 비교 그림](experiments/spring_subset6000_test.md) |
| spring 안정화 | 구현·CPU 검증, 실학습 미실행 | 미측정 | 미측정 | [실행 안내](stable_training.md) |

겨울 두 번째 부분의 전후만 동일 평가 조건입니다. spring best validation을 겨울 test보다 우수하다고 해석하지 않습니다.

## 보존 자료

- `reproduction/winter_half1`, `winter_half2`: 기존 평가 JSON·CSV·설정·이미지·평가용 가중치. 파일과 위치 유지.
- `docs/experiments/`: 개별 보고서와 공통 양식.
- spring 원시 로그·best/last checkpoint: 서버 보관. 이번 정리에서 다운로드·커밋·삭제하지 않음.

겨울 과거 학습 곡선과 선정 manifest는 미확보입니다. spring은 epoch 7~18의 실제 재개 로그 곡선, 사전 고정 5 ROI 사례, 전체 test 지표와 checkpoint 해시를 `reproduction/spring_subset6000_test_20261005`에 추가했습니다. 없는 과거 자료를 채우거나 기존 겨울 그림을 사전 선정 표본으로 재분류하지 않습니다.
