# 가변 배율 확장 실험

[연구 소개](../../../README.md) · [실행 안내](../../README.md) · [판정 기준](../../docs/judgement_criteria.md)

고정 4배 복원을 여러 배율에서 동작하도록 확장하는 연구입니다. **2026-10-27 실용성이 낮아 종료**했으며 새 실험은 [성능 개선](../performance/README.md)에 등록합니다.

## 결론

- 배율별 노출량을 맞춘 3번에서 공동 모델의 3·4배 RR이 전용 모델보다 나아졌지만 2배에서는 격차가 남았습니다.
- RR·FR 전체 장면 평가가 끝나지 않아 세 실험 모두 판단 보류입니다.
- 확인한 배율 밖(비정수 배율, 학습하지 않은 배율)은 지원한다고 쓰지 않습니다.

<a id="experiments"></a>

## 실험 표

<!-- experiments:start -->
| 실험 | 핵심 결과 | 상태 | 판정 |
|---|---|:---:|:---:|
| [3. 배율별 학습 노출량 통제](3_scale_exposure_control/README.md)<br>김윤지 | D가 전용 모델 대비 3·4배 +0.142·+0.147 dB, 2배 −0.226 dB | 학습&nbsp;완료·일부&nbsp;평가 | 판단&nbsp;보류 |
| [2. 배율별 학습 노출량 확장](2_scale_exposure_extension/README.md)<br>김윤지 | 모든 배율에서 RR 개선, 전용 모델과의 PSNR 격차는 유지 | 학습&nbsp;완료·일부&nbsp;평가 | 판단&nbsp;보류 |
| [1. 가변 배율 복원과 배율 조건](1_scale_conditioned_restoration/README.md)<br>김윤지 | 공동 모델이 전용 모델 대비 0.40-0.98 dB 하락, FiLM은 3·4배 소폭 개선 | 학습&nbsp;완료·일부&nbsp;평가 | 판단&nbsp;보류 |
<!-- experiments:end -->
