# SSA-MRN 연구·실행 안내

[연구 소개와 구조도](../README.md) · [알고리즘·지표 설명](docs/research.md)

<a id="decision"></a>

## 연구 기준

원본 SSA-MRN의 K=4·MSE 손실을 기준으로 개선 구조를 비교합니다. 실제 K·센서·학습 조건은 각 보고서에 기록합니다. 과거 결과의 설정은 그대로 유지합니다.

<a id="status"></a>

## 결과를 읽는 방법

후보와 best 체크포인트는 validation으로 선택합니다. RR·FR test 결과는 선택 후 확인하며 학습 완료와 평가 완료를 구분합니다. 아래 표는 저장된 보고서 기준이며 실시간 서버 상태가 아닙니다.

<a id="experiments"></a>

## 실험 보고서

실험은 **오래된 순서부터 최근 순서**로 정렬했습니다. 같은 날의 기록은 보고서에 확인된 완료·보관 순서를 따릅니다.

| 실험 | 변경한 내용 | 확인된 결과 | 상⁠태 | 개⁠선⁠ ⁠여⁠부 |
|---|---|---|:---:|:---:|
| [공개 코드 재현](docs/experiments/public_code_reproduction.md) | 공개 코드 설정을 재현해 초기 기준을 확보했습니다. | RR·FR 재현 결과 확보; 구조 개선 실험이 아님 | 평⁠가⁠ ⁠완⁠료 | 비⁠교⁠ ⁠기⁠준 |
| [논문 설정 재현](docs/experiments/paper_setting_reproduction.md) | 논문의 내부 특징 차원 설정으로 재현했습니다. | RR·FR 재현 결과 확보; 공개 코드 재현과 장치 조건이 다름 | 평⁠가⁠ ⁠완⁠료 | 비⁠교⁠ ⁠기⁠준 |
| [MS 관측 연산자 구현](docs/experiments/ms_observation.md) | 센서 MTF·샘플링 위상·경계 처리를 구성했습니다. | 관측 구현 대응 확인; 학습 성능 비교가 아님 | 사⁠전⁠ ⁠검⁠증⁠ ⁠완⁠료 | 성⁠능⁠ ⁠비⁠교⁠ ⁠아⁠님 |
| [23탭 확대·저해상도 보정·고주파 경로](docs/experiments/interpolation_correction_detail.md) | 확대 연산, 출력 보정, PAN 고주파 경로를 개별 변경했습니다. | 23탭: PSNR +0.0557 dB, SAM −0.0429°, 잠정 QNR +0.0031; 다른 후보 이득 혼재 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [23탭 확대 위치와 센서 확장](docs/experiments/interpolation_scope_sensors.md) | 23탭 적용 위치를 분리하고 시드·센서를 확장했습니다. | RR 이득이 있으나 FR·센서별 결과가 엇갈림 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [23탭 확대의 반복 시드 비교](docs/experiments/interpolation_seed_repeat.md) | 전체 확대 및 입력 확대 후보를 추가 시드로 반복했습니다. | 전체 확대의 GF2 PSNR·QNR 개선 반복; SAM 및 QB QNR 혼재 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [밴드별 게이트 고주파 주입](docs/experiments/band_gated_detail.md) | 밴드별 게이트로 PAN 고주파 주입량을 조절했습니다. | RR 악화로 후보 채택 보류 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [MTF 변화량 데이터 증강](docs/experiments/mtf_augmentation.md) | 센서 열화 특성의 변화량을 학습 데이터에 추가했습니다. | 100에폭 validation 기록 확보; 이 보고서의 RR·FR 결과는 미확정 | 학⁠습⁠ ⁠완⁠료<br>평⁠가⁠ ⁠대⁠기 | 판⁠단⁠ ⁠보⁠류 |
| [SSAI 내부 융합 연산 변경](docs/experiments/ssai_fusion.md) | SSAI 핵심 융합 연산을 네 구조로 비교했습니다. | validation MSE는 기존 A0가 최저; RR·FR 평가 대기 | 학⁠습⁠ ⁠완⁠료<br>평⁠가⁠ ⁠대⁠기 | 판⁠단⁠ ⁠보⁠류 |
| [반복 복원에 MS 관측 오차 전달](docs/experiments/recurrent_observation.md) | 반복 단계에 입력 MS와 관측한 출력의 차이를 전달했습니다. | PSNR +0.1158 dB, SAM −0.0183°, 잠정 QNR −0.0182 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [Haar 대역별 단계 복원과 게이트](docs/experiments/haar_reconstruction.md) | Haar 대역별 복원에 밴드·방향·위치 게이트를 적용했습니다. | 원본 대비 RR 큰 폭 악화; 학습 게이트의 추가 이득도 없음 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [SSAI 내부 특징 차원 비교](docs/experiments/feature_dimension.md) | 같은 환경에서 내부 특징 차원 4와 6을 비교했습니다. | 센서별 validation 다수결 2:1로 차원 4 선택; test는 일부만 평가 | 학⁠습⁠ ⁠완⁠료<br>일⁠부⁠ ⁠평⁠가 | 비⁠교⁠ ⁠기⁠준 |
| [반복 복원·관측 오차 전달의 재개 학습](docs/experiments/recurrent_observation_resume.md) | 반복 복원과 관측 오차 후보를 재개해 기준선과 비교했습니다. | 기준선 대비 RR 개선·잠정 QNR 하락; 별도 세션 비교 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [분광·관측 일치·경계 손실 추가](docs/experiments/auxiliary_losses.md) | 기본 MSE에 보조 손실을 하나씩 추가했습니다. | edge 0.1: PSNR −0.0661 dB, SAM −0.0119°, 잠정 QNR +0.0098 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [단일 복원과 3단계 공유 복원](docs/experiments/shared_reconstruction.md) | 동일 초기값과 학습 조건에서 복원 횟수를 변경했습니다. | 반복 복원: PSNR +0.3000 dB, SAM −0.1078°, 잠정 QNR −0.0393 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [반복 복원의 축소 전 저역통과 필터](docs/experiments/bandlimited_downsampling.md) | 축소 전에 고정 Gaussian 필터를 추가했습니다. | PSNR −0.0196 dB, SAM +0.0022°, 잠정 QNR −0.0686 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [SSAI 단계·잔차 복원·초기 입력 변경](docs/experiments/structural_ablation.md) | SSAI 단계, 복원 잔차, LRPAN 입력을 분리해 비교했습니다. | LRPAN 제거: PSNR +0.1297 dB, SAM −0.0704°, 잠정 QNR −0.0111 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [작은 크기 특징 보존과 PAN 기여량 조절](docs/experiments/coarse_feature_fusion.md) | 작은 특징을 병렬 전달하고 PAN 기여량을 학습했습니다. | N3: PSNR +0.1875 dB, SAM −0.0134°, 잠정 QNR −0.0147 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [반복 복원과 PAN·MS 분리 경로](docs/experiments/dual_stream_reconstruction.md) | 3단계 공유 복원과 분리 특징 경로를 비교했습니다. | C1: PSNR +0.2442 dB, SAM −0.1883°, 잠정 QNR +0.0250. 새 분리 경로 C5는 RR 악화 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |

**판정 기준:** 개선 = 같은 조건에서 주요 RR·FR 지표가 함께 개선, 부분 개선 = 지표·후보·센서별 결과 혼재, 개선 없음 = 평가에서 향상을 확인하지 못함, 판단 보류 = 최종 평가 부족입니다. 비교 기준·성능 비교 아님은 개선 판정 대상에서 제외합니다. 모든 FR QNR은 구현 대응 검증이 끝나지 않은 잠정 지표이며, 단일 실험의 개선은 최종 채택을 뜻하지 않습니다.

<a id="files"></a>

## 폴더 구조와 실행

```text
SSA-MRN/
├── src/ssamrn/       모델·데이터·평가 구현
├── scripts/          학습·평가·실행 관리
├── tests/            코드 검증
├── docs/
│   ├── research.md   알고리즘·지표 설명
│   ├── experiments/ 실험별 보고서·작성 양식
│   ├── operations/  환경 설정·실행·재개 방법
│   └── assets/      보고서의 이미지·지표·가중치·재현 자료
├── data/             로컬 데이터 위치; 원본 데이터는 Git 제외
├── experiments/      실험 목록·보존 해시·기존 재현 가중치
└── references/       공식 코드·논문 등 참고 자료
```

- [환경 설정·기본 학습·평가](docs/operations/pan_ms.md)
- [학습 상태·로그·재개](docs/operations/feature_dimension_losses.md)
- [실험 보고서 양식](docs/experiments/template.md)

운영체제·장치는 재현 조건에 기록합니다. 문서와 공개 실행 파일은 변경한 구조를 나타내는 같은 이름을 사용합니다. 과거 증거 폴더 내부의 run ID·snapshot은 원본 설정을 유지합니다.

[실험별 코드·재평가 안내](scripts/README.md) · [필수 보존 파일 목록](experiments/catalog.json)
