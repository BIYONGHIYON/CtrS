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
| [1. 공개 코드 재현](experiments/1_public_code_reproduction/README.md)<br>[이병현] | 공개 코드 설정을 재현해 초기 기준을 확보했습니다. | RR·FR 재현 결과 확보; 구조 개선 실험이 아님 | 평⁠가⁠ ⁠완⁠료 | 비⁠교⁠ ⁠기⁠준 |
| [2. 논문 설정 재현](experiments/2_paper_setting_reproduction/README.md)<br>[김경찬] | 논문의 내부 특징 차원 설정으로 재현했습니다. | RR·FR 재현 결과 확보; 공개 코드 재현과 장치 조건이 다름 | 평⁠가⁠ ⁠완⁠료 | 비⁠교⁠ ⁠기⁠준 |
| [3. MS 관측 연산자 구현](experiments/3_ms_observation/README.md)<br>[이병현] | 센서 MTF·샘플링 위상·경계 처리를 구성했습니다. | 관측 구현 대응 확인; 학습 성능 비교가 아님 | 사⁠전⁠ ⁠검⁠증⁠ ⁠완⁠료 | 성⁠능⁠ ⁠비⁠교⁠ ⁠아⁠님 |
| [4. 23탭 확대·저해상도 보정·고주파 경로](experiments/4_interpolation_correction_detail/README.md)<br>[이병현] | 확대 연산, 출력 보정, PAN 고주파 경로를 개별 변경했습니다. | 23탭: PSNR +0.0557 dB, SAM −0.0429°, 잠정 QNR +0.0031; 다른 후보 이득 혼재 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [5. 23탭 확대 위치와 센서 확장](experiments/5_interpolation_scope_sensors/README.md)<br>[이병현] | 23탭 적용 위치를 분리하고 시드·센서를 확장했습니다. | RR 이득이 있으나 FR·센서별 결과가 엇갈림 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [6. 23탭 확대의 반복 시드 비교](experiments/6_interpolation_seed_repeat/README.md)<br>[이병현] | 전체 확대 및 입력 확대 후보를 추가 시드로 반복했습니다. | 전체 확대의 GF2 PSNR·QNR 개선 반복; SAM 및 QB QNR 혼재 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [7. 밴드별 게이트 고주파 주입](experiments/7_band_gated_detail/README.md)<br>[김윤지] | 밴드별 게이트로 PAN 고주파 주입량을 조절했습니다. | RR 악화로 후보 채택 보류 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [8. MTF 변화량 데이터 증강](experiments/8_mtf_augmentation/README.md)<br>[이병현] | 센서 열화 특성의 변화량을 학습 데이터에 추가했습니다. | MTF: PSNR -0.0064 dB, SAM -0.0061°; 잠정 QNR -0.0014 | 평가⁠ ⁠완료 | 부분⁠ ⁠개선 |
| [9. SSAI 내부 융합 연산 변경](experiments/9_ssai_fusion/README.md)<br>[이병현] | SSAI 핵심 융합 연산을 네 구조로 비교했습니다. | A1: SAM -0.0058°, 잠정 QNR +0.0127; 모든 후보의 PSNR은 하락 | 평가⁠ ⁠완료 | 부분⁠ ⁠개선 |
| [10. 반복 복원에 MS 관측 오차 전달](experiments/10_recurrent_observation/README.md)<br>[김윤지] | 반복 단계에 입력 MS와 관측한 출력의 차이를 전달했습니다. | PSNR +0.1158 dB, SAM −0.0183°, 잠정 QNR −0.0182 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [11. Haar 대역별 단계 복원과 게이트](experiments/11_haar_reconstruction/README.md)<br>[김윤지] | Haar 대역별 복원에 밴드·방향·위치 게이트를 적용했습니다. | 원본 대비 RR 큰 폭 악화; 학습 게이트의 추가 이득도 없음 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [12. SSAI 내부 특징 차원 비교](experiments/12_feature_dimension/README.md)<br>[이병현] | 같은 환경에서 내부 특징 차원 4와 6을 비교했습니다. | K4: 세 센서 RR PSNR·SAM 우세; WV3 잠정 QNR은 K6 우세 | 평가⁠ ⁠완료 | 비교⁠ ⁠기준 |
| [13. 반복 복원·관측 오차 전달의 재개 학습](experiments/13_recurrent_observation_resume/README.md)<br>[김윤지] | 반복 복원과 관측 오차 후보를 재개해 기준선과 비교했습니다. | 기준선 대비 RR 개선·잠정 QNR 하락; 별도 세션 비교 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [14. 분광·관측 일치·경계 손실 추가](experiments/14_auxiliary_losses/README.md)<br>[이병현] | 기본 MSE에 보조 손실을 하나씩 추가했습니다. | edge 0.1: PSNR −0.0661 dB, SAM −0.0119°, 잠정 QNR +0.0098 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [15. 단일 복원과 3단계 공유 복원](experiments/15_shared_reconstruction/README.md)<br>[김윤지] | 동일 초기값과 학습 조건에서 복원 횟수를 변경했습니다. | 반복 복원: PSNR +0.3000 dB, SAM −0.1078°, 잠정 QNR −0.0393 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [16. 반복 복원의 축소 전 저역통과 필터](experiments/16_bandlimited_downsampling/README.md)<br>[이병현] | 축소 전에 고정 Gaussian 필터를 추가했습니다. | PSNR −0.0196 dB, SAM +0.0022°, 잠정 QNR −0.0686 | 평⁠가⁠ ⁠완⁠료 | 개⁠선⁠ ⁠없⁠음 |
| [17. SSAI 단계·잔차 복원·초기 입력 변경](experiments/17_structural_ablation/README.md)<br>[이병현] | SSAI 단계, 복원 잔차, LRPAN 입력을 분리해 비교했습니다. | LRPAN 제거: PSNR +0.1297 dB, SAM −0.0704°, 잠정 QNR −0.0111 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [18. 작은 크기 특징 보존과 PAN 기여량 조절](experiments/18_coarse_feature_fusion/README.md)<br>[이병현] | 작은 특징을 병렬 전달하고 PAN 기여량을 학습했습니다. | N3: PSNR +0.1875 dB, SAM −0.0134°, 잠정 QNR −0.0147 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [19. 반복 복원과 PAN·MS 분리 경로](experiments/19_dual_stream_reconstruction/README.md)<br>[이병현] | 3단계 공유 복원과 분리 특징 경로를 비교했습니다. | C1: PSNR +0.2442 dB, SAM −0.1883°, 잠정 QNR +0.0250. 새 분리 경로 C5는 RR 악화 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [20. 반복 복원의 단계 간 특징 전달과 관측·PAN 차이](experiments/20_recurrent_guidance/README.md)<br>[이병현] | 밴드별 SSAI를 유지하며 단계 간 정보 전달을 비교했습니다. | D3: PSNR +0.0956 dB; D4·D5 관측 오차 감소; D6 잠정 QNR +0.0166, RR 악화 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [21. 특징 선택·복원 후 관측 보정·고주파 PAN 차이](experiments/21_guidance_refinement/README.md)<br>[이병현] | 공간·밴드별 특징 gate, 복원 후 관측 보정, 고주파 PAN 차이를 비교합니다. | E1: PSNR +0.3345 dB, SAM -0.2219°; E7: 잠정 QNR +0.0437, 지표별 이득 상이 | 평⁠가⁠ ⁠완⁠료 | 부⁠분⁠ ⁠개⁠선 |
| [22. 단계별 갱신량·특징 변환·마지막 단계 PAN 주입](experiments/22_stage_aware_restoration/README.md)<br>[이병현] | 공유 복원기의 단계별 조절과 마지막 단계 PAN 차이를 비교했습니다. | F0 기준 F4: PSNR -0.0009 dB, SAM +0.0064°, 잠정 QNR +0.0018; 추가 이득 혼재 | 평⁠가⁠ ⁠완료 | 부⁠분⁠ ⁠개선 |
| [23. 반복 복원·단계 간 특징 전달의 센서별 수렴 비교](experiments/23_multi_sensor_comparison/README.md)<br>[이병현] | 원본·3단계 공유 복원·특징 전달을 세 센서에서 비교합니다. | QB 세 모델 150에포크·자동 평가 완료; GF2·WV3 학습 진행 중 | 학습&nbsp;중<br>2026-10-13 14:14 KST | 판단&nbsp;보류 |

**판정 기준:** 개선 = 같은 조건에서 주요 RR·FR 지표가 함께 개선, 부분 개선 = 지표·후보·센서별 결과 혼재, 개선 없음 = 평가에서 향상을 확인하지 못함, 판단 보류 = 최종 평가 부족입니다. 비교 기준·성능 비교 아님은 개선 판정 대상에서 제외합니다. 모든 FR QNR은 구현 대응 검증이 끝나지 않은 잠정 지표이며, 단일 실험의 개선은 최종 채택을 뜻하지 않습니다.

<a id="files"></a>

## 폴더 구조

```text
SSA-MRN/
├── common/                  여러 실험이 재사용하는 모델·데이터·평가 코드
├── experiments/
│   ├── 1_public_code_reproduction/
│   ├── 2_paper_setting_reproduction/
│   └── 21_guidance_refinement/
│       ├── README.md         목적·변경 구조·조건·결과·판정
│       ├── model.py          이 실험에서 바꾼 모델
│       ├── run.py            학습 실행 또는 Kaggle 한 셀 코드
│       ├── prepare.py        필요한 경우 데이터 캐시·실행 설정 준비
│       ├── checkpoints/      실제 학습한 best/latest 가중치
│       ├── results/          수치·compact 학습 기록·그래프·예시
│       ├── provenance/       학습 당시 원본 소스·필수 설정
│       └── manifest.json    보존 파일의 크기·SHA-256
├── data/                    로컬 H5 데이터; Git에 업로드하지 않습니다.
├── docs/                    연구 설명·문서 양식·구조도
├── references/              공식 코드·논문·보존이 필요한 원본 소스
└── tests/                   공통 연산·학습 재개 검증
```

실험 번호는 과거부터 최근 순서이며 새 실험은 마지막 번호 다음에 추가합니다. 완료된 실험 번호는 바꾸지 않습니다. 폴더에 실제로 필요한 파일만 두며, 공통 모델만 쓰는 실험에는 중복 `model.py`를 만들지 않습니다. 가중치 종류는 `best.pt`, `latest.pt`로 통일하고 후보·센서·시드는 상위 폴더와 설정으로 구분합니다.

`provenance`는 기존 가중치를 만든 원본입니다. 그 내부의 역사적 import·파일명과 체크포인트 설정은 보존합니다. 현행 실행은 실험 폴더의 코드 또는 `common`을 사용합니다. 실행 결과는 `runtime/`에 저장하고 정리된 결과만 실험 폴더에 반영합니다.

## 환경 준비

저장소 루트에서 실행합니다.

```bash
git submodule update --init SSA-MRN/references/upstream
python -m pip install -r SSA-MRN/requirements.txt
```

데이터는 `SSA-MRN/data/` 아래에 두거나 실행 옵션으로 별도 경로를 지정합니다. `data/processed` 단계는 사용하지 않습니다. 기존 서버의 원본 데이터와 실행 중인 checkout은 이동하지 않았습니다.

<a id="evaluate"></a>

## 가중치 확인·재평가

실험 번호와 해당 실험 안의 가중치 경로를 지정합니다. 이 명령은 학습을 시작하지 않습니다.

```bash
# 데이터 없이 strict load와 CPU 추론 확인
python SSA-MRN/common/checkpoint.py --experiment 20 --checkpoint checkpoints/D4/best.pt

# 지정한 H5 전체 RR 평가; FR이면 --protocol FR과 해당 H5를 사용합니다.
python SSA-MRN/common/checkpoint.py --experiment 20 --checkpoint checkpoints/D4/best.pt --h5 /path/to/test_qb_multiExm1.h5 --protocol RR --output SSA-MRN/runtime/D4_rr.json

# 보존 파일의 해시 확인
python SSA-MRN/common/verify.py --experiment 20
```

`--device cuda`로 GPU 평가가 가능합니다. 공통 재평가는 FP32이며 과거 실험의 추가 지표·관측 방식과 차이가 있을 수 있습니다. 저장된 결과를 덮어쓰지 않습니다. QNR·Q2n·SCC의 MATLAB parity는 미검증입니다.

## 새 학습

서버형 실험은 해당 `run.py --help`와 `prepare.py --help`에서 옵션을 확인합니다. Kaggle형 실험은 `run.py` 전체를 한 셀에 붙여 넣고 파일 상단의 데이터·학습·재개 설정을 지정합니다. Kaggle 코드는 학습 당시 고정된 원본 소스를 사용하므로 Internet이 필요할 수 있습니다.

실행 중인 서버 작업은 기존 checkout과 명령을 유지합니다. 새 폴더의 코드를 실행 중인 checkout에 덮어쓰거나 기존 runtime에 재개하지 않습니다. 재개 fingerprint가 달라질 수 있습니다.

[실험 보고서 양식](docs/experiment_template.md) · [실험 목록](experiments/catalog.json) · [관리 규칙](AGENTS.md)
