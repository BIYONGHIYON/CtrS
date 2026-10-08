# QB · 밴드별 게이트 고주파 주입 실험 계획

[실험 목록](../../README.md#experiments) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [가중치·근거](#evidence)

| 비교 기준 | 이번에 바꿀 점 | 관측 차이 | 판단 |
|---|---|---|---|
| QB K6 기준선 및 기존 고정 고주파 경로 | PAN 고주파 잔차를 MS 밴드·위치별 게이트로 조절 | 아직 학습·평가 전 | 계획 단계. 기존 고정 주입의 단점을 줄이는지 같은 Kaggle 조건에서 비교 |

## 1. 목적과 상태

**상태: 실험 계획 및 노트북 준비 완료, 학습·테스트 미실행.** 이 문서는 결과 보고서가 아닙니다.

기존 고주파 변형은 PAN에서 5×5 binomial 저주파를 뺀 뒤 1→16→4 Conv 잔차를 모든 MS 밴드에 더합니다. QB 단일 시드 결과에서는 일부 SAM 개선과 함께 RR PSNR·ERGAS 및 잠정 FR QNR이 기준선보다 낮았습니다. PAN 상세 성분이 모든 밴드·위치에서 똑같이 유용하지 않을 수 있다는 가설을 검토합니다.

후보는 기존 SSA-MRN 출력에 PAN 고주파 잔차를 더하되, 기준 출력의 밴드별 공간 기울기와 PAN 고주파를 입력으로 하는 학습형 게이트가 밴드·픽셀마다 주입량을 정합니다. 학습 손실·데이터·SSA-MRN 본체는 바꾸지 않습니다.

실행 코드는 [Kaggle 노트북](../../scripts/kaggle_band_gated_hf.ipynb)입니다. 기준 저장소 commit은 be02b7c이며, 실제 학습 시 Kaggle Notebook Version, 실행 코드 commit, 장치·PyTorch 버전과 데이터 해시를 결과 기록에 추가합니다.

## 2. 변경 사항과 평가 조건

| 항목 | 사전 계획 |
|---|---|
| 대상 | QuickBird(QB), 4 MS 밴드, K=6 |
| 기준 조건 | baseline, 기존 high_frequency, 새 band_gated_hf를 같은 시드·GPU 조건으로 학습 |
| 고주파 | PAN에서 5×5 separable binomial blur를 뺀 성분. 가장자리 복제 padding |
| 새 주입 경로 | PAN 고주파로 1→16→4 Conv 잔차 생성. 마지막 Conv는 0 초기화 |
| 게이트 | PAN 고주파 4채널 복제와 기준 출력의 \|dx\|·\|dy\|, 총 12채널을 Conv 12→32→4에 입력하고 sigmoid 적용 |
| 초기 동작 | 잔차 마지막 Conv가 0이므로 학습 시작 시 후보 출력은 기준 SSA-MRN 출력과 같음. 게이트 마지막 bias는 −2 |
| 데이터 | PanCollection QB의 train_qb.h5 / valid_qb.h5와 RR·FR test H5. 기존 분할을 사용하고 원본 H5는 수정하지 않음 |
| 입력 형식·범위 | HDF5의 pan/lms/ms/gt 배열, NCHW. QB 최대 DN 2047로 나눠 학습하고 지표 계산 시 DN으로 환산 |
| 학습 | MSE, Adam, 학습률 1e−4, 100 epochs, effective batch 32, micro batch 4 gradient accumulation, CUDA FP32 |
| 시드 | 먼저 42로 Kaggle 실행을 확인한 뒤, 전체 비교는 42·43·44 paired seeds로 수행. 각 시드에서 세 변형은 같은 공통 초기화와 데이터 순서를 사용 |
| 가중치 선택 | 각 run의 최소 validation MSE인 best checkpoint. Test 지표로 epoch·모델·설정을 고르지 않음 |
| 테스트 | RR reduced-resolution 20 scenes, FR full-resolution 20 scenes 전체. 패치나 장면 일부로 평균을 대체하지 않음 |
| 평가·기록 | RR SAM/ERGAS/PSNR/SCC/Q2n 및 peak=1 MSE, FR Dλ/Ds/QNR. 장면별 CSV·JSON, 입력 SHA-256, 학습 곡선, 고정 5장면 예측 패널 |

Kaggle 입력 데이터의 실제 파일 위치·배열 크기·장면 수는 노트북 실행 때 출력되는 H5 inventory로 확인합니다. 문서의 예상 장면 수와 다르면 결과를 합치지 말고 차이를 기록합니다. GPU 기종·PyTorch 버전·실행 세션이 바뀌면 paired 조건을 다시 확인합니다.

## 3. 정량 결과

**미측정.** 학습과 전체 test 평가를 아직 실행하지 않았습니다.

| 변형 | Best epoch | Best validation MSE | RR 평균 | FR 평균 |
|---|---:|---:|---|---|
| baseline | N/A | N/A | N/A | N/A |
| 기존 high_frequency | N/A | N/A | N/A | N/A |
| band_gated_hf | N/A | N/A | N/A | N/A |

RR의 PSNR은 저장소 구현에 따라 밴드별 peak 2047 PSNR을 평균합니다. 별도로 정규화 peak=1 MSE를 기록합니다. FR에는 정답 영상이 없으므로 RR 정답 지표를 FR 성능처럼 해석하지 않습니다.

## 4. 직전 연구와 수치 차이

직전 QB 구조 비교는 A6000·시드 42 조건에서 수행했습니다. 그때 baseline 대비 기존 고정 고주파 경로는 RR PSNR 37.4242→37.3621 dB(−0.0621 dB), SAM 4.9639→4.9408°(−0.0231°), 잠정 FR QNR 0.914868→0.910940(−0.003928)이었습니다. 일부 지표만 좋아진 결과여서 고정 주입의 부작용을 새 가설의 출발점으로 삼습니다.

이번 Kaggle run은 GPU와 소프트웨어 환경이 다를 수 있습니다. 따라서 위 숫자는 근거·맥락으로만 제시하며 새 Kaggle 수치와 직접적인 통제 비교 또는 개선량으로 해석하지 않습니다. 같은 Kaggle run 안의 baseline·기존 경로·게이트 경로 차이를 주 비교로 사용합니다. QB test 장면은 이전 연구에도 사용되어 독립 검증 test가 아닙니다.

<a id="graphs"></a>

## 5. 그래프

아직 그래프는 없습니다. 노트북은 실제 측정 후 epoch별 train MSE, validation MSE, validation band-mean PSNR(peak=1), validation SAM 곡선과 RR/FR 기준선 대비 그래프를 생성하도록 준비했습니다. 값이 없는 에폭을 추정해 채우지 않습니다.

<a id="images"></a>

## 6. 결과 이미지 예시

아직 생성한 예측 이미지는 없습니다. RR test에서 사전에 고정한 서로 다른 5개 H5 row index의 전체 장면을 사용합니다. 노트북은 seed 42로 index를 수치 평가 전에 고정하고 selection.json에 seed·index·순서·full-scene 규칙을 보존합니다. 비기준 변형의 패널은 LR MS · PAN · 예측 · 정답 순서이며, RGB 표시 밴드 [2,1,0]은 표시를 위한 가정입니다. MS에는 최근접 확대와 공통 GT 대비를 적용합니다. FR에는 GT 패널을 만들지 않습니다.

<a id="evidence"></a>

## 7. 가중치와 검증 근거

아직 학습 가중치·결과 JSON·그래프·이미지·manifest가 없습니다. 학습 후 Kaggle output 경로 /kaggle/working/ctrS_band_gated_hf에 run별 best.pt/latest.pt, history.jsonl, complete.json, test_metrics.json, per_sample_metrics.csv, selection.json, 이미지와 artifact_manifest.json을 보관합니다. 결과 ZIP은 /kaggle/working/ctrS_band_gated_hf_results.zip에 생성됩니다.

실제 run의 weight epoch·SHA-256, 실행 commit/Notebook Version, 데이터 경로·해시와 전체 산출물 보존 여부는 실행 후에만 기록합니다. 현재 브랜치의 계획 문서를 학습 완료나 가중치 보관 증거로 사용하지 않습니다.

## 8. 한계와 다음 판단

이번 실험은 QB 한 센서에서 PAN 고주파 주입을 선택적으로 만드는 구조 가설을 시험합니다. 같은 QB test 장면이 과거 탐색에 사용됐으므로 결과로 독립적인 일반화 성능이나 새로운 state-of-the-art를 주장하지 않습니다. FR Ds/QNR은 scikit-image PAN resize가 MATLAB과 일치하는지 검증되지 않아 잠정 지표입니다.

먼저 seed 42 세 변형의 실행·저장·전체 평가 흐름을 확인합니다. 이후 가능한 경우 세 paired seeds를 완료하고 validation 선택 규칙을 유지합니다. 게이트 후보가 기준선과 기존 고정 경로 모두에 대해 반복적으로 균형 잡힌 개선을 보이지 않으면 채택하지 않습니다. Test 결과를 보고 게이트 구조나 하이퍼파라미터를 다시 조정하지 않습니다. 효과가 불분명하면 새 독립 데이터·장면 또는 센서 확장 실험을 계획합니다. 구조·손실·23탭을 한 번에 결합하지 않습니다.
