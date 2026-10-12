# MTF 변화량 데이터 증강

**변경 내용:** 센서 열화 특성의 변화량을 학습 데이터에 추가했습니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|
| 100에폭 validation 기록 확보; 이 보고서의 RR·FR 결과는 미확정 | 학습 완료·평가 대기 | **판단 보류** |


> **이력 구분:** 이 보고서는 당시 **K=6**으로 수행한 실험입니다. 현재 연구 기본값은 [K=4](../../README.md#decision)로 확정했으며, 아래 실제 설정·수치·가중치는 당시 K6 기록을 유지합니다.

[실험 목록](../../README.md#experiments) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [가중치·근거](#evidence)

| 비교 기준 | 바꿀 점 | 관측 차이 | 판단 |
|---|---|---|---|
| 원본 PAN/MS/LMS로 학습하는 SSA-MRN K6 | TRAIN MS의 검증된 내부에 MTF 변화량 추가, LMS도 함께 갱신 | 100에폭 완료; best validation PSNR +0.00576 dB, SAM −0.00499° | 최종 best/latest 보관·검증, RR/FR 평가 대기 |

## 1. 목적과 상태

흐림 조건 변화에 대한 SSA-MRN의 민감도를 줄이는지 확인합니다. QB 단일 paired seed 46을 첫 탐색으로 사용하며, 개선을 확정하려면 반복 시드와 다른 조건의 평가가 필요합니다.

**완료 확인:** 2026-10-09 01:25 KST. baseline과 mtf_aug 모두100에폭·exit code0, validation MSE 최소 checkpoint는 둘 다99에폭입니다. 실제 Kaggle ZIP 다운로드와 전체 파일 해시를 검증했습니다. 학습 완료이며 전체 RR/FR 연구 평가는 아직 미완료입니다.

**실행 파일:** 한 셀 Notebook (보존된 자료는 재현 자료 절을 확인합니다.) 또는 [한 셀 Python 코드](run.py). Notebook에는 코드 셀 하나만 있으며 모델·공식 네트워크·관측 프로파일·worker 코드 스냅샷을 포함합니다. 원격 브랜치의 미완료 변경에 의존하지 않습니다. 기본 코드 기준은 main `be02b7c`입니다.

실행 방법:

1. Kaggle의 **File → Import Notebook**으로 위 `.ipynb`를 가져옵니다.
2. **GPU T4 x2**를 선택하고 기존 `biyonghiyon/ctrs-pan-sharpening-dataset`을 Add Input으로 연결합니다.
3. 코드 셀 한 개를 실행합니다. 기본값 `SMOKE=False`는 **두 모델의 본 학습 시작**입니다. 먼저 환경 확인만 하려면 `SMOKE=True`로 32패치·1에폭을 별도 폴더에서 실행합니다.
4. 장시간 실행은 **Save Version → Save & Run All**로 시작하고 출력 저장 여부를 확인합니다. 이 기능은 새 실행을 시작하므로 인터랙티브 학습을 함께 켜두지 않습니다. 세션별 기본 학습 시간 예산은 9시간이며, 저장 후 정상 종료하고 ZIP을 생성합니다.
5. 실제 GPU 속도와 최종 완료 시각은 첫 에폭 이후 측정합니다. 9시간은 완료 보장이 아닌 세션당 실행 한도입니다.

## 2. 변경한 구조와 비교 조건

| 항목 | 조건 |
|---|---|
| 데이터 | QB 원본 train/validation 전부; 수정·재업로드 없음 |
| GPU 배정 | GPU 0: baseline / GPU 1: mtf_aug, 독립 프로세스 |
| 공통 모델 | SSA-MRN K6, 내부 bilinear 유지, 구조·고주파 경로·보조 손실 추가 없음 |
| 공통 학습 | seed 46, Adam lr 1e-4, MSE, 목표 100에폭, effective batch 32, micro batch 16 |
| 속도 설정 | FP16 autocast + GradScaler, 공통 cuDNN autotune, 기본 GPU 상주 캐시; 공간 부족 시 mmap + worker 2로 fallback |
| 검증 | 원본 validation, 두 모델 모두 FP32 추론, MSE·밴드 평균 PSNR(peak=1)·SAM 기록 |
| 선택 | 최소 validation MSE의 best; 재개는 latest |
| MTF 변경 | QB 밴드별 GNyq `[0.34,0.32,0.30,0.22]`의 공통 배율 0.9 또는 1.1; 41탭 signed FIR 유지 |
| 증강 선택 | 샘플·에폭·시드로 결정하는 원본/0.9/1.1의 3개 상태. test/validation으로 선택하지 않음 |
| 경계·위상 | 공식 TRAIN의 고정 위상 `(2,2)/(1,2)/(2,1)` 사용. LR 경계 5픽셀은 증강하지 않음 |

### 잘린 학습 패치에서의 증강 정의

64×64 GT 패치에는 원래 영상의 주변 정보가 없습니다. 전체를 새 MTF로 재생성하면 원본의 필터링·잘림 순서와 경계가 달라져 기준선과의 비교를 흐릴 수 있습니다. 따라서 다음의 **패치 내부 변화량 증강**을 사용합니다.

```text
ΔMS = 내부 마스크 × [D변경(GT) − D기준(GT)]
MS증강 = 원본 MS + ΔMS
LMS증강 = 원본 LMS + interp23(ΔMS)
```

원본 상태는 MS/LMS가 완전히 동일합니다. PAN과 GT는 유지합니다. 41탭 필터의 주변 정보가 유효한 LR 중앙 6×6만 바꾸고, 동일 변화량을 선형 23탭 확대해 LMS에도 반영합니다. 23탭은 LMS 변화량 계산에만 사용하며 SSA-MRN 내부 확대층을 교체하지 않습니다.

실행 시 전체 TRAIN의 기준 관측값과 원본 MS 내부를 대조합니다. 최대 패치 RMSE가 2 DN을 넘으면 증강을 중단합니다. 0.9/1.1 캐시는 GPU FP32로 한 번 계산하고 재사용합니다. 매 배치에서 41탭 필터를 다시 실행하지 않습니다.

**한계:** 이 방식은 전체 영상에서 다시 만든 연속·비등방성 MTF 데이터와 다릅니다. 미지의 FR 센서 물리를 재현했다고 주장하지 않습니다. 경계는 원본에 고정되어 있어 일반화 효과가 제한될 수 있습니다. 전체 영상의 안전한 재생성이 가능해지면 별도 실험으로 비교합니다.

### CPU 데이터 처리 병목 개선 · 2026-10-09

`GPU_RESIDENT=True`가 기본입니다. CPU mmap 캐시를 한 번 읽어 각 GPU에 TRAIN/validation을 올립니다. 증강 모델은 MS/LMS 변화량 캐시도 함께 올립니다. 이후 샘플 선택·변화량 합산을 GPU 텐서에서 처리하여 매 배치 CPU 패치 복사·콜레이션·Host→Device 전송과 DataLoader worker 작업을 줄입니다. 모델·패치·micro/effective batch·시드·AMP·손실·순서는 유지합니다.

GPU 데이터 버퍼는 QB 전체 배열 형태 기준 baseline 약 2.69 GiB, 증강 약 4.91 GiB입니다. 이는 데이터 텐서만의 예상치이며 모델·활성값·cuDNN 공간은 추가됩니다. 로딩 전에 필요한 바이트 수와 실제 여유 메모리를 대조하고 3 GiB의 여유를 남깁니다. 부족하면 기존 CPU 경로를 사용합니다. 단순 메모리 점유율 대신 진행률 표의 `패치/초`로 속도를 비교합니다. 첫 로딩·cuDNN 준비 시간과 중간 재개 시 짧은 구간은 비교에서 구분합니다.

loss 누계를 장치 텐서로 유지해 매 micro batch마다 `.item()`을 호출하지 않습니다. AMP scale을 매 optimizer step마다 두 번 조회하던 코드는 optimizer의 실제 step hook으로 교체해 skipped step을 기록합니다. 유한 loss 검사와 실제 저장·표 갱신 때의 동기화는 유지합니다.

기존 worker SHA-256 `6d920ca4dfc03fdda0228567eeab167888d799acf471477a03717ea6f222eb9d`에서의 이관만 허용합니다. 이전·현재 코드 차이가 검증된 캐시/동기화 경로이고, 나머지 코드·데이터·학습 조건이 같을 때 기존 fingerprint를 정확히 유도해 latest를 복원합니다. 시드·batch·AMP·모델이 바뀌면 이관을 거부합니다. 기존 코드 snapshot을 남기고 새 snapshot과 `resume_migration.json`을 추가합니다. 같은 세션에 검증된 이전 mmap/증강 캐시가 있으면 재사용합니다.

**적용:** 현재 셀에서 Cancel Run을 한 번 누르고 저장·종료 메시지를 기다립니다. 세션과 `/kaggle/working`을 유지한 채 같은 셀을 새 Python 코드로 교체하고 실행하면 latest부터 재개합니다. 커널 재시작이나 새 세션 전에는 출력 보관을 먼저 확인합니다. 새 세션에서는 기존 재개 절차대로 이전 Output을 연결합니다.

실제 복원 SSA-MRN·합성 데이터의 CPU 검증에서 원본/증강 모든 선택의 상주 텐서 배치가 기존 샘플 경로와 정확히 일치했습니다. 두 모델 각각 기존 구현의 에폭 중간 체크포인트 → 새 상주 배치 경로로 이어 학습했을 때 연속 학습 가중치와 validation MSE가 정확히 일치했습니다. 이는 CUDA AMP의 비트 단위 일치나 T4 속도 개선 실측을 뜻하지 않습니다. 실제 T4×2에서 GPU 상주 경로가 적용된 채 두 모델이100에폭을 종료했습니다. 중간 진행 표에서 약440–465 패치/초, allocated baseline2.70 / MTF4.92 GiB를 관측했습니다. 이전 경로와 동일 조건의 속도 비교가 없어 최적화 배율은 주장하지 않습니다.

## 3. 정량 결과

<a id="results"></a>

본 학습100에폭 완료. 원본 validation에서 각 모델의 최소 MSE checkpoint를 선택했습니다. 아래는 validation 결과이며 RR/FR test 결과가 아닙니다. PSNR은 peak1의 밴드별 평균, SAM은 유효 픽셀의 각도 평균(도)입니다.

| 모델 | 완료 / best 에폭 | Best validation MSE | PSNR (dB) | SAM (°) | RR/FR |
|---|---:|---:|---:|---:|---|
| baseline | 100 / 99 | 0.000171972981 | 39.310003 | 4.550377 | 미평가 |
| mtf_aug | 100 / 99 | 0.000171249088 | 39.315762 | 4.545383 | 미평가 |
| MTF − baseline | — | −0.000000723893 | +0.005759 | −0.004994 | N/A |

[검증·수치 원본](results/preservation.json). validation MSE는 약0.421% 낮아졌입니다. 단일 seed의 작은 관측 차이이며 분광·FR 일반화 개선을 확정하지 않습니다. AMP skipped step은 누적 baseline8 / MTF10으로 기록됐습니다. best의 optimizer·scaler·RNG 상태도 원본대로 보존했습니다.

## 4. 기준선 대비 차이와 해석

23탭 후속 연구의 QB RR 개선이 FR에서 일관되지 않았던 관측이 출발점입니다. 새 실험의 변경점은 학습 입력의 MTF 변화량이며, 기존 23탭 구조나 밴드별 고주파 게이트와 결합하지 않습니다.

FP16·micro batch 16·T4 조건이 기존 FP32 연구와 달라 **이번 T4의 두 모델끼리만 통제 비교**합니다. BatchNorm은 실제 micro batch 단위로 동작하므로 gradient accumulation이 batch 32의 BatchNorm과 동등하다는 주장을 하지 않습니다. best validation 기준 MSE −7.23893e-7, PSNR +0.005759 dB, SAM −0.004994°를 관측했습니다. 기존23탭 연구의 test PSNR과 이 validation 차이는 직접 비교하지 않습니다. test는 기존 탐색에 사용됐으며 모델·에폭 선택에 재사용하지 않습니다.

## 5. 학습·평가 그래프

<a id="graphs"></a>

실제100에폭 history의 train MSE, validation MSE·PSNR·SAM 곡선을 보존했습니다.

![MTF100에폭 학습·validation 곡선](results/validation_curves.png)

[baseline 기록](results/baseline/history.json) · [MTF 기록](results/mtf_aug/history.json). 이어진 세션의 체크포인트에는 전체 이력을 포함합니다. 세션 간 부분 에폭의 소요 시간은 전체 에폭 시간으로 해석하지 않습니다.

## 6. 결과 이미지

<a id="images"></a>

현재 파일은 학습·재개 실행기입니다. 전체 RR/FR test 평가 및 고정 5장면 예측 이미지는 후속 평가 단계에서 생성해야 합니다. 학습 완료만으로 연구 결과 완료를 기록하지 않습니다. FR의 QNR은 기존 MATLAB resize 일치성 검증 상태를 함께 기록합니다.

## 7. 재현 자료

[공통 실행·평가 안내](../../README.md#evaluate) · [파일 해시](manifest.json) · [실행 코드](run.py) · [가중치](checkpoints/) · [수치·그래프·예시](results/) · [원본 실행 근거](provenance/)

다음 명령으로 저장된 가중치를 확인합니다. 실제 test 평가는 `--h5`, `--protocol`, 새 `--output` 경로를 함께 지정합니다.

```bash
python SSA-MRN/common/checkpoint.py --experiment 8 --checkpoint checkpoints/baseline/best.pt
```

`run.py`는 Kaggle 한 셀 실행 코드입니다. 파일 전체를 셀에 넣고 상단 설정을 지정합니다. 학습 당시 고정된 소스 경로는 재현을 위해 유지했습니다.

## 8. 개선 여부와 다음 단계

**판정: 판단 보류.** 위 결과는 해당 보고서의 비교 조건에 한정합니다. 지표 일부의 개선을 최종 모델 채택으로 해석하지 않습니다.

첫 비교는 단일 시드 탐색입니다. validation에서 후보를 고르고 그 규칙을 고정한 뒤 전체 RR/FR, 미리 정한 흐림 변화 조건, 반복 시드를 평가합니다. QB test는 이전 탐색에서도 사용했으므로 새 독립 test라고 주장하지 않습니다. 결과를 보고 증강 강도를 반복 변경하는 방식은 피합니다.

선행 아이디어: [Spatial Data Augmentation, TGRS 2023](https://doi.org/10.1109/TGRS.2023.3262262). 본 코드는 해당 논문의 전체 증강 기법 재현이 아니라, 원본 패치를 보존하는 제한된 변화량 증강 비교입니다.
