# QB · 3단계 공유 복원 B1/B2 · Kaggle 결과

[실험 목록](../../README.md#experiments) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [가중치·근거](#evidence)

| 비교 기준 | 이번에 바꾼 점 | 관측된 차이 | 판단 |
|---|---|---|---|
| 같은 QB K6·seed42의 B1 3단계 공유 복원 | B2에 입력 MS − 고정 MTF로 축소한 현재 복원 오차 전달 | RR PSNR +0.115759 dB, SAM -0.018282°, MSE -5.36438208154e-06; 잠정 FR QNR -0.018221 | RR에서는 관측 오차의 추가 이득 확인. FR 왜곡·입력 일치도는 악화해 종합 채택 보류 |

## 1. 목적과 상태

**B1·B2 각각 100에폭 학습과 RR 20장·FR 20장 전체 평가 완료.** 마지막 확인은 **2026-10-09 05:59 KST**, 다운로드 파일 검증 시각이며 실제 학습 종료 시각이나 서버의 실시간 상태를 뜻하지 않는다. 기존 B0은 이 셀에서 재학습하지 않았다.

검증할 가설은 반복 계산만 하는 B1보다 관측 오차를 전달하는 B2가 개선되는지이다. 단일 출력 SSA-MRN과 구별해 공유 가중치로 3번 반복한다. 둘의 본체·오차 encoder·학습형 계수 초기 상태, 100에폭 샘플 순서 해시가 일치한다. train은 최종 출력 MSE만 사용한다.

재현 코드는 [한 셀 Python](../../scripts/kaggle_b1_b2_one_cell.py) · [한 셀 notebook](../../scripts/kaggle_b1_b2.ipynb)이며 코드 보관 commit은 `f0c97568e9285ea671d33b734ca43f27a17234cd`이다. 기반 CtrS commit은 `be02b7cbf03fe00681194ae9fc9419661f24cd56`, 공식 network commit은 `a4ca40e407b12bf4c30f804384405ce321d11c51`이다. 실제 실행된 [worker](../assets/kaggle_b1_b2/code/kaggle_worker.py)·[model](../assets/kaggle_b1_b2/code/B2_models.py)와 재현 셀의 내장 문자열 SHA-256이 일치한다. Kaggle Notebook Version ID와 정확한 종료 시각은 산출물에 없어 미기록이다. 로컬에서 학습·원본 H5 평가를 다시 실행한 결과가 아니다.

## 2. 변경 사항과 평가 조건

| 항목 | 실제 조건 |
|---|---|
| 입력·범위 | QuickBird 4 MS 밴드·PAN 1밴드, NCHW, DN2047로 정규화, K6 |
| train / validation | 17,139 / 1,905 패치, HR 64×64·LR 16×16 |
| 독립 RR / FR | 각각 별도 H5 20장 전체, RR HR 256×256·LR 64×64, FR HR 512×512·LR 128×128 |
| 초기값·3단계 | 제공 LMS에서 시작. 같은 SSA-MRN core와 Conv error encoder를 3회 공유 |
| B1 / B2 | B1 error=0. B2 error=MS−D(current), LR 경계 5픽셀 mask 후 bilinear 확대 |
| 상태 업데이트 | conditioned=current+encoder(error), proposal=core(PAN,conditioned,MS), current←current+α(proposal−current) |
| 보정 계수 | 공유 scalar α, 초기 0.1, 학습됨. 계수에 범위 제한은 없음 |
| 오차 encoder | 4→4, 3×3 Conv, bias 포함. B1의 0 입력에서도 bias는 학습 가능 |
| 고정 관측 모델 | QB GNyq [0.34,0.32,0.30,0.22], 41×41 MTF FIR·replicate padding, ×4 decimation |
| train 위상 | 원본 TRAIN GT/MS 전체 정합 검사 후 패치별 (2,2)/(1,2)/(2,1) 고정. GT는 추론에 전달하지 않음 |
| validation/test 위상 | 입력 LMS/MS만으로 후보 위상 선택 후 반복 중 고정. test GT를 위상 선택에 사용하지 않음 |
| 학습 | seed42, Adam LR1e−4, 100에폭, effective/micro batch32, 최종 출력 MSE |
| 실행 | T4×2, GPU0=B1/GPU1=B2, AMP FP16·GradScaler, sensor 계산 FP32, channels-last, deterministic=True, TF32=False |
| 데이터·배치 | 원본 H5→GPU FP32 직접 적재, train/val GPU 상주, CUDA randperm/index_select, 중간 정규화 디스크 캐시 없음 |
| 환경 | PyTorch2.11.0+cu128, CUDA12.8, Python3.13.15 |
| 선택·평가 | 최소 FP32 validation MSE의 best. Test FP32 전체 장면, 평가 crop·재정합 없음 |

[데이터 경로·크기·해시](../assets/kaggle_b1_b2/data_manifest.json) · [환경](../assets/kaggle_b1_b2/environment.json) · [실행 변경 기록](../assets/kaggle_b1_b2/plan_deviations.json) · [관측·위상 manifest](../assets/kaggle_b1_b2/observation_train.json)

TRAIN 17,139개 모두 MTF 정합 검사를 통과했다. FP64 GPU FFT 검사에서 RMSE 약 2.244e−13 DN, 최악 패치 RMSE 9.805e−13 DN이며 사전 기준은 전체 RMSE≤1 DN·최악≤2 DN이다. 이 결과는 합성 TRAIN의 패치 내부 정합 검증이며 FR 센서 물리 타당성 검증이 아니다.

<a id="results"></a>

## 3. 정량 결과

각 모델 RR·FR 전체 20장 평균이다. PSNR은 peak2047의 장면별 밴드 PSNR 평균, MSE는 DN2047 정규화 원소 MSE, SAM은 도 단위다. best는 FP32 validation MSE로 선택한다.

| 모델 | Best epoch | Best validation MSE | RR PSNR dB ↑ | RR SAM ° ↓ | RR MSE peak1 ↓ |
|---|---:|---:|---:|---:|---:|
| B1 · 오차 없이 반복 | 90 | 0.000164994460647 | 37.604745 | 4.832698 | 0.000205615557702 |
| B2 · 관측 오차 전달 | 99 | 0.000163592107128 | 37.720504 | 4.814416 | 0.00020025117562 |
| B2−B1 | — | -1.40235351864e-06 | +0.115759 | -0.018282 | -5.36438208154e-06 |

| 지표 | B1 | B2 | B2−B1 |
|---|---:|---:|---:|
| RR ERGAS | 4.02901281269 | 3.97521540769 | -0.0537974050017 |
| RR SCC | 0.977928571296 | 0.978905877718 | 0.000977306421946 |
| RR Q2n | 0.924317338985 | 0.926348050797 | 0.00203071181194 |
| RR MS_consistency_MSE_peak1_interior | 1.93308246992e-06 | 1.80390329518e-06 | -1.29179174735e-07 |
| RR inference_ms | 67.450853157 | 69.1594551086 | 1.7086019516 |
| FR D_lambda | 0.0455222870036 | 0.0500101010164 | 0.00448781401277 |
| FR D_s | 0.0526757668321 | 0.0671829618592 | 0.014507195027 |
| FR QNR | 0.904310112616 | 0.886088768371 | -0.0182213442458 |
| FR MS_consistency_MSE_peak1_interior | 0.000134618213997 | 0.00016363571558 | 2.90175015834e-05 |
| FR inference_ms | 227.438302612 | 275.184449768 | 47.7461471558 |

RR MSE는 B1 대비 **2.61% 감소**했다. RR의 PSNR·SAM·ERGAS·SCC·Q2n이 함께 개선됐다. 반면 FR Dλ·Ds가 모두 증가했고 QNR은 감소했다. FR 입력 MS 일치도 MSE도 **21.56% 증가**했다. 입력 일치도는 HR GT 정확도가 아니며, FR QNR·Ds와 RR Q2n·SCC는 저장소의 MATLAB parity 제한을 유지한다.

[전체 수치](../assets/kaggle_b1_b2/results.json) · [CSV](../assets/kaggle_b1_b2/summary.csv) · [B2−B1 paired 차이](../assets/kaggle_b1_b2/paired_comparison.json) · [B1 RR 장면별](../assets/kaggle_b1_b2/runs/B1_QB_B1_k6_s42/RR_per_scene.json) · [B2 RR 장면별](../assets/kaggle_b1_b2/runs/B2_QB_B2_k6_s42/RR_per_scene.json) · [B1 FR 장면별](../assets/kaggle_b1_b2/runs/B1_QB_B1_k6_s42/FR_per_scene.json) · [B2 FR 장면별](../assets/kaggle_b1_b2/runs/B2_QB_B2_k6_s42/FR_per_scene.json)

학습+validation 누적은 B1 **155.78분**, B2 **188.84분**으로 B2가 **21.23%** 더 오래 걸렸다. 병렬 실행이므로 두 시간을 합해 벽시계 시간으로 표시하지 않는다. 시동·checkpoint·test는 이 합계 밖이며 [세션 시간](../assets/kaggle_b1_b2/runs/B2_QB_B2_k6_s42/session_timing.json)에 별도 기록돼 있다. 추론은 다운로드 결과의 단일 장면 CUDA event 시간이고 추가 MTF 지표·CPU 복사·평가·저장은 제외한다. 별도 warm-up이 없어서 안정적인 latency benchmark로 해석하지 않는다.

## 4. 직전 연구와 수치 차이

직전 같은 Kaggle QB [게이트 고주파 실험](kaggle_band_gated_hf.md)의 원본 baseline을 B0 맥락으로 기록한다. **B0 재학습·재평가 없이 보관된 값을 인용**했다. train/val/RR/FR H5 SHA-256 4개는 동일하고, K6·seed42·100에폭·effective/micro32·AMP 조건도 같다. 다만 실행 세션·배치 순서 생성 위치·출력 구조·계수·추가 encoder가 달라 아래 수치만으로 반복 횟수의 독립 효과를 확정할 수 없다. 주 결론은 이번 B2−B1 비교다. 최근 MTF 증강 실험은 seed46 validation만 있어 독립 test 비교 대상으로 사용하지 않았다.

| 모델 | RR PSNR dB | RR SAM ° | RR MSE peak1 | B0 대비 ΔPSNR | ΔSAM | ΔMSE |
|---|---:|---:|---:|---:|---:|---:|
| 직전 B0 · baseline | 37.453879 | 4.955443 | 0.000215111126896 | — | — | — |
| B1 | 37.604745 | 4.832698 | 0.000205615557702 | +0.150865 | -0.122744 | -9.49556919417e-06 |
| B2 | 37.720504 | 4.814416 | 0.00020025117562 | +0.266624 | -0.141027 | -1.48599512757e-05 |

<a id="graphs"></a>

## 5. 그래프

실측 100에폭의 train MSE와 FP32 validation MSE·PSNR·SAM이다. 누락 에폭을 추정해 채운 값은 없다.

![B1/B2 학습·validation 곡선](../assets/kaggle_b1_b2/learning_seed42.png)

![B2−B1 RR/FR 전체 test 차이](../assets/kaggle_b1_b2/test_comparison_seed42.png)

<a id="images"></a>

## 6. 결과 이미지 예시

수치 평가 전에 seed42로 고정한 RR row index **1, 8, 11, 13, 19**의 전체 장면, 오름차순이다. [selection](../assets/kaggle_b1_b2/selection.json)에 seed·ID·full scene·순서가 있다. 왼쪽부터 **LR MS · PAN · B2 예측 · GT**이며, RGB [2,1,0]은 표시 가정이다. 같은 장면 GT의 1–99% 대비를 LR·예측·GT에 공통 적용하고 LR만 최근접 확대한다. B1은 수치 비교에 남기며 보고서 패널에는 B2를 표시한다. 원본 B1 예시도 산출물 폴더에 보존했다. FR에는 GT가 없다.

![B2 RR 장면 1](../assets/kaggle_b1_b2/runs/B2_QB_B2_k6_s42/RR_scene_01.png)

<details>
<summary>같은 조건의 나머지 고정 4장면 보기</summary>

![B2 RR 장면 8](../assets/kaggle_b1_b2/runs/B2_QB_B2_k6_s42/RR_scene_08.png)

![B2 RR 장면 11](../assets/kaggle_b1_b2/runs/B2_QB_B2_k6_s42/RR_scene_11.png)

![B2 RR 장면 13](../assets/kaggle_b1_b2/runs/B2_QB_B2_k6_s42/RR_scene_13.png)

![B2 RR 장면 19](../assets/kaggle_b1_b2/runs/B2_QB_B2_k6_s42/RR_scene_19.png)

</details>

<a id="evidence"></a>

## 7. 가중치와 검증 근거

실제 best/latest 원본 `.pt` 4개와 전체 예측 80개를 Git에 포함한다. checkpoint 내부 config·optimizer·RNG를 수정하지 않았다. 다운로드 원본 manifest의 152개 파일 SHA-256을 확인했고, Python 바이트코드만 저장소에서 제외했다. 보관본의 `.pt`·`.npz` ZIP CRC와 PNG를 검증했다. 로컬 PyTorch state_dict 로딩이나 원본 H5 재평가를 했다는 뜻은 아니다.

| 모델 | 가중치 | Epoch | SHA-256 |
|---|---|---:|---|
| B1 | [best.pt](../assets/kaggle_b1_b2/runs/B1_QB_B1_k6_s42/best.pt) | 90 | `e89c367db3300db97672ff5fac9cd3ed0f19553800679e68028fdca01e875b44` |
| B1 | [latest.pt](../assets/kaggle_b1_b2/runs/B1_QB_B1_k6_s42/latest.pt) | 100 | `587c30f3fbf9e1ff068ef5c6b0845aedc3e45b735c8a42449ab8c3df19ed340f` |
| B2 | [best.pt](../assets/kaggle_b1_b2/runs/B2_QB_B2_k6_s42/best.pt) | 99 | `352648c01a08c621ae09a81c00e348ab6607a6afe93d413b89006d25d73420ba` |
| B2 | [latest.pt](../assets/kaggle_b1_b2/runs/B2_QB_B2_k6_s42/latest.pt) | 100 | `6bef8460631908043dca5071a6bba898bbe8b2698b34ccb38f35e67c26f5bc6f` |

[검증 기록](../assets/kaggle_b1_b2/verification.json) · [compact epoch 기록](../assets/kaggle_b1_b2/compact_epochs.json) · [원본 artifact manifest](../assets/kaggle_b1_b2/artifact_manifest.json) · [실제 보관 manifest](../assets/kaggle_b1_b2/repository_manifest.json) · [실행 소스](../assets/kaggle_b1_b2/code/) · [B1 원본 전체 예측](../assets/kaggle_b1_b2/runs/B1_QB_B1_k6_s42/predictions/) · [B2 원본 전체 예측](../assets/kaggle_b1_b2/runs/B2_QB_B2_k6_s42/predictions/)

원본 artifact manifest에는 제외한 `__pycache__` 파일도 포함돼 있다. 해당 목록은 verification에 명시하며, repository manifest는 실제 커밋 대상만 기록한다. 중간 캐시·학습 runtime·원본 H5는 복사하지 않았고 사용자가 다운로드한 폴더도 삭제하지 않았다.

## 8. 한계와 다음 판단

**판단: RR 개선 후보 유지, FR 포함 최종 채택 보류.** 한 시드에서 관측 오차 전달의 이득은 RR에서 보였으나 FR Dλ·Ds·QNR 및 입력 MS 일치도가 악화했다. 합성 TRAIN에서의 정확한 MTF 정합이 FR 모델의 타당성을 보증하지 않는다.

TRAIN은 GT/MS로 검증한 패치별 위상을 사용하고 validation/test는 입력 LMS/MS로 추정한다. 이 정책 차이와 FR 센서 모델을 후속 검토 대상으로 남긴다. test는 선행 구조 탐색에 사용한 동일 QB split이며 완전히 새로운 확증 데이터가 아니다. 단일 시드·서로 다른 best epoch·B0 별도 세션·추론 warm-up 부재 제한이 있다. 다음 판단은 test로 새 설정을 선정하기보다 반복 시드와 독립 데이터, validation 중심의 센서/위상 검증을 먼저 수행한다.
