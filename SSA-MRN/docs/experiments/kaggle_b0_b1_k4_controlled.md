# QB · B0/B1 K4 통제 재학습 · Kaggle 결과

[실험 목록](../../README.md#experiments) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [가중치·근거](#evidence)

| 비교 기준 | 이번에 바꾼 점 | 관측된 차이 | 판단 |
|---|---|---|---|
| 같은 실행에서 새로 학습한 원본 B0 | B1 3단계 공유 복원, K4·동일 초기 core·배치 순서 | RR PSNR +0.300014 dB, SAM −0.107839°, MSE −1.55227536e−5; 잠정 FR QNR −0.039261 | RR 반복 복원의 이득 확인, FR 포함 종합 채택 보류 |
| [과거 K6 B0/B1](kaggle_b1_b2_100_resume.md) | K4, B0도 동시 새 학습·동일 fused Adam으로 통제 | B1의 K4−K6 PSNR −0.019324 dB, SAM +0.016031°, MSE +1.20286534e−6 | 데이터·복원식은 같지만 전체 조건은 다름. K만의 효과로 해석하지 않음 |

> **실제 K4 결과다.** 과거 K6 보고서·가중치·수치는 유지한다. 같은 연구 묶음의 별도 통제 실험이며, 두 실험을 동일 실행이나 반복 시드 평균으로 합치지 않는다.

## 1. 목적과 상태

기존 K6의 B1−보관 B0 비교는 별도 세션·optimizer·배치 생성·재개 조건이 달랐다. 현재 기본값 K4에서 B0/B1을 처음부터 같은 조건으로 학습해 반복 복원의 성능 이득을 확인했다. **두 모델 각각100에포크, best로 RR20장·FR20장 전체 평가 완료.** seed43은 실행하지 않았다.

Run ID는 `ssa_B0_B1_K4_s42_20261011T090100414462Z`. 모델 status에 기록된 평가 완료 시각은 B0 **2026-10-11 18:51:14 KST**, B1 **20:25:29 KST**다. 사용자 다운로드 산출물의 검증 날짜는 **2026-10-11 KST**이며 실시간 서버 상태 조회는 아니다. Notebook Version ID는 산출물에 없다. 제공한 다운로드 원본은 수정·삭제하지 않았다.

실행 기반은 CtrS `be02b7cbf03fe00681194ae9fc9419661f24cd56`, 공식 upstream `a4ca40e407b12bf4c30f804384405ce321d11c51`. 보관 branch는 `ssa-b0-b1-k4-results`이고 출발 main은 `128e284`다. 실행 당시 모델·worker 원문은 [code snapshot](../assets/kaggle_b0_b1_k4_controlled/code/)에 보존했다. [현재 단일 셀](../../scripts/kaggle_b0_b1_k4_controlled_one_cell.py)은 실행 후 보완한 launcher를 포함하며 **내장 model/worker는 snapshot과 바이트 동일**하다. 실행 당시 launcher 전체가 제공됐다고 주장하지 않는다.

## 2. 변경 사항과 평가 조건

| 항목 | 실제 조건 |
|---|---|
| 입력·데이터 | QB PAN1·MS4밴드, DN2047 정규화, train17,139 / validation1,905 패치 |
| 해상도 | train/val HR64×64·LR16×16; RR HR256×256·LR64×64; FR HR512×512·LR128×128 |
| B0 | 원본 SSA-MRN 단일 출력 adapter, K4; 23탭·추가 손실·관측 오차 피드백 없음 |
| B1 | 동일 core·4→4 3×3 Conv error encoder·scalar α를 3회 공유; observation error=0 |
| B1 갱신식 | conditioned=current+encoder(0); proposal=core(PAN,conditioned,MS); current←current+α(proposal−current), current 초기값=LMS |
| encoder / α | bias 학습 가능, α 초기0.1·범위 제한 없음; best α=0.114275217, latest α=0.111207850 |
| optimizer / loss | 두 모델 fused Adam, betas=(0.9,0.999), eps=1e−8, weight_decay=0, lr1e−4 고정, 최종 출력 MSE |
| seed / batch / budget | seed42, effective=micro32, 마지막 불완전 batch 포함, 각각100에포크 새 학습 |
| 실행 | T4×2, GPU0=B0/GPU1=B1; AMP FP16+GradScaler, channels-last; strict deterministic·TF32=False·CUBLAS :4096:8 |
| 환경 | PyTorch2.11.0+cu128, CUDA12.8, Python3.13.15, Tesla T4 |
| best / test | FP32 validation MSE 최소, 동률이면 먼저 선택한 best 유지; best FP32로 전체 장면, 추가 crop·재정합 없음 |
| GPU 데이터·저장 | train/val FP32 GPU 상주, GPU randperm/index_select; 약60초·배치 경계·에포크 저장, 모델/optimizer/scaler/RNG/순서/커서/누적loss 보존 |

초기 core SHA-256은 양쪽 모두 `4efb4367e985d49f043a1014bb10f9e249415e5db2e96f0774df525575088e34`다. [초기화 증거](../assets/kaggle_b0_b1_k4_controlled/initialization_pair.json)와 checkpoint history의100개 sample-order SHA-256이 일치한다. 양쪽 config는 `resume=false`이고 원시 worker 기록에도 resume 이벤트가 없다. 과거 체크포인트를 가져온 실행으로 표시하지 않는다.

원본 upstream에서 정의되지 않은 `ms` 입력을 기존 adapter가 명시한다. B0는 논문 표의 수치나 논문 bitwise 재현이 아니라 이 원본 구조 adapter의 새 실측 기준선이다. B1은 zero error에서 encoder weight가 출력에 기여하지 않지만 bias는 학습된다. 과거 B1 zero-error 분기의 세 번 갱신식과 일치하며, 사용하지 않는 observation buffer·위상·FFT를 제거했다. B2는 이번 실험에서 제외했다.

## 3. 정량 결과

<a id="results"></a>

장면별 계산 후 **20장 산술평균**이다. RR PSNR은 peak2047 밴드별 PSNR 평균, MSE는 peak1 정규화 원소 MSE, SAM은 도 단위다. FR에는 HR GT가 없다. QNR·D_s 및 Q2n/SCC의 MATLAB parity 제한은 유지한다.

| 모델 | Best epoch | Best validation MSE ↓ | RR PSNR dB ↑ | RR SAM ° ↓ | RR MSE peak1 ↓ | FR QNR ↑ · 잠정 |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 97 | 0.000172096363 | 37.371084 | 4.946089 | 0.000218160682 | 0.909722 |
| B1 | 94 | 0.000164554876 | 37.671097 | 4.838250 | 0.000202637929 | 0.870462 |
| B1−B0 | — | −7.54148641e−6 | +0.300014 | −0.107839 | −1.55227536e−5 | −0.039261 |

| 지표 | B0 | B1 | B1−B0 |
|---|---:|---:|---:|
| RR ERGAS ↓ | 4.147542 | 3.999888 | −0.147655 |
| RR SCC ↑ | 0.976649 | 0.978558 | +0.001909 |
| RR Q2n ↑ | 0.922543 | 0.925454 | +0.002911 |
| FR D_lambda ↓ · 잠정 | 0.049951 | 0.052136 | +0.002184 |
| FR D_s ↓ · 잠정 | 0.042746 | 0.081345 | +0.038599 |

B1의 validation MSE는 B0보다 **4.382% 감소**했다. RR PSNR은 **20/20장 모두 개선**했고 평균 MSE는 약7.12% 감소했다. FR에서는 두 왜곡 지표가 증가하고 QNR이 하락했다. 따라서 **RR 반복 복원의 이득은 확인했지만 FR 포함 종합 복원 향상은 확인하지 못했다.** test로 best나 조건을 다시 선택하지 않았다.

[전체 결과](../assets/kaggle_b0_b1_k4_controlled/results.json) · [B1−B0 차이](../assets/kaggle_b0_b1_k4_controlled/paired_comparison.json) · 장면별 [B0 RR](../assets/kaggle_b0_b1_k4_controlled/B0/RR_per_scene.json) / [B1 RR](../assets/kaggle_b0_b1_k4_controlled/B1/RR_per_scene.json) / [B0 FR](../assets/kaggle_b0_b1_k4_controlled/B0/FR_per_scene.json) / [B1 FR](../assets/kaggle_b0_b1_k4_controlled/B1/FR_per_scene.json).

실측 history 누적 학습/validation 시간은 B0 **43.05/4.47분**, B1 **127.52/14.21분**이다. 학습 시간에는 주기 저장·validation 직전 저장이 포함되며 에포크 끝 저장·bootstrap·test·최종 ZIP 포장은 제외된다. 두 GPU 병렬 실행 시간을 더한 값을 총 wall time으로 쓰지 않는다. [검증·실측 시간](../assets/kaggle_b0_b1_k4_controlled/verification.json)

## 4. 직전 연구와 수치 차이

기존 [K6 보고서](kaggle_b1_b2_100_resume.md)와 **완전히 동일한 실험은 아니다.** 조건 확인 결과는 다음과 같다.

| 확인 항목 | K6→이번 K4 |
|---|---|
| QB 4개 H5 해시·사용 배열 shape | 모두 동일 |
| 기반 adapter·upstream·평가 metrics 원문 | 모두 바이트 동일 |
| B1의 zero-error 복원식·3회 공유·bias·α 초기0.1 | 동일(사용하지 않는 관측 모듈 제거) |
| B1 sample order | 100에포크 해시 동일 |
| seed·100에포크·batch/micro32·lr·AMP/layout·T4·PyTorch/CUDA | 공통 설정 동일 |
| K·가중치 shape·초기 가중치 | K6→K4, 서로 다른 차원·가중치; K 사이 동일 초기화 주장 안 함 |
| B0 학습 | 이전 별도 세션·foreach Adam → 이번 새 대조군·fused Adam |
| B1 재개 | 이전200셀의 초반 상태를 가져와100에포크 완료 → 이번 처음부터100에포크 |
| 모델군·worker·checkpoint 형식 | 이전 B1/B2 → 이번 B0/B1; 전체 코드·저장 형식은 다름 |

따라서 다음은 **별도 실행의 관측 차이(K4−K6)**다. K만의 구조 효과나 통제된 K 선택 결과가 아니다.

| 모델·비교 | ΔRR PSNR dB | ΔRR SAM ° | ΔRR MSE peak1 | ΔFR QNR · 잠정 |
|---|---:|---:|---:|---:|
| 이번 B0−과거 보관 B0 | −0.082796 | −0.009353 | +3.04955532e−6 | −0.010034 |
| 이번 B1−과거 재개 B1 | −0.019324 | +0.016031 | +1.20286534e−6 | −0.012246 |

과거 K6 B1−B0는 PSNR **+0.236541dB**, SAM **−0.133224°**, MSE **−1.36760636e−5**, 잠정 QNR **−0.037049**였다. 이번 통제 K4에서도 RR 개선·FR 악화 방향은 같지만 그 개선량을 평균하거나 K4가 K6보다 우세하다고 결론내리지 않는다. [기계 검증된 조건·모델별 전체 차이](../assets/kaggle_b0_b1_k4_controlled/verification.json)

<a id="graphs"></a>

## 5. 그래프

실제100에포크 학습 MSE·validation MSE·PSNR·SAM이다. 원본 [B0 history](../assets/kaggle_b0_b1_k4_controlled/B0/history.json) · [B1 history](../assets/kaggle_b0_b1_k4_controlled/B1/history.json)를 checkpoint 내부 history와 대조했다. 없는 에포크·과거 곡선을 만들어 넣지 않았다.

![K4 B0/B1 실측 학습·validation 곡선](../assets/kaggle_b0_b1_k4_controlled/learning.png)

![K4 B0/B1 전체 RR/FR 지표 비교](../assets/kaggle_b0_b1_k4_controlled/test_comparison.png)

<a id="images"></a>

## 6. 결과 이미지 예시

학습 전에 seed42로 고정한 RR **0-based H5 index1·8·11·13·19**이며 순서는 오름차순, tile은 full scene이다. [선택 기록](../assets/kaggle_b0_b1_k4_controlled/selection.json)에 seed·scene ID·tile·순서·RGB 가정 밴드[2,1,0]를 보존했다. 왼쪽부터 **LR MS · PAN · B1 예측 · GT**이고 MS에 같은 GT 1–99% 대비, LR에 최근접 확대를 적용한다. FR에는 GT 패널을 만들지 않았다. B0도 수치와 원본 산출물에 보존하며 본문 예시는 B1을 표시한다.

![B1 고정 RR index1](../assets/kaggle_b0_b1_k4_controlled/B1/RR_scene_01.png)

<details><summary>나머지 사전 고정 4장면</summary>

![B1 RR index8](../assets/kaggle_b0_b1_k4_controlled/B1/RR_scene_08.png)

![B1 RR index11](../assets/kaggle_b0_b1_k4_controlled/B1/RR_scene_11.png)

![B1 RR index13](../assets/kaggle_b0_b1_k4_controlled/B1/RR_scene_13.png)

![B1 RR index19](../assets/kaggle_b0_b1_k4_controlled/B1/RR_scene_19.png)

</details>

<a id="evidence"></a>

## 7. 가중치와 검증 근거

[실행·재개](../operations/kaggle_b0_b1_k4_controlled.md) · [현재 단일 셀](../../scripts/kaggle_b0_b1_k4_controlled_one_cell.py) · [복사 텍스트](../../scripts/kaggle_b0_b1_k4_controlled_one_cell.txt) · [결과 검증 코드](../../scripts/verify_kaggle_b0_b1_k4_results.py) · [원본 artifact manifest](../assets/kaggle_b0_b1_k4_controlled/artifact_manifest.json) · [검증 기록](../assets/kaggle_b0_b1_k4_controlled/verification.json).

| 원본 가중치 | epoch | SHA-256 |
|---|---:|---|
| [B0 best.pt](../assets/kaggle_b0_b1_k4_controlled/B0/best.pt) | 97 | `1b3aeacba79071f17809a83554f4673ee4cc3a726169f8d1bc033d74c153f86a` |
| [B0 latest.pt](../assets/kaggle_b0_b1_k4_controlled/B0/latest.pt) | 100 | `a5545785f2e301a806cd6aa15e8b856d02200e5ed69de3a38fdd072e1fdf1816` |
| [B1 best.pt](../assets/kaggle_b0_b1_k4_controlled/B1/best.pt) | 94 | `366328247b2a82203e298109906787b280b29dce8bb21094bcc8174969108cfa` |
| [B1 latest.pt](../assets/kaggle_b0_b1_k4_controlled/B1/latest.pt) | 100 | `7fd93cce26c3bc45bcbf667dc1082574b55275ba0b20227c359d5da441bb643a` |

실제 원본 best/latest4개와 latest.prev2개를 포함해 제공 폴더 전체를 원본 바이트로 보관했다. prev는 completed epoch99·다음 epoch100의536개 학습 batch 완료/validation 전 상태이며, epoch100 완료 가중치로 표시하지 않는다. 각 `.pt`에는 optimizer/scaler/RNG/history와 cursor가 남아 있다. ignored `.pt`·`.npz`는 정확한 파일 목록으로 force-add한다.

원본 manifest129개 크기·SHA-256, 코드7개 hash·문법, 장면별 평균·B1−B0 차이, validation 최저best, 초기화·100개순서, checkpoint6개 ZIP CRC·metadata·NumPy tensor storage 유한값, 예측80개 shape·유한값, 패널10개의1024×296 크기를 확인했다. 신뢰되지 않는 pickle global은 제한 reader가 거부하며 산출물의 모델/worker를 실행하지 않았다. PyTorch `torch.load`/모델 forward/원본 H5로 지표를 다시 계산한 검증은 아니다.

사용자가 ZIP/SHA sidecar가 아니라 추출 폴더를 제공했으므로 원래 ZIP 전체 해시는 미검증이다. 파일별 원본 manifest를 검증하고 보관했다. 최초 보관 commit [`155312a4f5cf588b27891063f0b3e88268152683`](https://github.com/BIYONGHIYON/CtrS/commit/155312a4f5cf588b27891063f0b3e88268152683)의 **변경142개 GitHub tree blob·크기**를 로컬 Git 바이트와 대조했고, **best/latest4개를 원격에서 다시 다운로드해 SHA-256 일치**를 확인했다. 검증 기록은 verification.json의 `remote_verification`에 보존하며 후속 commit은 이 기록·문서만 추가한다. 작업 중 main에 병합된 `423c843`의 다른 연구 요약·실험표도 유지했다. 다운로드 원본·이전 실험 폴더·공용 환경은 삭제하지 않는다.

## 8. 한계와 다음 판단

**K4에서 B1의 RR 이득을 확인하고, FR 포함 종합 채택은 보류한다.** 잠정 QNR 악화는 HR 정답 오차의 직접 측정이 아니지만 RR 이득만으로 실제 해상도 품질 향상을 보장할 수 없다. QNR/D_s의 scikit-image bicubic과 MATLAB imresize 차이, Q2n/SCC parity 제한은 유지한다.

QB 단일seed42·동일 test split 반복 탐색·GPU0/1 고정배정·GPU교환 반복 없음·서로 다른 best epoch·실측 학습 비용 증가라는 제한이 있다. 동일 optimizer/초기 core/순서를 확인했지만 서로 다른 모델의 학습 궤적이나 CUDA 환경 간 bitwise 재현은 보장하지 않는다. seed43 판단은 사용자가 결과·비용을 검토한 뒤 진행하며 이번 보관 작업에서 추가 학습을 시작하지 않았다.
