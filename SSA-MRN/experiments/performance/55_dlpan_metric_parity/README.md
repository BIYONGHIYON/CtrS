# 55. Q2n 및 FR PAN 축소의 DLPan 참조 구현 정합 검증

**연구 분야:** 기존 성능 개선
**실험자:** 김경찬
**변경 내용:** Q2n 곱셈 부호·uint16 반올림·제로 평균 처리를 맞추고, 기존 FR 기본 동작을 유지하며 `pan_resize="matlab"` 옵션을 추가했습니다.

| 확인된 결과 | 상태 | 개선 여부 |
|---|---|---|
| QB RR 100항목, FR 검사용 출력 60항목 정합 확인 | 평가 완료 | 성능 비교 아님 |

[실험 목록](../../../README.md#experiments) · [연구 목록](../../../research/performance/README.md) · [수치](#results) · [그래프](#graphs) · [이미지](#images) · [재현 자료](#evidence)

## 1. 목적과 상태

**마지막 검증 확인일:** 2026-10-23. 학습·모델 추론은 수행하지 않았습니다.

동일한 배열에 대한 Python 지표와 DLPan-Toolbox MATLAB 참조 구현의 일치를 검증했습니다. 이는 모델 학습·성능 비교가 아닌 평가 구현 검증입니다. MATLAB Online R2026b에서 기준 결과를 생성했고 Python에서 비교했습니다.

## 2. 변경한 구조와 비교 조건

| 항목 | 조건 |
|---|---|
| 데이터 | PanCollection QB RR·FR 각 20장; 파일 해시는 config.json |
| 입력 형식 | Python CHW, MATLAB HWC, float64 DN |
| 전처리 | 외곽 추가 자르기·추가 클리핑·/2047 정규화 없음 |
| 공간 배율 / 블록 | 4 / 32 |
| QNR 변형 | 참조 원본 flag_orig_paper=0, p=q=alpha=beta=1 |
| PSNR | peak 2047, 밴드별 PSNR 평균; Toolbox 통합 함수에 없는 프로젝트 정의 |
| 모델 K·시드·에폭·배치·손실·best 선택 | 미적용: 모델 학습·추론하지 않았습니다. |
| 합성 입력 | 4·8밴드, .5 반올림, 상수·제로 참조 등 10조건 |
| 허용 오차 | abs(Python−MATLAB) ≤ 1e-10 + 1e-10×abs(MATLAB); 비유한 값은 통과시키지 않습니다. |

RR은 GT 대 LMS, FR은 LMS 자체 및 밴드별 gain·PAN detail을 더한 검사용 출력을 사용했습니다. FR에 GT를 만들지 않았습니다. 공용 Q2n 동작은 바뀌지만 FR은 기본 legacy를 유지하고 명시적 matlab 옵션만 새 축소를 사용합니다. 옵션은 ratio=4만 지원합니다.

<a id="results"></a>

## 3. 정량 결과

| 검사 | 최대 절대 차이 / 결과 |
|---|---|
| 합성 Q2n 10조건 | 1.44e-15 |
| 합성 RR 5지표 | 1.22e-15 |
| 실제 QB RR SAM / ERGAS | 3.02e-14 / 1.42e-14 |
| 실제 QB RR Q2n / SCC / 프로젝트 PSNR | 1.33e-15 / 2.22e-16 / 0; 100/100 통과 |
| MATLAB 호환 PAN 축소 | 20장 최대 0 DN |
| FR 검사용 출력 Dλ / Ds / QNR | 1.28e-16 / 1.67e-16 / 2.22e-16; 60/60 통과 |

기존 FR의 LMS 시험에서 Ds·QNR 최대 차이는 0.00454246이었고, 같은 축소 PAN을 공유하면 계산 오차 수준으로 줄었습니다. 이 값은 모델 성능 변화가 아닙니다. RR2·FR_WALD·관측 일치 및 개선 판정 v2.3은 이번 구현 정합 검증의 대상이 아닙니다.

## 4. 기준선 대비 차이와 해석

Q2n의 다차원 곱셈 부호, MATLAB uint16의 .5 반올림 및 제로 평균 특수 처리가 기존 구현과 달랐습니다. FR의 skimage 축소와 MATLAB imresize 차이는 검증한 20장에서 주요 불일치 원인이었습니다. 동일 축소 PAN 이후 interp23tap과 QNR 연산은 일치했습니다. 참조 구현 일치는 수학적 유일성이나 지표의 품질 타당성을 증명하지 않습니다. 과거 결과는 덮어쓰지 않았습니다.

<a id="graphs"></a>

## 5. 학습·평가 그래프

학습하지 않았으므로 학습·validation 곡선은 없습니다. 장면별 평가 차이는 results/의 CSV와 FR 기준 수치 JSON에 보존했습니다. 없는 학습 곡선을 생성하지 않았습니다.

<a id="images"></a>

## 6. 결과 이미지

모델 예측 비교가 아니므로 후보·기준 모델 예시 이미지는 생성하지 않았습니다. 입력 배열은 원본 H5와 결정적 검사용 출력 생성식으로 재생성합니다.

<a id="evidence"></a>

## 7. 재현 자료

`config.json`에 버전·원본 데이터 해시·허용 오차를 기록했습니다. `provenance/`에 검사 당시 소스와 MATLAB 실행 코드를 보존했습니다. `results/`에는 MATLAB 기준 수치, 장면별 비교 및 합성 검사 기록을 보존했습니다. 대용량 데이터·재생성 가능한 예측 배열은 포함하지 않았습니다.

저장소 루트에서 다음을 실행합니다(가상환경의 numpy, scipy, h5py, scikit-image 필요).

```bash
python SSA-MRN/experiments/performance/55_dlpan_metric_parity/run.py --data ~/Downloads
python SSA-MRN/common/verify.py --research-track performance --experiment 55
```

run.py는 원본 해시를 확인하고 현재 Python 코드로 재계산한 수치를 보존된 MATLAB 기준과 대조합니다. MATLAB을 새로 실행한 것으로 표시하지 않습니다. 기준 MATLAB 실행은 보존된 .m 코드 및 버전과 DLPan 원본 커밋을 사용합니다.

보존된 MATLAB 스크립트는 당시 실행 소스입니다. 새 MATLAB 실행에는 해당 스크립트가 요구하는 inputs.mat을 원본 데이터와 입력 생성 절차로 다시 준비해야 합니다. 현재 run.py는 저장된 MATLAB 기준 수치에 대한 Python 재검증을 제공합니다.

## 8. 개선 여부와 다음 단계

**개선 여부: 성능 비교 아님.**


검증 범위는 QB 4배·블록 32·명시한 float64 조건입니다. 실제 C2MS 예측, 다른 센서·배율·블록 크기·비정렬 크기·다른 전처리·임의 퇴화 입력을 검증한 것은 아닙니다. 기존 Q2n/FR 모델 점수와 순위 영향은 미평가입니다. 외부 논문과 비교하려면 데이터·전처리·평가 프로토콜도 일치해야 합니다. 참조의 특수 상수 입력 결과를 지표 의미의 타당성으로 해석하지 않습니다.
