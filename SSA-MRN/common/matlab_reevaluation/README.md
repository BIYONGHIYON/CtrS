# 실험 53·54의 실제 MATLAB 재평가

중간발표의 논문 비교에 사용할 기존 결과를 재학습 없이 다시 계산합니다. 정규화 후보·새 학습 실험이 아닙니다. 기존 checkpoint와 원래 보고서·지표 JSON은 수정하지 않습니다.

## 범위

- 실험 53: QB C0·C2·F1·C2MS 각 seed42·43·44, F1MS seed42 — 13개 기존 모델
- 실험 54: GF2·WV3·WV2 전이의 C0·C2MS 각 seed42·43·44 — 18개 평가
- 각 모델마다 원래 RR20·FR20 전체 장면을 재추론하고 RR·RR2·FR_WALD·FR을 실제 MATLAB으로 채점합니다. WV2는 WV3 학습 가중치를 그대로 사용합니다.

## GitHub Actions 실행

`.github/workflows/matlab-reevaluate-53-54.yml`은 전용 브랜치 push에서 전체 평가를 시작하고, main에 반영된 뒤에는 Actions → **MATLAB reevaluate experiments 53 and 54** → Run workflow로 재실행합니다. 실험 53만/54만/둘 다를 선택할 수 있습니다. main 병합은 사용자가 진행합니다.

31개 case를 독립 job으로 나누고 최대6개를 동시에 실행합니다. CPU FP32 추론만 실행하며 train.py를 가져오거나 optimizer·학습을 실행하지 않습니다. 추론 시 저장된 validation best의 manifest SHA-256을 확인하고 strict load합니다. 모델을 새 test 결과로 고르지 않습니다.

[공식 MathWorks setup-matlab](https://github.com/matlab-actions/setup-matlab)과 [run-command](https://github.com/matlab-actions/run-command)를 통해 MATLAB R2025b와 Image Processing Toolbox를 실행합니다. 공개 프로젝트의 일반 제품 자동 라이선스를 사용하며, Octave·Python 수식을 MATLAB 결과로 대신 표시하지 않습니다.

## 실제 계산

1. 기존 보고서와 같은 데이터 파일을 SHA-256으로 확인합니다. PanCollection 공개 과거 버전을 고정하며 WV3 RR은 그 파일의 원래 해시가 일치하는 별도 버전을 사용합니다. 최신 파일을 임의로 대입하지 않습니다.
2. 저장된 모델의 동일 예측 배열을 `.mat`로 내보냅니다. Python 기존 지표도 그 배열에서 다시 계산합니다.
3. 고정 DLPan commit의 원본 `indexes_evaluation.m`과 `indexes_evaluation_FS.m`를 실제 MATLAB에서 호출합니다. Q·SAM·ERGAS·SCC·Q2n 및 Dλ·Ds·QNR은 이 함수의 반환값입니다.
4. PSNR은 DLPan wrapper의 반환 지표가 아닙니다. MATLAB에서 global MSE 기반과 밴드별 dB 평균을 각각 계산해 이름을 나눠 보존합니다. GF2는 peak1023, QB·WV3·WV2는 peak2047을 사용합니다. 기존 GF2 Python peak2047 값과의 차이를 숨기지 않습니다.
5. 기존 Python 재추론 − 과거 보고값(실행환경 차이)과 MATLAB − 같은 예측의 Python(지표 차이)을 별도로 기록합니다. FR_OBS는 기존 Python 관측 연산의 값으로 보존하며 MATLAB 지표라고 표시하지 않습니다.

기본 조건은 기존 결과처럼 전체 장면·crop 없음·출력 clipping 없음입니다. 수동 실행에서 cut_bounds·dim_cut·clip_output을 바꿀 수 있습니다. dim_cut은 DLPan의 정확한 MATLAB 인덱스이며 `dim_cut:end-dim_cut`입니다. th_values의 상한은 원본 코드대로 `2^bits`입니다. 설정을 바꾸면 기존 수치와의 차이는 순수한 구현 차이가 아니므로 그 사실을 report에 표시합니다. QNR은 원본 flag_orig_paper=0의 Toolbox 1.0 방식입니다. HQNR·원 논문 QNR과 혼합하지 않습니다.

## 발표 자료 받기

모든 case가 끝나면 Actions의 **presentation-matlab-experiments-53-54** artifact를 받습니다.

- `presentation_table.md`: 발표에 사용할 모델·센서별 실제 MATLAB 수치 표
- `matlab_per_seed.csv`: 시드별 전체20장 평균
- `matlab_seed_mean_sd.csv`: 각 시드 평균을 다시 집계한 평균±표본 SD
- `all_cases.json`: 기존 수치·Python 재추론·MATLAB 값과 차이, 실행 조건
- `completion.json`: 요청한 모든 case의 완료 여부

case별 `matlab-e53_...` / `matlab-e54_...` artifact에는 장면별 CSV·MATLAB 원시 숫자 JSON·코드/데이터/weight/예측 hash manifest·MATLAB release·제품 버전·실제로 호출된 함수 위치가 포함됩니다. 대용량 원본 데이터와 재생성 가능한 예측 배열은 artifact에 중복 업로드하지 않고 SHA-256을 남깁니다. artifact 보존 기간은30일이므로 발표 자료와 증거 파일을 다운로드해 보관합니다. 하나라도 빠지면 최종 발표용 완료 artifact를 만들지 않습니다.

다른 논문의 숫자는 여기서 임의로 가져오지 않습니다. 해당 논문의 test 장면·crop·clipping·PSNR 정의·QNR 변형이 같은지 확인한 뒤 비교표에 합칩니다. 실제 MATLAB을 썼다는 사실만으로 모든 논문과 평가 설정이 같아지는 것은 아닙니다.

## 확인 범위

`test_suite.py`는 기존31개 checkpoint 해시·strict load, seed42의 각 모델/센서 합성 forward, 보고서 집계·누락 차단을 CPU에서 검사합니다. 합성 보고서 검사는 MATLAB 실행이나 실제 성능 수치가 아닙니다. 실제 MATLAB과 전체 test 재평가는 Actions의 성공한 job과 artifact로 확인합니다.

```powershell
python SSA-MRN/common/matlab_reevaluation/test_suite.py
```
