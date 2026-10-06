# SSA-MRN 연구 및 결과 보관 규칙

## 문서 구조

- README에는 최신 완료 연구와 현재 학습만 표시한다.
- 문서 입구는 `docs/research.md`(연구 설명), `docs/reproduction.md`(재현), `docs/previous_experiments.md`(실험 이력)로 둔다.
- 보고서는 `docs/experiments/reproduction/`, `rgb_hsi/`, `pilot/`로 분류한다. 이미지와 숫자 근거는 기존 실험 ID별 경로를 유지한다.
- 각 보고서 첫머리에 비교 기준·바꾼 점·관측 차이·판단 표를 넣는다. README에는 대표 이미지와 요약만 두고 5장면·곡선은 상세 보고서에 둔다.
- 공개 204밴드 뷰어에는 선택 기준을 밝힌 대표 연구 하나만 배포한다. 모든 지표에서 최고가 아니면 차이를 명시한다.
- 각 실험은 `docs/experiments/template.md`의 8개 절을 모두 사용한다.
- 첫 연구 외에는 직전 연구와 PSNR/SAM/MSE 수치 차이를 적는다. 조건이 다르면 직접 비교 불가 사유도 적는다.
- 서로 다른 데이터·장치·평가 타일의 수치 차이로 구조 개선을 확정하지 않는다.

## 새 결과의 완료 조건

- 전체 독립 test 장면 평가와 장면별 숫자 JSON, 코드 commit, weight epoch/해시를 남긴다.
- 학습 MSE/검증 PSNR·SAM 그래프 및 test baseline 비교 그래프를 반드시 만든다.
- 사전 고정한 서로 다른 test 장면 5개를 LR HSI · RGB 입력 · 예측 · 정답의 4패널로 저장한다. baseline은 수치 비교에 남기되 이미지 패널에는 넣지 않는다.
- selection에 seed, scene ID, tile, 순서를 기록한다. 결과 수치를 본 뒤 예시를 바꾸지 않는다.
- `scripts/export_results.py`와 `report_manifest.json`으로 산출물 완전성을 확인한다.
- 없는 과거 곡선/예측/epoch는 만들어내지 않는다. 과거 결과의 예시 수 부족은 보고서에 명시한다.

## 정리

- 원본 데이터 및 공식 submodule을 수정/삭제하지 않는다. 대용량 데이터를 Git에 넣지 않는다.
- 실제 학습된 best/latest `.pt`, checksum, 숫자 결과, compact epoch 기록, 정합 manifest, 결과 그래프·예시는 보존한다.
- 완료된 실험의 학습 JSON, 원시 실행 로그, smoke/random 가중치·산출물, 코드 캐시는 삭제할 수 있다. checkpoint 안의 config는 기존 가중치 평가에 필요하므로 수정하지 않는다.
- 현재/재개 예정인 학습 설정, snapshot, controller runtime, 로그, 가중치는 보존한다. 서버에서 active 상태를 확인한 뒤 완료 run만 정리한다.
- 현재 학습 중인 checkout을 pull하거나 실행 코드를 덮어쓰지 않는다.
- 더 이상 쓰지 않는 benchmark/retrain entrypoint는 제거하되 기존 모델 inference와 지표 구현은 유지한다.
- 관련 없는 checkout·환경·팀원의 파일을 정리하지 않는다.
- 커밋 메시지는 간단한 한국어. 승인된 실험 브랜치에만 push한다. main은 사용자만 변경한다.
