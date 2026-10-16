# 32번 실행·재개·결과 보관

## 처음 실행

1. 새 Kaggle Notebook을 만들고 GPU T4×2를 선택합니다.
2. CtrS PAN Sharpening Dataset을 Input에 연결합니다. 파일 경로 기본값은 코드에 포함되어 있습니다.
3. run.py 전체를 하나의 코드 셀에 붙여 넣습니다. 외부 다운로드와 이전 가중치가 필요 없어 Internet OFF로 실행할 수 있습니다.
4. 기본값 MODE='train', SEED=42를 유지합니다. 전체 GPU 사전 검사 후 N0·N1 각각100에포크를 학습하고 RR·FR 평가·ZIP 생성을 이어서 수행합니다.
5. Save Version → Save & Run All로 실행합니다. 이 동작은 편집 세션과 별도의 세션에서 실행됩니다. 편집 창의 세모 실행과 별도 버전 실행을 동시에 시작하지 않습니다.
6. 두 군의 preflight_pass를 확인합니다. train 로그의 첫 완전한 에포크 이후 실제 속도 기반 ETA가 나옵니다. 학습 ETA에는 현재 대기 중 validation/checkpoint 일부가 누락될 수 있으며 평가·압축 미측정 구간은 전체 종료 예상에 포함되지 않습니다.

학습 전 검사만 원하면 MODE='check'와 별도 RUN_ID를 사용합니다. 체크 실행 폴더에는 본 학습용 latest가 없으므로 체크 폴더를 학습 재개로 사용하지 않습니다. 정상 학습도 같은 검사를 자동 수행하므로 바로 train으로 시작해도 됩니다.

## 로그 의미

- N0: 원본 SSA-MRN입니다.
- N1: MS 기준 가역 밝기·대비 정규화입니다.
- loading_gpu_cache: 각 worker가 train·validation을 자신의 GPU에 적재합니다.
- preflight_pass: 해당 Kaggle 실행에서 유한 forward/backward, N0 원본 대응, 짧은 AMP 중단재개 검사 등을 통과했습니다.
- epoch0_validation: 무작위 초기 모델을 평가합니다. 두 군 초기 출력은 다를 수 있습니다.
- train / epoch_complete: 실제 손실, 업데이트 수, 메모리, 시간과 검증 결과입니다.
- COMPLETE LOCAL OUTPUT: 로컬 산출물·전체 평가·해시·ZIP 확인을 완료했습니다. 원격 보관 완료를 뜻하지 않습니다.

## Validation만 먼저 확인하기

추가seed 여부를 test 전에 결정하려면 시작 전 EVALUATE_AFTER_TRAINING=False로 설정합니다. training.zip을 보존하고 validation 이력을 확인합니다. 같은 설정의 동일 학습을 RESUME_FROM으로 재개하면서 이 값만 True로 바꾸면100에포크를 다시 학습하지 않고 기존best의 평가를 수행합니다. 나머지 학습 설정과 코드 변경은 fingerprint 불일치로 거부합니다.

## 중단·재개

latest.pt는 약60초마다 유효 배치 경계에서 원자적으로 저장하며 매 에포크 검증 후에도 저장합니다. best.pt는 epoch0 포함 validation MSE 최솟값의 모델입니다. latest 안에 best_model도 포함되어 best 파일이 유실되어도 복원할 수 있습니다. 다만 정상 manifest가 있는데 파일이 유실된 경우에는 먼저 무결성 검사에서 중단합니다. 별도 복구가 필요하다는 신호이며 검사를 임의로 제거하지 않습니다.

같은 세션에서 중단했다면:

    RESUME_FROM = '/kaggle/working/32_ms_anchored_reversible_normalization_s42_v1'

RUN_ID와 다른 설정은 그대로 유지합니다. 저장한 에포크 순서와 다음 배치부터 이어갑니다. 마지막 저장 이후 실행한 배치는 다시 실행될 수 있습니다.

새 세션에서는 이전 Saved Output을 Notebook Input으로 연결합니다. ZIP만 연결했다면 파일을 먼저 별도 위치에 풀어야 합니다. 이전 실행 폴더의 config.json·provenance/source·두latest가 모두 있어야 합니다. RESUME_FROM은 Input 아래의 실제 실행 폴더로 지정합니다. 새 /kaggle/working 목적지와 이름이 같아도 해당 목적지가 아직 없어야 복사가 가능합니다. 자동 경로 추측이나 무관한 가중치 선택은 하지 않습니다.

코드·설정·데이터가 달라졌다면 동일 실험으로 재개하지 말고 새RUN_ID에서 처음부터 시작합니다. Saved Output 폴더를 다른 새 코드로 덮어쓰지 않습니다. seed43·44는 SEED와 RUN_ID를 함께 바꾸고 RESUME_FROM=''로 새 학습합니다.

## 저장과 안전한 종료

- 설정에 Session Persistence가 제공되면 Files only 또는 해당 파일 보존 옵션을 선택할 수 있습니다. 이 기능은 best effort이며 별도 보관을 대체하지 않습니다. 계정·UI에 따른 현재 옵션을 확인해야 합니다.
- Draft 코드 저장과 checkpoint 보관은 다릅니다. Save & Run All이 새 세션에서 시작되므로 이전 학습 재개에는 Saved Output의 Input 연결과 RESUME_FROM 지정이 필요합니다.
- 버전이 끝나면 Notebook Viewer의 Output에서 결과ZIP과 .zip.sha256, 실행 폴더의 checkpoints/N0_QB_s42 및 N1_QB_s42/best.pt·latest.pt, manifest.json이 실제로 보이는지 확인합니다.
- ZIP과 해시를 다운로드해 별도 위치에 보관합니다. ZIP CRC와 manifest의 SHA256을 확인하고 실제 best/latest를 열어 복구 가능 여부를 확인한 뒤 보관 완료로 봅니다.
- 본 코드의 ZIP 해시 검사는 Kaggle 로컬 생성물만 확인합니다. 원격Saved Output이나 다운로드 파일은 자동으로 확인하지 않습니다.
- 단일 배치·validation·저장 시간이 길면 체크포인트 간격이60초를 넘을 수 있습니다. SIGTERM/SIGINT는 안전한 배치 경계 저장을 시도하지만 강제 종료·런타임 삭제·SIGKILL에는 저장이 보장되지 않습니다. controller는 종료 요청 후 최대60초 기다린 뒤 종료할 수 있습니다.
- 원격Saved Output·Dataset·다운로드 복사본 어디에도 없는 삭제된 결과는 코드로 복구할 수 없습니다.

공식 안내 확인: https://www.kaggle.com/docs/notebooks
Persistence 안내: https://www.kaggle.com/product-feedback/355440
현재 로그인 계정의 실제 설정 화면은 이번 작업에서 확인하지 않았습니다.

## 검증 범위

전달 파일은 CPU 합성 데이터의 실제 모델 forward/backward, N0 동등성, 가역 변환, 상수/낮은 대비 입력, 샘플 통계 독립성, Adam/BatchNorm/배치 순서 체크포인트 복구를 통과했습니다. 이는 실제 데이터 성능 결과가 아닙니다. 실제H5·T4 AMP·GPU상주·장시간 중단재개·RR/FR 수치는 아직 미검증입니다. Kaggle에서 사전 검사 실패 시 임의로 fallback하거나 에러를 무시하지 않습니다.
