# 40번 실험 실행 안내

직접 실행할 파일은 run.py 또는 같은 내용의 40_ms_center_scale_ablation.txt입니다. 파일 전체를 Kaggle 코드 셀 하나에 붙여 넣습니다. 기존 실험 결과나 가중치는 필요하지 않습니다.

## 1. 처음 실행

1. 새 Kaggle Notebook에서 GPU T4 두 장을 선택합니다.
2. CtrS PAN Sharpening Dataset을 Input에 연결합니다.
3. run.py 전체를 하나의 셀에 붙여 넣습니다.
4. 기본값 MODE='train', SEED=42, RESUME_FROM='', EVALUATE_AFTER_TRAINING=False를 유지합니다.
5. Save Version → Save & Run All로 실행합니다. 편집 세션과 저장 버전 실행을 동시에 시작하지 않습니다.

외부 소스 다운로드가 없으므로 Internet OFF로 실행하도록 만들었습니다. 현재 이 답변에서는 로그인한 Kaggle UI나 최신 공식 문서를 직접 확인하지 않았습니다. 화면의 실제 GPU·Input·저장 옵션을 확인합니다.

기본 결과 폴더는 /kaggle/working/40_ms_center_scale_ablation_s42_v1입니다. 같은 이름의 폴더가 있으면 덮어쓰지 않고 중단합니다.

## 2. 자동 진행 순서

- 첫 번째 묶음: GPU0 A0(원본), GPU1 A3(전체 정규화)
- 두 번째 묶음: GPU0 A1(평균 제거), GPU1 A2(대비 조절)
- 각 군: 실제 GPU 사전 검사 → 초기 validation → 100 epoch → best/latest 저장
- 네 군 완료: 초기 상태·학습 순서·업데이트·best 검증 → 학습 곡선 → validation_decision.json → training ZIP과 해시

각 군 preflight_pass는 유한 forward/backward, GPU 데이터 상주, 짧은 실제 AMP 중단·재개 검사를 의미합니다. 두 업데이트 검사이므로 장시간 전체 세션 강제 종료를 모두 보장하지 않습니다. 검사 실패·OOM·데이터 오류를 조용히 우회하지 않습니다.

현재 모델 ETA는 최근 실제 epoch 시간을 사용합니다. 아직 측정하지 않은 다음 묶음의 시간과 test·압축 시간을 확정해서 표시하지 않습니다. wave_start 로그의 GPU 대기열과 worker의 KST ETA를 함께 봅니다.

## 3. 학습 종료 후 확인

TRAINING COMPLETE와 validation_decision.json을 확인합니다. 이 단계는 정상적인 학습 완료이며 RR·FR 평가 전입니다.

A1 또는 A2의 best validation MSE가 A3의 1.005배 이내이면 repeat_all_four_arms_seeds43_44가 true입니다. 그러면 네 군 모두 seed43·44를 새로 학습하는 후속 계획을 유지합니다. false이면 반복 확대를 중단합니다. 추가 시드는 자동으로 실행되지 않습니다. 이 판단을 test 전에 기록합니다.

우선 training ZIP과 .zip.sha256을 다운로드하고, Saved Version Output에 파일과 네 군의 best/latest가 실제로 있는지 확인합니다. validation_decision.json을 보내면 기록된 기준과 결과를 함께 검토할 수 있습니다.

## 4. RR·FR 평가 실행

학습 폴더를 그대로 보존한 상태에서 같은 코드를 사용합니다. 다음 두 설정만 바꿉니다.

    RESUME_FROM = '/kaggle/working/40_ms_center_scale_ablation_s42_v1'
    EVALUATE_AFTER_TRAINING = True

SEED·RUN_ID·나머지 소스와 설정은 유지합니다. 네 군 모두 이미 100 epoch라면 추가 학습 없이 validation-selected best만 평가합니다. 처음부터 True로 실행하는 것은 허용하지 않습니다. 사전에 완료한 validation-only 실행이 필요합니다.

새 Kaggle 세션이라면 이전 Saved Output을 Input에 연결하고 RESUME_FROM을 그 Input 안의 정확한 실행 폴더로 지정합니다. 폴더에는 config.json·provenance/source·checkpoints·결과가 함께 있어야 합니다. ZIP만 연결했다면 전체 ZIP을 별도 경로에 먼저 풀고 내부 실행 폴더를 지정합니다. /kaggle/working 목적지는 아직 없어야 자동 복사할 수 있습니다. 경로를 추측해 다른 실험 checkpoint를 고르지 않습니다.

평가가 끝나면 _results.zip과 .zip.sha256이 생성됩니다. 각 군 RR·FR 20장 지표, 고정 RR 5장면, 원본 initial/best/latest, 학습 곡선, 비교 그래프, 진단·manifest가 포함됩니다. QNR·Q2n·SCC는 공식 수치 대응이 남은 잠정 지표입니다.

## 5. 학습 중단·재개

학습 중 중단됐다면 같은 소스·설정·RUN_ID를 유지하고 RESUME_FROM만 정확한 이전 폴더로 지정합니다. EVALUATE_AFTER_TRAINING은 False로 유지합니다.

완료한 군은 건너뛰고, 시작한 군은 latest의 현재 epoch·다음 batch부터 재개하며, 대기 중이던 군은 동일 시드로 처음부터 시작합니다. 시작한 군의 initial.pt는 있는데 latest가 없다면 임의로 새 학습을 시작하지 않습니다. 기존 best/latest나 manifest가 실제로 유실된 경우 복구가 필요하다는 오류가 발생할 수 있습니다.

latest에는 best_model 복사본도 보존하지만, manifest 불일치까지 무시하고 자동 복구하지는 않습니다. 원본을 유지한 상태에서 별도 복구가 필요합니다.

저장은 안전한 유효 batch 경계에서 약 60초마다 수행합니다. 긴 batch·validation·저장 구간에는 60초를 넘을 수 있습니다. 마지막 저장 이후의 batch는 다시 실행될 수 있습니다. 강제 런타임 삭제·SIGKILL에는 종료 handler가 작동하지 않을 수 있습니다. 컨트롤러는 일반 중단 시 worker에 종료를 요청하고 최대 약45초 기다린 뒤 남은 worker를 종료합니다.

MODE='check'로 검사만 실행했다면 학습에는 새 RUN_ID를 사용합니다. 체크 폴더를 학습 재개 대상으로 지정하지 않습니다.

## 6. 추가 시드

validation 기준을 통과한 경우에만 새 실행으로 진행합니다.

    SEED = 43
    RUN_ID = '40_ms_center_scale_ablation_s43_v1'
    RESUME_FROM = ''
    EVALUATE_AFTER_TRAINING = False

44도 SEED와 RUN_ID의 숫자를 함께 바꿉니다. 각 시드에서 네 군은 같은 초기 상태·배치 순서를 사용하지만 서로 다른 시드의 가중치는 공유하지 않습니다. 실행 순서와 100 epoch·batch·학습률을 그대로 유지합니다.

## 7. 파일 보관

Draft 코드 저장은 checkpoint 보관과 다릅니다. Save & Run All이 새 세션에서 시작하면 이전 파일이 자동으로 연결된다고 가정하지 말고 Saved Output을 Input에 연결해 재개합니다.

Session Persistence의 파일 보존 옵션이 화면에 제공되더라도 유일한 보관 수단으로 삼지 않습니다. Saved Version Output에서 ZIP·해시·best/latest·manifest의 존재를 확인하고 다운로드본의 ZIP CRC와 manifest SHA256을 확인합니다. 로컬 ZIP 생성 로그는 원격 저장 완료를 뜻하지 않습니다.

Saved Output·Dataset·다운로드본 어디에도 없는 삭제된 결과는 코드로 복구할 수 없습니다. 기존 결과 폴더 정리·삭제는 이 실행 코드가 수행하지 않습니다.

## 8. 검증 범위

검증 파일은 validation/local_verification.json입니다. 현재 환경에서 구문, 공통 소스 보존, 정규화 수식의 성질, 컨트롤러 두 묶음 배정·완료군 건너뛰기·worker 실패 전파, fingerprint·ZIP 무결성 경로를 확인합니다.

PyTorch·GPU·원본 H5가 없어 실제 네트워크 CPU/CUDA forward/backward, 학습 성능, CUDA 메모리, 실제 AMP 재개는 여기서 실행하지 못했습니다. GPU 사전 검사는 Kaggle 실행 시 수행됩니다. 과거 실행이 통과했다는 이유로 새 네 군도 통과한 것으로 표시하지 않습니다.
