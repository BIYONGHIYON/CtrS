# RGB08 · 17특징 공동 디코더 + RGB 고주파 보정

[현재 연구](../../README.md) · [후속 예비실험](pilot_followup.md)

## 1. 목적과 상태

RGB07의 세 독립 core·공동 디코더에 RGB 고주파 보정 경로만 추가해 합성 ×4 복원 성능을 확인합니다. Windows 서버 `C:\CtrS-rgb-detail`에서 2026-10-06 05:30 KST에 전체100에폭 학습을 시작했습니다. run `20261006-053053-ea63614f1e45`. 실행 source는 기존 pilot 구현이며 이번 PR은 전체 학습 설정과 전용 컨트롤러를 추가합니다.

## 2. 변경 사항과 평가 조건

RGB의 저주파는 평균4배 축소→23탭4배 확대로 만듭니다. 원본 RGB에서 이를 뺀 고주파를 CNN3→16→17로 처리하고, HSI17특징과 함께 sigmoid gate를 계산합니다. gate를 곱한 특징을 204밴드 보정량으로 변환해 기존 보정량에 더합니다. 추가 decoder는0으로 초기화합니다. 파라미터2,623,454개(RGB07 대비+7,180개).

LIB-HSI 전체 train393 / validation45 / test75, HR256/LR64,204밴드,17특징,K4, 학습quadrants·평가tiles,area 축소·23탭, 배치4/검증2, seed42, MSE+0.01(1−cos), 정합 유효 마스크, 학습·평가 LR 평균 보정입니다. GPU3060Ti8GB,torch2.7.1+cu128,AMP입니다. 기존 RGB07 체크포인트의 주요 설정27개·정합 해시·분할 수가 일치함을 확인했습니다.

첫 단계는 처음부터 LR1e-4로100에폭입니다. 완료 후 best의 모델 가중치만 불러오고 optimizer·epoch·best 기록을 초기화해 LR1e-5로10에폭 미세조정합니다. validation45로 미세조정 전후를 비교하고 최종 모델만 test75로 평가합니다. 미세조정은 첫 단계 완료 후 별도로 시작하며 자동 실행하지 않습니다.

## 3. 정량 결과

현재 학습 중으로 최종 test 결과가 없습니다. 실행 확인용 smoke의 일부 장면 수치를 성능 결과로 사용하지 않습니다. test75는 모델 선택에서 보류합니다.

## 4. 직전 연구와 수치 차이

최종 차이는 미측정입니다. 기준 RGB07은 test75에서 MSE0.0004482036, PSNR34.2831dB,SAM2.1570°입니다. 같은 장면·정합·tile·LR 보정으로 최종 평가할 예정입니다. 후보 선정 근거는 [train155의30에폭 비교](pilot_followup.md)에 기록했습니다. 단일 seed 후보 선택을 반복 검증 없이 전체 학습으로 진행한 한계를 남깁니다.

## 5. 그래프

전체 학습 곡선은 측정된 history.jsonl에서 생성합니다. 학습 종료 후 validation 및 test 비교 그래프를 추가합니다. 미측정 곡선은 생성하지 않았습니다.

## 6. 결과 이미지 예시

RGB07과 동일하게 test 장면 인덱스[3,0,43,18,63]의 tile0을 순서대로 고정합니다. 최종 결과 보고에 LR HSI·RGB·예측·정답4패널5종과204밴드HTML을 포함합니다. 성능을 본 뒤 장면을 바꾸지 않습니다.

## 7. 가중치와 검증 근거

전체학습 출력: `C:\CtrS-rgb-detail\SSA-MRN\experiments\checkpoints\remote-runs-detail\20261006-053053-ea63614f1e45`. 로그: `SSA-MRN\experiments\logs\remote-control-detail\runs\20261006-053053-ea63614f1e45\train.log`. best/latest는 실제 에폭 저장 후 생성됩니다. 데이터·pt는 Git에 포함하지 않습니다.

[실행 상태·소스 해시·시작 검증 기록](../../references/rgb08_start_20261006.json).

기준 RGB07 best99 SHA256: `5e126b17f0d95f445e8226fce401d0891593b245b737e73dc2810a3e4510d704`. 정합 SHA256: `7a66ad28bc78fe8691fddbc2d94f0a871d7c10507407f7d1f36abe1b20f004f4`. 모델 검증5개 통과, 실제 데이터2step smoke exit0, 최대 allocated4179.6MiB/reserved4920MiB를 확인했습니다.

Remote-SSH의 PowerShell에서 확인합니다.

```powershell
cd C:\CtrS-rgb-detail
$py = 'C:\CtrS\.venv\Scripts\python.exe'
$ctl = 'scripts\remote-training-detail.py'
& $py $ctl status
& $py $ctl logs --follow
```

CtrS-RGB-Detail-Control 예약 작업이 학습을 소유해 클라이언트 종료에도 유지됩니다. Windows trainer 계정 로그인·서버 전원·절전 방지를 유지합니다. Ctrl+C는 로그 보기만 종료합니다. 중단 재개는 아래 명령이며 해당100에폭 설정으로 latest에서 이어집니다.

```powershell
& $py $ctl resume-latest --from-dir 'C:\CtrS-rgb-detail\SSA-MRN\experiments\checkpoints\remote-runs-detail\20261006-053053-ea63614f1e45'
```

전체100에폭이 정상 종료된 뒤 미세조정을 시작합니다.

```powershell
& $py scripts\start-detail-finetune.py --from-dir 'C:\CtrS-rgb-detail\SSA-MRN\experiments\checkpoints\remote-runs-detail\20261006-053053-ea63614f1e45'
```

미세조정은 resume가 아니라 best의 가중치만 초기화한 새10에폭입니다. 전용 컨트롤러가 가중치를 새 run에 복사하고 SHA256를 기록합니다. 미세조정 중 중단 복구에는 해당 run의 config.json과 latest를 사용합니다.

## 8. 한계와 다음 판단

RGB07과 학습 설정은 같지만 모델 추가에 따라 초기화·난수 소비가 달라집니다. 단일 seed의 최종 차이로 안정적인 개선을 확정하지 않습니다. 학습 종료 후 미세조정 전후 검증으로 후보를 결정하고 동일 test75 평가·보고서·이미지를 생성합니다. 합성 area 조건에서의 결과이며 실제 센서 HR-HSI 정답 성능이 아닙니다.
