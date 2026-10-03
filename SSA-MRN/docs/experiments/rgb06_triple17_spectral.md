# RGB 06 · RGB별 17특징 23탭 + 분광 손실

[현재 연구](../../README.md) · [이전 실험](../previous_experiments.md)

## 1. 목적과 상태

실행 준비 완료, 학습 미실행. 서버 수치 검증 통과: 2,601,086 parameters, 17특징 x4 출력, 마스크·0벡터·빈 마스크에서 유한 gradient. 예약 작업 `CtrS-Triple17-Spectral-Control` 등록 및 controller_online=true 확인. PSNR을 유지하면서 SAM을 줄일 수 있는지 검증합니다. 논문의 분광 각도 재질 분류 아이디어를 복원 손실로 옮기는 제안이며 원 논문의 복원 손실을 그대로 재현한 것은 아닙니다.

## 2. 변경 사항과 평가 조건

204→17 grouped 1×1 학습 압축(인접 12밴드/특징), RGB별 독립 경로, K=4, 204밴드 출력. 내부 평균 축소·23탭 확대를 복구합니다. 입력 area ×4 / 23탭 baseline, HR256/LR64, quadrants 학습·tiles 평가, 393/45/75 장면, seed42, 정합 manifest·마스크는 유지합니다.

손실은 `masked MSE + spectral_weight × mean(1−cos(pred,GT))`. float32로 계산하고 GT와 예측 모두 L2 norm >1e−6인 정합 유효 픽셀만 분광 항에 포함합니다. 빈 마스크는 0으로 처리합니다. MSE에는 어두운 유효 픽셀도 포함합니다. 로그에는 train_mse/train_spectral/train_loss를 분리합니다. best는 기존과 동일하게 validation MSE로 선택합니다.

λ=0.01은 최적값이 아닌 초기 후보입니다. 0/0.001/0.01/0.1을 별도 run으로 검증하고 validation PSNR·SAM으로 선택해야 합니다. test로 λ를 고르지 않습니다. 이번에는 학습을 실행하지 않습니다.

## 3. 정량 결과

미실행. 최종 test 성능 없음.

## 4. 직전 연구와 수치 차이

직전 Bilinear: PSNR 33.7897 dB, SAM 2.2190°. 복구 기준인 내부23탭/12특징: PSNR 34.0530 dB, SAM 2.2057°. 새 결과는 대기입니다. 압축·손실 및 직전 대비 resampling까지 바뀌므로 단일 요소의 효과를 주장하지 않습니다.

## 5. 그래프

완료 후 학습/검증 및 비교 그래프를 추가합니다. 미실행 수치는 만들지 않습니다.

## 6. 결과 이미지 예시

완료 후 기존 사전 고정 5장면 tile0의 LR HSI·RGB·예측·GT 4패널 및 204밴드 HTML을 생성합니다.

## 7. 가중치와 검증 근거

기존 12특징 가중치는 새 구조에 이어서 로드하지 않습니다. 새 학습이 필요합니다. 설정: `configs/lib_rgb_hsi_triple17_k4_spectral_tiles.json`. 서버 checkout: `C:\CtrS-triple17-spectral`.

원격 PowerShell에서 다음 명령으로 제어합니다. controller는 Windows 예약 작업에서 실행되며 SSH 종료와 분리됩니다. 서버 trainer 계정은 로그인 상태를 유지해야 하며 잠금은 가능합니다. 서버 절전·종료 시 연산은 중단됩니다.

```powershell
$py = 'C:\CtrS\.venv\Scripts\python.exe'
$ctl = 'C:\CtrS-triple17-spectral\scripts\remote-training.py'
& $py $ctl status
# 아래는 사용자가 학습을 시작할 때만 실행
& $py $ctl start
& $py $ctl logs --follow
# 중단된 이 실험을 이어갈 때: 해당 run 폴더 지정
& $py $ctl resume-latest --from-dir '<17특징 실험 run 폴더>'
```

설치/재로그인 후 controller 준비만 할 경우:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File C:\CtrS-triple17-spectral\scripts\setup-remote-control.ps1
```

## 8. 한계와 다음 판단

합성 축소 성능과 실제 센서 실험을 구분합니다. 독립 장면 수와 타일 수를 구분하며 같은 test protocol에서 결과를 비교합니다. λ=0의 17특징 대조 실험이 있어야 압축 변경과 분광 손실의 효과를 분리할 수 있습니다.
