# 실험 ID · 연구명

[현재 연구](../../README.md) · [실험 이력](../previous_experiments.md)

## 1. 목적과 상태

가설, 준비/실행/종료/test 평가/보고 상태, 실행 날짜·시간대, run ID와 코드 commit. 확인되지 않은 것은 미확보로 표시합니다.

## 2. 변경 사항과 조건

직전 실행 대비 변경, 모델·파라미터·입출력 밴드, 데이터 root 식별자·season·ROI·train/val/test 패치 수, 정합·정규화·crop·증강·TTA, seed·고정 subset, 장치, 정밀도·loss precision, optimizer·lr·scheduler·batch/accumulation, 초기화와 재개, best epoch를 기록합니다.

## 3. 정량 결과

test/validation을 분리합니다. RMSE·MAE·PSNR·SAM·SSIM·LPIPS 전체 평균과 패치별 JSON/CSV 근거를 연결합니다. 구름 비율별·밴드별 결과는 실제 측정한 경우만 추가합니다. 패치 수를 독립 장면 수로 쓰지 않습니다.

## 4. 이전 실행과 차이

같은 분할·패치 목록·metric 조건의 이전/현재/차이(현재−이전)를 적습니다. 초기 실험 또는 비교 불가 항목은 N/A와 이유. 여러 설정이 바뀌면 원인 분리 한계를 명시합니다.

## 5. 학습·검증 곡선

실제 train loss/MAE, validation PSNR/SAM/SSIM/loss, lr 그래프. best·발산·중단/재개를 표시합니다. 없으면 미확보로 둡니다.

## 6. 같은 입력의 이미지 비교

SAR · cloudy · prediction · target; 전후면 previous prediction 추가. 사전 고정 5사례의 seed·ROI/patch ID·순서, RGB 밴드·공통 밝기/대비, SAR 표시를 기록합니다. 독립 ROI 부족 및 과거 선정 근거 누락을 명시합니다. 새롭게 찾은 실패 사례는 별도 구분합니다.

## 7. 가중치와 증거 위치

code commit, epoch·checkpoint SHA-256, 서버 보관 경로, summary/sample metrics, compact curve, selection과 실제 이미지 링크. 서버 경로를 Git 링크처럼 표시하지 않습니다. 대용량 파일을 새로 Git에 넣지 않습니다.

## 8. 한계와 다음 판단

실제 관측·해석·가설을 구분합니다. 잔여 구름·분광 오차·정합·ROI 대표성·계절 일반화·동시 변경·FP16 probe 한계를 적고 다음 실험의 판단 기준을 정합니다.
