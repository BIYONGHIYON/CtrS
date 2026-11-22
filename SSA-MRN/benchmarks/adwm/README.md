# ADWM LAGConv

공식 ADWM `aa185a6`의 LAGConv 모델을 PanCollection에서 평가한다. 평가자는 김경찬이다.

| 평가 | 상태 | 결과 |
|---|---|---|
| WV3 저자 가중치 RR·FR | 예측 생성 근거 확인 대기 | 새 추론 후 확정 |
| WV3 → WV2 전이 RR | MATLAB 완료 · 20장 | PSNR 29.60093 dB · SAM 5.80065° · ERGAS 4.58945 · SCC 0.91267 · Q2n 0.82165 |
| WV3 → WV2 전이 FR | Python 참고 · 20장 | QNR 0.871322 · Dλ 0.066332 · Ds 0.073021 |
| QB·GF2 직접 학습 | 학습 준비 | 시드 42 · 배치 64 · QB 200에폭 · GF2 500에폭 |

WV2 전이는 WV3 저자 가중치와 `identity` 설정으로 예측했다. RR은 [MATLAB 실행 #19](https://github.com/BIYONGHIYON/CtrS/actions/runs/45058335188)에서 채점했고, 예측은 [ADWM author Release](https://github.com/BIYONGHIYON/CtrS/releases/tag/predictions-benchmarks-adwm-author)에 있다. FR은 CtrS Python 지표를 사용했다.

## 직접 학습

논문 부록 D에 따라 QB의 CACW 은닉 비율은 0.6, GF2는 0.8이다. Adam과 L1, 초기 학습률 0.002를 사용하고 공식 StepLR 설정을 적용한다. 매 에폭 전체 validation에서 L1이 가장 작은 가중치를 선택한다.

학교 서버 RTX A6000 한 장에서 QB·GF2가 동시에 학습 3스텝과 검증 1배치를 통과했다. 배치 64에서 PyTorch 피크 예약량은 QB 20,916 MiB, GF2 20,918 MiB였다.

학교 서버에서 팀에 사용을 알리고 공식 소스 하위 모듈을 준비한 뒤, 센서별로 서로 다른 출력 폴더를 지정한다.

```bash
git submodule update --init SSA-MRN/benchmarks/adwm/upstream
python -u SSA-MRN/benchmarks/adwm/train.py --sensor qb --data "$PAN_DATA" --out "$RUNS/qb"
python -u SSA-MRN/benchmarks/adwm/train.py --sensor gf2 --data "$PAN_DATA" --out "$RUNS/gf2"
```

`PAN_DATA`는 PanCollection 데이터 루트이고 `RUNS`는 결과를 저장할 경로다. 병렬 실행은 센서별로 별도 터미널에서 시작한다.
