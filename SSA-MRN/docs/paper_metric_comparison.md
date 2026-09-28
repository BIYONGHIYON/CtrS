# SSA-MRN 논문 수치와 현재 복구 모델 비교

2026-09-28 기준. PanCollection의 센서별 ReducedData(RR)·FullData(FR) H5를 각 20장씩, 총 160장 평가했다. QB·GF2·WV3는 각각 100-epoch `latest.pt`를, WV2는 논문처럼 WV3 가중치를 사용했다. 표의 `현재`는 **이미지별 지표를 계산한 뒤 20장 산술평균**한 값이다. 논문 열은 Tables I–IV의 *Ours* 행이다.

| 센서 | SAM↓ 논문/현재 | ERGAS↓ 논문/현재 | PSNR↑ 논문/현재 | SCC↑ 논문/현재 | Q4·Q8↑ 논문/현재 |
|---|---:|---:|---:|---:|---:|
| QB | 4.8478 / 4.9557 | 4.0726 / 4.1555 | 37.6645 / 36.8688 | .9702 / .9765 | .9243 / .9214 |
| GF2 | .9434 / .9668 | .8755 / .9125 | 46.9734 / 40.2439 | .9898 / .9816 | .9688 / .9661 |
| WV3 | 3.4873 / 3.5277 | 2.5866 / 2.5822 | 37.5691 / 35.9690 | .9735 / .9779 | .8930 / .8753 |
| WV2 | 5.8845 / 5.9265 | 4.7254 / 4.8002 | 29.4148 / 28.1845 | .9217 / .8959 | .8215 / .8190 |

| 센서 | Dλ↓ 논문/현재 | Ds↓ 논문/현재 | QNR↑ 논문/현재 |
|---|---:|---:|---:|
| QB | .0341 / .0479 | .0360 / .0396 | .9311 / .9147 |
| GF2 | .0395 / .0350 | .0487 / .0565 | .9137 / .9106 |
| WV3 | .0329 / .0221 | .0617 / .0382 | .9077 / .9408 |
| WV2 | .0657 / .0511 | .0549 / .0382 | .8831 / .9126 |

**해석 주의:** 지표 이름은 논문과 같지만, 논문에서 PSNR의 peak·밴드/이미지 평균, 경계 제외, FR용 QNR 구현 세부값을 명시하지 않아 아직 *동일 평가 프로토콜*이라고 확정할 수 없다. 위 PSNR은 센서별 peak(QB/WV3/WV2=2047, GF2=1023)의 **전 밴드 MSE**로 계산했다. 참고로 모든 센서에 peak 2047을 쓰고 밴드별 PSNR을 평균하면 GF2는 46.3520 dB로 바뀐다. 이는 차이의 원인을 찾는 진단값이지 논문 설정이라고 확인된 값은 아니다. SCC 경계 처리와 Q4/Q8의 MATLAB 구현 일치성도 미검증이다. FR의 PAN 축소는 MATLAB `imresize`가 아니라 scikit-image를 사용하므로 Dλ·Ds·QNR 모두 잠정치다. 따라서 일부 현재 값이 논문보다 좋아 보여도 성능 우위나 논문 재현 성공으로 해석하지 않는다.

또한 공개 `network.py`의 SSA 내부 차원 K=4·어텐션 연산은 논문 설명(K=6 등)과 다르다. 지금 가중치는 **공개 코드를 복구하여 학습한 모델**의 결과다. 향후 MATLAB 평가 툴박스와 동일 입력을 대조하고, 논문의 누락 설정을 확인한 뒤 수치를 다시 비교해야 한다.

## 재실행

테스트 H5는 Git에 없으며 `data/raw/{QuickBird,Gaofen2,WorldView3,WorldView2}/` 아래에 원본 파일명으로 둔다. 자세한 파일 배치는 `scripts/evaluate_paper.py`의 `SENSORS`와 파일명 생성 규칙을 따른다.

```bash
cd SSA-MRN
python scripts/evaluate_paper.py --sensor all --protocol both
python -m unittest discover -s tests -v
```

센서별·프로토콜별 상세 JSON(샘플별 지표와 평균)은 `experiments/results/paper_comparison/`에 생성된다. 대용량 입력 H5와 JSON은 Git 추적 대상이 아니다.

출처: [SSA-MRN 논문](https://www.researchgate.net/publication/389343736_Spectral-Spatial_Attention-guided_Multi-Resolution_Network_for_Pansharpening), [PanCollection](https://github.com/liangjiandeng/PanCollection), [DLPan-Toolbox 평가 코드](https://github.com/liangjiandeng/DLPan-Toolbox).
