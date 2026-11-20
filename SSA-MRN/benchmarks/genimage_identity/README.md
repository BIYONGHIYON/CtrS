# 변환 한계 대조군 (생성형 이미지 모델 시험)

| 구분 | 설정 | 상태 |
|:---:|:---:|:---:|
| 생성형 모델 시험의 대조군 | 모델 없음 · RGB 3밴드 · 실제 MATLAB 채점 | 평가 완료 |

**무엇인가:** 생성형 이미지 모델 시험([#209](https://github.com/BIYONGHIYON/CtrS/issues/209)-[#212](https://github.com/BIYONGHIYON/CtrS/issues/212))에서 모델에 주는 1번 입력 이미지(LMS RGB, 색 기준)를 **모델 없이 그대로** 되돌린 결과입니다. 데이터셋 LMS 보간과 8bit 변환·1024 확대·면적 평균 축소만 거쳤습니다. 생성형 모델의 결과가 이 값보다 나아야 PAN의 세부를 실제로 넣었다고 볼 수 있습니다.

**결과:** RR test 20장 RGB 3밴드의 실제 MATLAB PSNR은 QB 29.216, GF2 37.953, WV3 29.225, WV2 27.695 dB입니다. 8bit 변환으로 잃는 양은 보간만 한 LMS 대비 PSNR 약 0.04 dB였습니다. 같은 배열의 Python 지표는 MATLAB과 PSNR·SAM·ERGAS·Q2n이 소수 넷째 자리까지 같았습니다.

## 결과

### RR 실제 MATLAB 채점 (RGB 3밴드)

| 센서 | RGB 밴드 | PSNR (dB) | SAM (도) | ERGAS | SCC | Q2n |
|---|---|---:|---:|---:|---:|---:|
| QB | [2,1,0] | 29.216 | 4.0318 | 10.8878 | 0.7224 | 0.5994 |
| GF2 | [2,1,0] | 37.953 | 1.0331 | 2.2394 | 0.8600 | 0.7916 |
| WV3 | [4,2,1] | 29.225 | 2.4074 | 6.6941 | 0.7417 | 0.6131 |
| WV2 | [4,2,1] | 27.695 | 3.3521 | 6.9470 | 0.7232 | 0.6393 |

- 각 센서 RR test 20장 평균입니다. PSNR은 모든 센서에서 peak 2047의 밴드별 PSNR 평균입니다(GF2도 peak 2047로 표시).
- Q2n은 3밴드에 0 밴드 하나를 덧붙인 Q4입니다(DLPan `q2n.m`과 `common/metrics.q2n`이 같은 방식). 4·8밴드 전체의 Q4·Q8과 비교하지 않습니다.
- [MATLAB 요약](results/matlab_summary.json) · [표](results/matlab_table.md) · 장면별 [QB](results/matlab_scenes_QB.json) · [GF2](results/matlab_scenes_GF2.json) · [WV3](results/matlab_scenes_WV3.json) · [WV2](results/matlab_scenes_WV2.json)

### 정답 없는 지표

| 센서 | 관측 일치 오차 (DN) 대조군 / 정답 | PAN 구조 일치도 대조군 / 정답 | 위치 어긋남 (행, 열 픽셀) 대조군 / 정답 |
|---|---|---|---|
| QB | 18.37 / 0.00 | 0.368 / 0.758 | (0.22, −0.42) / (0.21, −0.44) |
| GF2 | 8.38 / 0.00 | 0.303 / 0.775 | (0.20, −0.32) / (0.27, −0.26) |
| WV3 | 19.39 / 0.00 | 0.378 / 0.767 | (−0.18, −0.10) / (−0.17, −0.02) |
| WV2 | 19.88 / 0.00 | 0.387 / 0.861 | (−0.18, −0.21) / (−0.10, −0.12) |

- **관측 일치 오차:** 출력을 센서 MTF로 ×4 축소(위상 2)한 결과와 입력 MS의 RGB 밴드 사이 RMSE입니다(가장자리 5픽셀 제외). 네 센서 모두 정답은 0에 가까워(1e-12 DN 이하), 성능 54번의 센서 연산이 test 데이터와 정확히 맞습니다. 그래서 생성형 모델이 입력에 없는 밝기·색을 지어내면 이 값이 커집니다.
- **PAN 구조 일치도:** 출력 밝기(RGB 평균)와 PAN에서 5×5 평균을 뺀 고주파 성분의 상관입니다. 정답은 0.76-0.86, 세부가 없는 대조군은 0.30-0.39입니다.
- **위치 어긋남:** PAN에 대한 출력 밝기의 위상 상관 이동량(10배 세분)입니다. 정답에도 PAN과 MS 사이의 고유한 차이(0.5픽셀 이내)가 있으므로, 생성형 모델은 **정답의 값과 비교**해 어긋남을 판단합니다.
- 장면별 값: [QB](results/checks_QB.json) · [GF2](results/checks_GF2.json) · [WV3](results/checks_WV3.json) · [WV2](results/checks_WV2.json)

## 해석

- 이 대조군은 PAN을 전혀 쓰지 않은 결과라, 생성형 모델이 넘어야 할 최저선입니다. 같은 RGB 3밴드에서 C2MS-SP·원본 SSA-MRN과의 비교는 생성형 모델 결과와 함께 정리합니다.
- 8bit 변환 손실(약 0.04 dB)이 작아, 생성형 모델과의 차이는 입력·출력 변환이 아니라 모델이 만든 내용에서 나온다고 볼 수 있습니다.

<details>
<summary>방법</summary>

- 입력 묶음: `python SSA-MRN/common/genimage_io.py inputs` → Release [`genimage-inputs`](https://github.com/BIYONGHIYON/CtrS/releases/tag/genimage-inputs)(`genimage_inputs_<SENSOR>.zip`, 장면별 1024×1024 PNG 두 장과 `meta.json`). 1번 이미지는 LMS의 RGB 밴드를 밴드별 0.1-99.9% 범위로 8bit 변환해 bicubic 4배 확대했고, 2번 이미지는 PAN을 같은 방식으로 변환했습니다. 정답은 쓰지 않았습니다. 같은 데이터로 다시 만들면 같은 SHA-256이 나옵니다.
- 대조군: `python SSA-MRN/common/genimage_io.py identity` — 1번 이미지를 그대로 `restore`(4×4 면적 평균, 1번 이미지 범위의 역변환)에 넣어 `output_mulExm_<i>.mat`(키 `sr`, DN, 256×256×3)을 만들었습니다. Release [`predictions-benchmarks-genimage_identity-zeroshot`](https://github.com/BIYONGHIYON/CtrS/releases/tag/predictions-benchmarks-genimage_identity-zeroshot).
- 채점: `matlab-benchmark-author.yml`(model `genimage_identity`, setting `zeroshot`, bands `rgb`), MATLAB R2025b, DLPan-Toolbox `a34f884`, 자르기·클리핑 없음.
- 정답 없는 지표: `python SSA-MRN/common/genimage_io.py checks --sensor <SENSOR> --predictions <rr zip>`

</details>

<details>
<summary>재현 자료</summary>

- 장면별 되돌리기 기록(입력 묶음 해시, 출력 zip 해시): [QB](results/restore_QB.json) · [GF2](results/restore_GF2.json) · [WV3](results/restore_WV3.json) · [WV2](results/restore_WV2.json)
- 예측 zip SHA-256: QB `51716d5e…`, GF2 `0c424c88…`, WV3 `cb9930cd…`, WV2 `bbfc48be…` (전체 값은 되돌리기 기록과 Release에 있습니다)

</details>
