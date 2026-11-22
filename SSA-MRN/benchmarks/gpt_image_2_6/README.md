# GPT Image 2.6 Sunburst

| 평가자 | 설정 | 상태 |
|---|---|---|
| QB 이병현 · GF2 김윤지 · WV3 김경찬 · WV2 송준현 | `zeroshot` · 학습 없음 · 단일 실행 | 평가 완료 |

**출처:** 동국 AI CHAT에서 제공한 상용 이미지 생성 서비스입니다. 논문·공개 가중치가 없으며 모델 이름과 설정은 생성 당시 기록을 따릅니다.

**결과:** 네 센서 RR test 각 20장을 생성하고 RGB 3밴드에서 실제 MATLAB으로 채점했습니다. PSNR은 아래 표에, 장면별 수치와 생성·복원·검사 기록은 센서별 결과 폴더에 보관했습니다.

[평가 안내](../README.md#generative) · [결과 비교 사이트](https://biyonghiyon.github.io/CtrS/)

## 결과

각 값은 RR test 20장의 평균입니다. PSNR은 밴드별 평균·peak 2047로 통일했으며, GF2의 센서 peak 1023 값도 원본 JSON에 있습니다. SAM은 도(°)이고, Q2n은 RGB 3밴드에 0 밴드 하나를 채운 DLPan Q4입니다. 전체 4·8밴드 논문 수치와 직접 비교하지 않습니다.

| 센서 | PSNR (dB) | SAM (°) | ERGAS | SCC | Q2n | 변환 한계 대조군 대비 PSNR |
|---|---:|---:|---:|---:|---:|---:|
| [QB](results/QB/matlab_summary.json) | 31.170 | 3.8760 | 8.5855 | 0.8815 | 0.7426 | +1.953 dB |
| [GF2](results/GF2/matlab_summary.json) | 36.893 | 1.5070 | 2.5274 | 0.9363 | 0.7693 | -1.060 dB |
| [WV3](results/WV3/matlab_summary.json) | 31.555 | 2.3352 | 5.0033 | 0.9086 | 0.7623 | +2.330 dB |
| [WV2](results/WV2/matlab_summary.json) | 31.030 | 3.1371 | 4.7349 | 0.9193 | 0.8168 | +3.335 dB |

## 해석과 정리

변환 한계 대조군은 LMS의 RGB 입력 이미지를 모델 없이 그대로 되돌린 결과입니다. 위 차이는 같은 RGB 밴드·peak 2047에서 계산했습니다. 시드를 지정할 수 없는 단일 실행이므로 반복 실행의 편차는 측정하지 않았습니다. SMOC와의 같은 RGB 3밴드 비교는 결과 비교 사이트에서 확인할 수 있습니다.

관측 일치 오차·PAN 구조 일치도·위치 어긋남은 정답의 같은 값과 함께 센서별 `checks.json`에 있습니다. 이 지표만으로 할루시네이션의 유무를 확정하지 않습니다.

<details>
<summary>방법 · 설정과 생성 경로</summary>

- **설정:** 1024x1024, quality high, background opaque.
- **입력·복원:** 같은 위치의 LMS RGB와 PAN을 각각 고정된 밝기 범위로 1024×1024 입력 이미지로 만들었습니다. 출력을 4×4 면적 평균으로 256×256으로 줄이고 LMS 입력의 밝기 범위를 역변환해 DN으로 복원했습니다. 정답은 생성 입력에 넣지 않았습니다.
- **생성 경로:** QB·GF2는 API, WV3는 동국 AI CHAT 웹 화면에서 만들었습니다. WV2는 제공된 원본 묶음을 평가했으며 고정 설정·프롬프트를 사용했다는 사용자 확인이 있습니다. WV2의 생성 시각·크레딧·재시도 내역은 서비스 로그로 확인하지 못했습니다.
- **채점:** MATLAB R2025b, DLPan-Toolbox `a34f884`의 `score_suite.m`, RGB 3밴드입니다. 같은 배열의 Python 채점과의 차이도 각 요약 JSON에 있습니다.

</details>

<details>
<summary>그림 · 재현 자료</summary>

고정 6장면(0·1·3·7·14·17)의 원본 크기·확대 이미지와 장면별 수치를 보관했습니다. 이 이미지의 비교 모델은 생성 당시 C2MS-SP이며 당시 비교 기록으로 보존합니다. 현재 SMOC 비교는 결과 비교 사이트를 이용합니다.

| 센서 | 채점·장면별 수치 | 생성·복원 | 정답 없는 지표 | 고정 6장면 이미지 |
|---|---|---|---|---|
| QB | [요약](results/QB/matlab_summary.json) · [장면별](results/QB/matlab_scenes.json) | [생성](results/QB/generation.json) · [복원](results/QB/restore.json) | [검사](results/QB/checks.json) | [이미지와 수치](results/QB/images/) |
| GF2 | [요약](results/GF2/matlab_summary.json) · [장면별](results/GF2/matlab_scenes.json) | [생성](results/GF2/generation.json) · [복원](results/GF2/restore.json) | [검사](results/GF2/checks.json) | [이미지와 수치](results/GF2/images/) |
| WV3 | [요약](results/WV3/matlab_summary.json) · [장면별](results/WV3/matlab_scenes.json) | [생성](results/WV3/generation.json) · [복원](results/WV3/restore.json) | [검사](results/WV3/checks.json) | [이미지와 수치](results/WV3/images/) |
| WV2 | [요약](results/WV2/matlab_summary.json) · [장면별](results/WV2/matlab_scenes.json) | [생성](results/WV2/generation.json) · [복원](results/WV2/restore.json) | [검사](results/WV2/checks.json) | [이미지와 수치](results/WV2/images/) |

- [예측·원본 이미지 Release](https://github.com/BIYONGHIYON/CtrS/releases/tag/predictions-benchmarks-gpt_image_2_6-zeroshot): 네 센서의 `rr.zip`·`raw.zip`을 보관했습니다. 예측 zip의 SHA-256은 각 `restore.json`에 있습니다.
- 담당 이슈: [QB #209](https://github.com/BIYONGHIYON/CtrS/issues/209) · [GF2 #210](https://github.com/BIYONGHIYON/CtrS/issues/210) · [WV3 #211](https://github.com/BIYONGHIYON/CtrS/issues/211) · [WV2 #212](https://github.com/BIYONGHIYON/CtrS/issues/212).

</details>
