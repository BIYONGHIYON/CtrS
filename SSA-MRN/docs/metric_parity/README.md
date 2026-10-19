# MATLAB 기준 지표 대조

현재 Python 지표와 고정 커밋의 DLPan-Toolbox MATLAB 함수를 실제 MATLAB R2025b·Image Processing Toolbox에서 대조합니다. 실제 결과는 실행 후 기록합니다. 실행 성공과 수치 일치는 구분합니다.

공개 저장소의 MathWorks 공식 Actions 자동 라이선스를 사용합니다. 고정 seed 합성 영상만 사용하고 원본 학습·평가 데이터는 업로드하지 않습니다. MATLAB이 직접 생성한 imresize 선형 연산 행렬을 별도 보존해 서버의 실제 PAN에도 같은 연산을 적용할 수 있습니다.

생산 지표 코드와 기존 실험 수치는 변경하지 않았습니다. 기존35번 기록의 Q2n 곱셈·PAN resize 차이를 확인한 뒤, 검증된 범위에서만 보고서 문구를 갱신합니다.
