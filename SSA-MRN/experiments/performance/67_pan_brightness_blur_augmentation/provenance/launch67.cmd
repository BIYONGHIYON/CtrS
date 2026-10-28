@echo off
rem 67 stage 1 (WV3 seed 42, six arms). Start only when no other GPU job runs. Prepare and optimize are skipped if already done.
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p67\SSA-MRN\experiments\performance\67_pan_brightness_blur_augmentation
if not exist C:\CtrS-p67\run mkdir C:\CtrS-p67\run
if not exist C:\CtrS-p67\run\runtime_config.json C:\CtrS\.venv\Scripts\python.exe -u run.py --mode prepare > C:\CtrS-p67\run\prepare.log 2>&1 || exit /b 1
if not exist C:\CtrS-p67\run\preflight\optimization.json C:\CtrS\.venv\Scripts\python.exe -u run.py --mode optimize > C:\CtrS-p67\run\optimize.log 2>&1 || exit /b 2
C:\CtrS\.venv\Scripts\python.exe -u run.py --mode run > C:\CtrS-p67\run\controller.log 2>&1
