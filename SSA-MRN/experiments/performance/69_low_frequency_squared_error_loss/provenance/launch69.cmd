@echo off
rem 69 stage 1 (QB and WV3 seed 42, two arms). Start only when no other GPU job runs. Prepare and optimize are skipped if already done.
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p69\SSA-MRN\experiments\performance\69_low_frequency_squared_error_loss
if not exist C:\CtrS-p69\run mkdir C:\CtrS-p69\run
if not exist C:\CtrS-p69\run\runtime_config.json C:\CtrS\.venv\Scripts\python.exe -u run.py --mode prepare > C:\CtrS-p69\run\prepare.log 2>&1 || exit /b 1
if not exist C:\CtrS-p69\run\preflight\optimization.json C:\CtrS\.venv\Scripts\python.exe -u run.py --mode optimize > C:\CtrS-p69\run\optimize.log 2>&1 || exit /b 2
C:\CtrS\.venv\Scripts\python.exe -u run.py --mode run > C:\CtrS-p69\run\controller.log 2>&1
