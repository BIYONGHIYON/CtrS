@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p79\SSA-MRN\experiments\performance\79_deep_narrow_branch_projection
if not exist C:\CtrS-p79\run mkdir C:\CtrS-p79\run
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode prepare > C:\CtrS-p79\run\prepare.log 2>&1 || exit /b 1
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode optimize > C:\CtrS-p79\run\optimize.log 2>&1 || exit /b 2
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode run > C:\CtrS-p79\run\controller.log 2>&1
