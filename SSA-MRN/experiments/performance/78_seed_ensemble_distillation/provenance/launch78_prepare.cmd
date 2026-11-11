@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p78\SSA-MRN\experiments\performance\78_seed_ensemble_distillation
if not exist C:\CtrS-p78\run mkdir C:\CtrS-p78\run
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode prepare > C:\CtrS-p78\run\prepare.log 2>&1 || exit /b 1
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode optimize > C:\CtrS-p78\run\optimize.log 2>&1 || exit /b 2
echo done > C:\CtrS-p78\run\prepare_optimize_done.txt
