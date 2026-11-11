@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p78\SSA-MRN\experiments\performance\78_seed_ensemble_distillation
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode run > C:\CtrS-p78\run\controller.log 2>&1
