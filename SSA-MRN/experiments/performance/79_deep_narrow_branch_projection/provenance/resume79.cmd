@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p79\SSA-MRN\experiments\performance\79_deep_narrow_branch_projection
C:\CtrS-p73\.venv\Scripts\python.exe -u run.py --mode run --resume > C:\CtrS-p79\run\controller_resume1.log 2>&1
