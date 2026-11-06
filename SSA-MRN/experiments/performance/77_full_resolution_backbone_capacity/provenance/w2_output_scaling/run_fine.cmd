@echo off
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-alpha
C:\CtrS-p73\.venv\Scripts\python.exe -u sweep.py --data C:\CtrS\PAN-MS --alphas 0.9 0.95 0.98 1.0 1.02 1.05 --out C:\CtrS-alpha\alpha_fine.json > C:\CtrS-alpha\fine.log 2>&1
