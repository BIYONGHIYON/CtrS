@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /d C:\CtrS-p70-71
C:\CtrS\.venv\Scripts\python.exe -X utf8 -u C:\CtrS-p70-71\SSA-MRN\experiments\performance\70_c2ms_dilated_backbone\provenance\run_suite.py > C:\CtrS-p70-71\suite.log 2>&1
