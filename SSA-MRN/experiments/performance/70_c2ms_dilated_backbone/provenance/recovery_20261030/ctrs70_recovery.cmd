@echo off
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d C:\CtrS-p70-71
C:\CtrS\.venv\Scripts\python.exe -X utf8 -u C:\CtrS-p70-71\ctrs70_recovery.py > C:\CtrS-p70-71\recovery.log 2>&1
