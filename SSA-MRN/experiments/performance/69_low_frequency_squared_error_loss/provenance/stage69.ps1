# Extract and verify the 69 package in its own folder (CPU only). Does not prepare or start training.
$ErrorActionPreference = 'Stop'
if (Test-Path C:/CtrS-p69/run) { throw 'C:/CtrS-p69/run exists; refusing to overwrite a runtime' }
New-Item -ItemType Directory -Force C:/CtrS-p69 | Out-Null
Copy-Item "$env:TEMP/p69pkg/*" C:/CtrS-p69 -Force
tar -xzf C:/CtrS-p69/ctrs69-source.tgz -C C:/CtrS-p69
if ($LASTEXITCODE -ne 0) { throw 'extract failed' }
C:/CtrS/.venv/Scripts/python.exe C:/CtrS-p69/verify_ctrs69_server.py
if ($LASTEXITCODE -ne 0) { throw 'source verification failed' }
