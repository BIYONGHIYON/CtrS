# Extract and verify the 67 package in its own folder (CPU only). Does not prepare or start training.
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force C:/CtrS-p67 | Out-Null
Copy-Item "$env:TEMP/p67pkg/*" C:/CtrS-p67 -Force
tar -xzf C:/CtrS-p67/ctrs67-source.tgz -C C:/CtrS-p67
if ($LASTEXITCODE -ne 0) { throw 'extract failed' }
C:/CtrS/.venv/Scripts/python.exe C:/CtrS-p67/verify_ctrs67_server.py
if ($LASTEXITCODE -ne 0) { throw 'source verification failed' }
