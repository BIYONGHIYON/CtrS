$ErrorActionPreference = 'Stop'
tar -xzf C:/CtrS-p65s3/ctrs65s3-source.tgz -C C:/CtrS-p65s3
if ($LASTEXITCODE -ne 0) { throw 'extract failed' }
C:/CtrS/.venv/Scripts/python.exe C:/CtrS-p65s3/verify_ctrs65s3_server.py
if ($LASTEXITCODE -ne 0) { throw 'source verification failed' }
$active = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($active) { throw 'another Python process is running; refusing to launch' }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c C:\CtrS-p65s3\launch65s3.cmd'}
if ($r.ReturnValue -ne 0) { throw 'independent launcher creation failed' }
$r | ConvertTo-Json | Set-Content -Encoding UTF8 C:/CtrS-p65s3/launcher.json
'LAUNCHED pid ' + $r.ProcessId
