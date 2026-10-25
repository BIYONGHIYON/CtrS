$ErrorActionPreference = 'Stop'
tar -xzf C:/CtrS-p62/ctrs62-source.tgz -C C:/CtrS-p62
if ($LASTEXITCODE -ne 0) { throw 'extract failed' }
C:/CtrS/.venv/Scripts/python.exe C:/CtrS-p62/verify_ctrs62_server.py
if ($LASTEXITCODE -ne 0) { throw 'source verification failed' }
$active = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if ($active) { throw 'another Python process is running; refusing to launch' }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c C:\CtrS-p62\launch62.cmd'}
if ($r.ReturnValue -ne 0) { throw 'independent launcher creation failed' }
$r | ConvertTo-Json | Set-Content -Encoding UTF8 C:/CtrS-p62/launcher.json
'LAUNCHED pid ' + $r.ProcessId
