$ErrorActionPreference='Stop'
$active=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
if($active){throw 'Python job already active'}
$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c C:\CtrS-p70-71\ctrs70_recovery.cmd'}
if($r.ReturnValue -ne 0){throw 'launch failed'}
$r | ConvertTo-Json | Set-Content -Encoding UTF8 C:/CtrS-p70-71/recovery_launcher.json
$r | ConvertTo-Json -Compress
