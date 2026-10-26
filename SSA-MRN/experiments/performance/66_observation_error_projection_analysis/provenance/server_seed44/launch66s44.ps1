$ErrorActionPreference = 'Stop'
foreach ($id in 10944, 7720) { $p = Get-CimInstance Win32_Process -Filter "ProcessId=$id"; if ($p -and $p.CommandLine -match 'wait66s44|launcher.json') { Stop-Process -Id $id -Force } }
Start-Sleep 2
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\CtrS-p66s44\wait66s44.ps1'}
if ($r.ReturnValue -ne 0) { throw 'independent launcher creation failed' }
$r | ConvertTo-Json | Set-Content -Encoding UTF8 C:\CtrS-p66s44\launcher.json
'LAUNCHED pid ' + $r.ProcessId
