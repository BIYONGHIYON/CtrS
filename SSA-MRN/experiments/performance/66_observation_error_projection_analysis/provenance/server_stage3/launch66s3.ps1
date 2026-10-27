$ErrorActionPreference = 'Stop'
$dst = 'C:\CtrS-p66s3'; $tmp = $env:TEMP
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$tmp\p66s3\run66s3.ps1", "$tmp\p66s3\prepare.py" $dst -Force
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="powershell.exe -NoProfile -ExecutionPolicy Bypass -File $dst\run66s3.ps1"}
if ($r.ReturnValue -ne 0) { throw 'independent launcher creation failed' }
$r | ConvertTo-Json | Set-Content -Encoding UTF8 "$dst\launcher.json"
'LAUNCHED pid ' + $r.ProcessId
