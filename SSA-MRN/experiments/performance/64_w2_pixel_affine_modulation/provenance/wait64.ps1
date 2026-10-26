# Starts experiment 64 after experiment 63 has finished on this server (scheduled by the user on 2026-10-26).
$ErrorActionPreference = 'Stop'
$log = 'C:\CtrS-p64\watcher.log'
function Note($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding UTF8 $log }
Note 'watcher started; waiting for C:\CtrS-p63\run\all_complete.json'
$deadline = (Get-Date).AddHours(14)
while ((Get-Date) -lt $deadline) {
  if (Test-Path 'C:\CtrS-p63\run\controller_failure.json') { Note '63 controller failure found; 64 NOT started'; exit 1 }
  if (Test-Path 'C:\CtrS-p63\run\all_complete.json') {
    Start-Sleep -Seconds 60
    $active = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
    if ($active) { Note ('python still running: ' + ($active.ProcessId -join ',')); Start-Sleep -Seconds 60; continue }
    Note '63 complete and no python running; staging 64'
    try {
      tar -xzf C:/CtrS-p64/ctrs64-source.tgz -C C:/CtrS-p64
      if ($LASTEXITCODE -ne 0) { throw 'extract failed' }
      & C:/CtrS/.venv/Scripts/python.exe C:/CtrS-p64/verify_ctrs64_server.py *>> $log
      if ($LASTEXITCODE -ne 0) { throw 'source verification failed' }
      $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c C:\CtrS-p64\launch64.cmd'}
      if ($r.ReturnValue -ne 0) { throw 'launcher creation failed' }
      $r | ConvertTo-Json | Set-Content -Encoding UTF8 C:/CtrS-p64/launcher.json
      Note ('64 LAUNCHED pid ' + $r.ProcessId); exit 0
    } catch { Note ('64 NOT started: ' + $_); exit 2 }
  }
  Start-Sleep -Seconds 60
}
Note 'deadline reached without 63 completion; 64 NOT started'; exit 3
