# Starts experiment 65 after experiment 64 has finished (or failed to start) on this server, which itself follows 63
# (scheduled by the user on 2026-10-26). Never starts while any python process is running.
$ErrorActionPreference = 'Stop'
$log = 'C:\CtrS-p65\watcher.log'
function Note($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding UTF8 $log }
function Done64 {
  if (Test-Path 'C:\CtrS-p64\run\all_complete.json') { return 'complete' }
  if (Test-Path 'C:\CtrS-p64\run\controller_failure.json') { return 'controller failure' }
  if ((Test-Path 'C:\CtrS-p64\watcher.log') -and (Select-String -Path 'C:\CtrS-p64\watcher.log' -Pattern 'NOT started' -Quiet)) { return 'not started' }
  return $null
}
Note 'watcher started; waiting for experiment 63 complete and experiment 64 finished (or not started)'
$deadline = (Get-Date).AddHours(24)
$idle = 0
while ((Get-Date) -lt $deadline) {
  $s64 = Done64
  # 64 launched but its chain stopped without all_complete (e.g. prepare/optimize failure): no python for 5 checks in a row
  if (-not $s64 -and (Test-Path 'C:\CtrS-p64\watcher.log') -and (Select-String -Path 'C:\CtrS-p64\watcher.log' -Pattern 'LAUNCHED' -Quiet)) {
    if (Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }) { $idle = 0 } else { $idle += 1 }
    if ($idle -ge 5) { $s64 = 'stopped without completion' }
  }
  if ((Test-Path 'C:\CtrS-p63\run\all_complete.json') -and $s64) {
    Start-Sleep -Seconds 60
    $active = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
    if ($active) { Note ('python still running: ' + ($active.ProcessId -join ',')); Start-Sleep -Seconds 60; continue }
    Note ("63 complete, 64 $s64, no python running; staging 65")
    try {
      tar -xzf C:/CtrS-p65/ctrs65-source.tgz -C C:/CtrS-p65
      if ($LASTEXITCODE -ne 0) { throw 'extract failed' }
      & C:/CtrS/.venv/Scripts/python.exe C:/CtrS-p65/verify_ctrs65_server.py *>> $log
      if ($LASTEXITCODE -ne 0) { throw 'source verification failed' }
      $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c C:\CtrS-p65\launch65.cmd'}
      if ($r.ReturnValue -ne 0) { throw 'launcher creation failed' }
      $r | ConvertTo-Json | Set-Content -Encoding UTF8 C:/CtrS-p65/launcher.json
      Note ('65 LAUNCHED pid ' + $r.ProcessId); exit 0
    } catch { Note ('65 NOT started: ' + $_); exit 2 }
  }
  if ((Test-Path 'C:\CtrS-p63\run\controller_failure.json') -and -not (Test-Path 'C:\CtrS-p63\run\all_complete.json')) { Note '63 controller failure found; 65 NOT started'; exit 1 }
  Start-Sleep -Seconds 60
}
Note 'deadline reached; 65 NOT started'; exit 3
