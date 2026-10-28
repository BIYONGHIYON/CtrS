# Safe stop of experiment 67 before a reboot: CTRL+C to the trainer console (its handler saves latest.pt), then wait for exit.
$ErrorActionPreference = 'Stop'
$run = 'C:\CtrS-p67\run'; $log = 'C:\CtrS-p67\stop67.log'
function Log($m) { Add-Content -Encoding UTF8 $log ((Get-Date -Format s) + ' ' + $m); $m }
$tr = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'CtrS-p67' -and $_.CommandLine -match 'train\.py' }
if (-not $tr) { Log 'no 67 trainer process'; exit 1 }
$target = ($tr | Sort-Object ProcessId | Select-Object -Last 1).ProcessId
Log ("trainer processes: " + (($tr | ForEach-Object { $_.ProcessId }) -join ','))
& C:\CtrS\.venv\Scripts\python.exe C:\CtrS-p67\ctrlc.py $target C:\CtrS-p67\ctrlc_result.json
Log ('ctrl-c: ' + (Get-Content C:\CtrS-p67\ctrlc_result.json -Raw))
for ($i = 0; $i -lt 60; $i++) {
  Start-Sleep 5
  $left = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\.exe$' }
  if (-not $left) { break }
}
if ($left) { Log ('still running: ' + (($left | ForEach-Object { $_.ProcessId }) -join ',')); exit 2 }
Log 'all python processes (67 trainer and controller) exited'
Get-ChildItem "$run\runs" -Directory | ForEach-Object { $s = "$($_.FullName)\status.json"; if (Test-Path $s) { Log ($_.Name + ' ' + (Get-Content $s -Raw).Replace("`r`n",' ')) } }
