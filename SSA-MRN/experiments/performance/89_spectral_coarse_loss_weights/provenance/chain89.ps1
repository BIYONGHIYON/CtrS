# 89: wait for the 88 chain -> copy the 88 source tree + overlay the 89 folder -> gate88 -> prepare -> optimize -> run.
# Stops at the first nonzero exit (user decision 2026-11-19: on any problem do not continue).
$log = "C:\CtrS-p89\chain.log"
$py = "C:\CtrS-p73\.venv\Scripts\python.exe"
$d89 = "C:\CtrS-p89\SSA-MRN\experiments\performance\89_spectral_coarse_loss_weights"
function Note($m) { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) }
Note "chain started (pid $PID), waiting for the 88 chain"
while ($true) {
    $c88 = Get-Content "C:\CtrS-p88\chain.log" -ErrorAction SilentlyContinue
    if ($c88 -match "chain finished") { break }
    if ($c88 -match "stopped after") { Note "88 chain stopped with an error; 89 not started"; exit 2 }
    $alive = Get-CimInstance Win32_Process -Filter "ProcessId=12192" -ErrorAction SilentlyContinue
    if (-not $alive) { Note "88 chain process ended without 'chain finished'; 89 not started"; exit 2 }
    Start-Sleep -Seconds 60
}
Note "88 chain finished"
if (Test-Path "C:\CtrS-p89\SSA-MRN") { Note "C:\CtrS-p89\SSA-MRN already exists; not overwriting"; exit 2 }
robocopy "C:\CtrS-p88\SSA-MRN" "C:\CtrS-p89\SSA-MRN" /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Note "copy of the 88 source tree failed ($LASTEXITCODE)"; exit 2 }
robocopy "C:\CtrS-p89\stage\89_spectral_coarse_loss_weights" $d89 /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Note "copy of the 89 folder failed ($LASTEXITCODE)"; exit 2 }
Note "source ready"
foreach ($mode in @("gate88", "prepare", "optimize", "run")) {
    Note "89 $mode start"
    $p = Start-Process -FilePath $py -ArgumentList @("-u", "run.py", "--mode", $mode) -WorkingDirectory $d89 -RedirectStandardOutput "C:\CtrS-p89\$mode.log" -RedirectStandardError "C:\CtrS-p89\$mode.err.log" -NoNewWindow -PassThru -Wait
    Note "89 $mode exit $($p.ExitCode)"
    if ($p.ExitCode -ne 0) { Note "stopped after $mode"; exit $p.ExitCode }
}
Note "chain finished"
