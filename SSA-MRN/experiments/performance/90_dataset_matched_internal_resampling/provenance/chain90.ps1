# 90 chain (user decision 2026-11-20 21h: if the 89 reduced stage B picks c* = 0.25, tidy up the server and run the 89 confirmation and
# the 90 single-seed comparison automatically). Started detached; stops at the first problem and never kills a training.
$log = "C:\CtrS-p90\chain.log"; $py = "C:\CtrS-p73\.venv\Scripts\python.exe"
$run89 = "C:\CtrS-p89\run"; $src = "C:\CtrS-p89\SSA-MRN"; $dst = "C:\CtrS-p90\SSA-MRN"
$stage = "C:\CtrS-p90\stage\90_dataset_matched_internal_resampling"; $d90 = "$dst\experiments\performance\90_dataset_matched_internal_resampling"
function Note($m) { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) }
Note "chain90 started (pid $PID); waiting for the 89 reduced stage B"
while (-not (Test-Path "$run89\fine_c_complete.json") -and -not (Test-Path "$run89\fine_c_stopped.json")) { Start-Sleep -Seconds 60 }
if (Test-Path "$run89\fine_c_stopped.json") { Note "89 reduced stage B stopped: $(Get-Content "$run89\fine_c_stopped.json" -Raw)"; Note "chain90 stopped (nothing started)"; exit 2 }
$sel = Get-Content "$run89\fine_c_complete.json" -Raw | ConvertFrom-Json
Note "89 c* = $($sel.c_star) (lambda* $($sel.lambda_star), boundary $($sel.at_boundary))"
if ([math]::Abs([double]$sel.c_star - 0.25) -gt 1e-9) { Note "c* is not 0.25; waiting for the user decision (nothing started)"; exit 2 }
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_fine_c|CtrS-p89\\run\\runtime_config' }) { Start-Sleep -Seconds 30 }
# tidy-up: only empty stage-B folders (a worker.lock and nothing else) and temporary status scripts; no result, checkpoint or log is deleted
$n = 0; Get-ChildItem "$run89\runs" -Directory -Filter '*_BL*' | ForEach-Object { $c = @(Get-ChildItem $_.FullName -Force); if ($c.Count -eq 1 -and $c[0].Name -eq 'worker.lock') { Remove-Item $_.FullName -Recurse -Force; $n++ } }
$tmp = @('st89.ps1','st89b.ps1','st89c.ps1','st89f.ps1','cmdl.ps1','sz.ps1','vals.ps1','cmp90.ps1','src90.json','unblock89.ps1','unblock89b.ps1') | ForEach-Object { "C:\CtrS-p89\$_" } | Where-Object { Test-Path $_ }
$tmp | ForEach-Object { Remove-Item $_ -Force }
Note ("tidy-up: removed {0} empty stage-B folders and {1} temporary scripts; C: free {2:N1} GB" -f $n, @($tmp).Count, ((Get-PSDrive C).Free / 1GB))
robocopy $src $dst /E /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { Note "robocopy failed $LASTEXITCODE"; exit 4 }
New-Item -ItemType Directory -Force -Path $d90 | Out-Null
Copy-Item "$stage\*" $d90 -Recurse -Force
Note "source tree copied from p89 and the 90 folder overlaid"
Set-Location $d90
foreach ($mode in 'prepare','optimize','run') {
    Note "90 $mode start"
    & $py -u run.py --mode $mode *> "C:\CtrS-p90\$mode.log"
    $code = $LASTEXITCODE; Note "90 $mode exit $code"
    if ($code -ne 0) { Note "chain90 stopped after $mode"; exit $code }
}
Note "chain90 finished"
