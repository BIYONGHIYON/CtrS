# 89: wait for the original controller (chain89.ps1, "89 run exit N"). On exit 3 start continue_without_margin.py, which continues only
# when stopped.json names nothing but the validation PSNR margin (user decision 2026-11-19 14h). Exit 0 or other codes: do nothing.
$log = "C:\CtrS-p89\watch_margin.log"
$py = "C:\CtrS-p73\.venv\Scripts\python.exe"
$d89 = "C:\CtrS-p89\SSA-MRN\experiments\performance\89_spectral_coarse_loss_weights"
$script = "$d89\provenance\continue_without_margin.py"
function Note($m) { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) }
Note "watcher started (pid $PID)"
while ($true) {
    $c = Get-Content "C:\CtrS-p89\chain.log" -ErrorAction SilentlyContinue
    $line = $c | Where-Object { $_ -match "89 run exit" } | Select-Object -Last 1
    if ($line) { break }
    $alive = Get-CimInstance Win32_Process -Filter "ProcessId=13768" -ErrorAction SilentlyContinue
    if (-not $alive) { Note "chain89 ended without a run exit line; nothing started"; exit 2 }
    Start-Sleep -Seconds 60
}
Note "original controller: $line"
if ($line -notmatch "exit 3$") { Note "not a stop (exit 3); nothing started"; exit 0 }
Note "continue_without_margin start"
$p = Start-Process -FilePath $py -ArgumentList @("-u", $script) -WorkingDirectory $d89 -RedirectStandardOutput "C:\CtrS-p89\run_continue.log" -RedirectStandardError "C:\CtrS-p89\run_continue.err.log" -NoNewWindow -PassThru -Wait
Note "continue_without_margin exit $($p.ExitCode)"
