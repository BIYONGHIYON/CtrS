# 90 chain after the preflight fix (2026-11-21 02h): the first optimize stopped in the 58 preflight (B/C model built without the
# training sensor). prepare had passed; run optimize again (controls are cached) and then the controller. Stops at the first problem.
$log = "C:\CtrS-p90\chain.log"; $py = "C:\CtrS-p73\.venv\Scripts\python.exe"
$d90 = "C:\CtrS-p90\SSA-MRN\experiments\performance\90_dataset_matched_internal_resampling"
function Note($m) { Add-Content -Path $log -Value ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) }
Note "chain90_resume started (pid $PID); train.py sha256 $((Get-FileHash "$d90\train.py").Hash.Substring(0,16).ToLower())"
Set-Location $d90
foreach ($mode in 'optimize','run') {
    Note "90 $mode start"
    & $py -u run.py --mode $mode *> "C:\CtrS-p90\${mode}_2.log"
    $code = $LASTEXITCODE; Note "90 $mode exit $code"
    if ($code -ne 0) { Note "chain90 stopped after $mode"; exit $code }
}
Note "chain90 finished"
