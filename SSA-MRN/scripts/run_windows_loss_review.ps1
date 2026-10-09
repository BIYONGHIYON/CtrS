param(
    [ValidateSet('start','status','logs')][string]$Action = 'status',
    [string]$Root = 'C:\CtrS-budget-suite\SSA-MRN',
    [string]$Python = 'C:\CtrS\.venv\Scripts\python.exe',
    [string]$DataRoot = 'C:\CtrS\SSA-MRN\data\dataset',
    [string]$Output = 'C:\Users\trainer\windows_losses_20261010',
    [string]$Evaluator = "$PSScriptRoot\evaluate_windows_losses.py"
)
$ErrorActionPreference = 'Stop'
$Log = "$Output.log"
if ($Action -eq 'start') {
    if (Test-Path $Output) { throw 'Use a new output directory; existing evidence must remain intact.' }
    foreach ($Path in @($Root,$Python,$DataRoot,$Evaluator)) {
        if (-not (Test-Path $Path)) { throw "Missing path: $Path" }
    }
    $Training = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -match 'train_controlled|controlled_suite.py|train.py'
    }
    if ($Training) { throw 'Training is active. Do not overlap the evidence evaluation.' }
    $Command = 'cmd.exe /d /c ""{0}" -u "{1}" --root "{2}" --output "{3}" --data-root "{4}" > "{5}" 2>&1"' -f $Python,$Evaluator,$Root,$Output,$DataRoot,$Log
    $Result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$Command}
    if ($Result.ReturnValue -ne 0) { throw "Process creation failed: $($Result.ReturnValue)" }
    $Result | Select-Object ProcessId,ReturnValue
} elseif ($Action -eq 'logs') {
    Get-Content $Log -Tail 20
} else {
    $State = Join-Path $Output 'evaluation_state.json'
    if (Test-Path $State) { Get-Content $State }
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python' -and $_.CommandLine -like "*$Evaluator*"
    } | Select-Object ProcessId,CommandLine
}
