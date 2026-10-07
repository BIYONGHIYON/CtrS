param([ValidateSet("start","resume-latest","resume-best","status","inspect","logs","check")][string]$Command="status",[switch]$Follow,[switch]$Raw)
$ErrorActionPreference="Stop"
$Python="C:\CtrS\.venv\Scripts\python.exe"
if($Command -eq "logs" -and $Follow -and -not $Raw){
    $Arguments=@("$PSScriptRoot\live_progress.py")
}else{
    $Arguments=@("$PSScriptRoot\controlled_suite.py",$Command)
    if($Follow){$Arguments+="--follow"}
}
& $Python @Arguments
if($LASTEXITCODE -ne 0){throw "Suite command failed: $LASTEXITCODE"}
