# Runs every stage of a training job, in order, stopping at the first one that
# fails.
#
#     .\train.ps1 jobs\es-mx-whisper.yaml
#
# If a stage stops, fix what it says, then carry on from that stage:
#
#     .\train.ps1 jobs\es-mx-whisper.yaml -From W4
#
# A stage that already finished isn't repeated. The stage you start from with
# -From is redone; -Force redoes every stage from the start (or from -From).
# Before the first run, download the job's inputs with .\fetch.ps1 <job>.

param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Job,
    [string]$From = "",
    [switch]$Force
)
$repo = $PSScriptRoot

# Each stage: its ID, its script, and which environment runs it. Training
# stages run in training\ (CUDA PyTorch); stages that use volis's engine run in
# the converter's environment. The environments share nothing but runs\<job>\.
$stagesByKind = @{
    "whisper" = @(
        @("W1", "training\steps\whisper\w1_check.py", "training"),
        @("W2", "training\steps\whisper\w2_prepare.py", "training"),
        @("W3", "training\steps\whisper\w3_baseline.py", "training"),
        @("W4", "training\steps\whisper\w4_train.py", "training"),
        @("W5", "training\steps\whisper\w5_evaluate.py", "training"),
        @("W6", "training\steps\whisper\w6_merge.py", "training"),
        @("W7", "training\steps\whisper\w7_convert.py", "training"),
        @("W8", "training\steps\whisper\w8_int8.py", "converter")
    )
}

if (-not (Test-Path $Job)) {
    Write-Host "STOP: there is no job file at $Job" -ForegroundColor Red
    exit 1
}
$jobPath = (Resolve-Path $Job).Path
$kindLine = Select-String -Path $jobPath -Pattern '^kind:\s*(\S+)' | Select-Object -First 1
if (-not $kindLine) {
    Write-Host "STOP: $Job has no 'kind:' line" -ForegroundColor Red
    exit 1
}
$kind = $kindLine.Matches[0].Groups[1].Value
if (-not $stagesByKind.ContainsKey($kind)) {
    Write-Host "STOP: '$kind' jobs aren't built yet (known: $($stagesByKind.Keys -join ', '))" -ForegroundColor Red
    exit 1
}
$stages = $stagesByKind[$kind]
$ids = $stages | ForEach-Object { $_[0] }
$start = 0
if ($From) {
    $start = [array]::IndexOf($ids, $From.ToUpper())
    if ($start -lt 0) {
        Write-Host "STOP: -From $From isn't a stage of a $kind job ($($ids -join ', '))" -ForegroundColor Red
        exit 1
    }
}
foreach ($envName in @("training\.venv", ".venv")) {
    if (-not (Test-Path (Join-Path $repo $envName))) {
        Write-Host "STOP: $envName is missing. Run .\setup.ps1 first." -ForegroundColor Red
        exit 1
    }
}

# Not "$ErrorActionPreference = Stop": Windows PowerShell 5.1 turns anything a
# program writes to stderr into an error when output is redirected, which
# would stop working stages (Hugging Face and transformers warn on stderr).
# Each stage's exit code decides instead.
$started = Get-Date
Push-Location $repo
try {
    for ($n = $start; $n -lt $stages.Count; $n++) {
        $id, $script, $envName = $stages[$n]
        $project = if ($envName -eq "training") { Join-Path $repo "training" } else { $repo }
        $stageArgs = @("run", "--project", $project, "--locked", "--no-sync", "python", $script, $jobPath)
        if ($Force -or ($From -and $n -eq $start)) { $stageArgs += "--force" }

        Write-Host ""
        Write-Host ("=" * 70) -ForegroundColor Cyan
        Write-Host " $id  ($($n + 1) of $($stages.Count))  $script" -ForegroundColor Cyan
        Write-Host ("=" * 70) -ForegroundColor Cyan
        & uv @stageArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "$id stopped. Fix what it says above, then continue with:" -ForegroundColor Red
            Write-Host "    .\train.ps1 $Job -From $id" -ForegroundColor Yellow
            exit 1
        }
    }
}
finally {
    Pop-Location
}

$minutes = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
Write-Host ""
Write-Host "Every stage passed ($minutes minutes)." -ForegroundColor Green
