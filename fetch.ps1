# Downloads every Hugging Face model and dataset a job file names (hf: entries)
# into inputs\. Anything already there is skipped.
#
#     .\fetch.ps1 jobs\es-mx-whisper.yaml
#
# The work is done by training\steps\fetch.py, in the training environment.

param([Parameter(Mandatory = $true, Position = 0)][string]$Job)
$repo = $PSScriptRoot

if (-not (Test-Path $Job)) {
    Write-Host "STOP: there is no job file at $Job" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $repo "training\.venv"))) {
    Write-Host "STOP: the training environment isn't set up yet. Run .\setup.ps1 first." -ForegroundColor Red
    exit 1
}
$jobPath = (Resolve-Path $Job).Path

# Not "$ErrorActionPreference = Stop": Windows PowerShell 5.1 turns anything a
# program writes to stderr (such as Hugging Face's warnings) into an error when
# output is redirected, which would stop a working download. The exit code
# decides instead.
& uv run --project (Join-Path $repo "training") --locked python (Join-Path $repo "training\steps\fetch.py") $jobPath
exit $LASTEXITCODE
