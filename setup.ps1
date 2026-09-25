# Sets up model-converter on this machine. Run it once after cloning, and again
# whenever the lock file changes or a reinstall is needed:
#
#     .\setup.ps1              install (or bring up to date)
#     .\setup.ps1 -Reinstall   reinstall every package, e.g. after security
#                              software removed a DLL and an exclusion was added
#
# It keeps uv's download cache, the Python interpreter and the environment all
# inside this folder, so every native file lives under one path: the one to
# give security software as an exclusion. Nothing is changed outside this folder.
# Afterwards, run the steps with "uv run python ..." as the README shows.

param([switch]$Reinstall)
$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

function Fail($message) {
    Write-Host "`nSTOP: $message" -ForegroundColor Red
    exit 1
}

Write-Host "== Checking tools"
if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
    Fail "this needs 64-bit x86 Windows (found $env:PROCESSOR_ARCHITECTURE)."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "git is not installed. Install it from https://git-scm.com and open a new terminal."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail "uv is not installed. Install it with: winget install astral-sh.uv   (then open a new terminal)"
}
$uvVersion = ((uv --version) -split " ")[1]
$v = [version]$uvVersion
if ($v -lt [version]"0.11.26" -or $v -ge [version]"0.12") {
    Fail "uv is $uvVersion; this project is locked with uv 0.11.x (0.11.26 or newer). Install that version: uv self update 0.11.26"
}
Write-Host "  git: $((git --version) -replace 'git version ','')"
Write-Host "  uv:  $uvVersion"

# For this process and the uv commands it starts only.
$env:UV_CACHE_DIR = Join-Path $repo ".uv\cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $repo ".uv\python"
# Copy files out of the cache rather than hard-linking them: a scanner that
# quarantines a cached file then cannot take the installed copy with it.
$env:UV_LINK_MODE = "copy"
# Install the uv-managed CPython from .python-version, never an installed one.
# (Only here: as a project setting, a later "uv run" without UV_PYTHON_INSTALL_DIR
# would not recognise the interpreter in .uv\python and rebuild the environment.)
$env:UV_PYTHON_PREFERENCE = "only-managed"
Write-Host "  cache:  $env:UV_CACHE_DIR"
Write-Host "  python: $env:UV_PYTHON_INSTALL_DIR"

Write-Host "`n== Installing (uv sync --locked)"
Push-Location $repo
try {
    $syncArgs = @("sync", "--locked")
    if ($Reinstall) { $syncArgs += "--reinstall" }
    & uv @syncArgs
    if ($LASTEXITCODE -ne 0) {
        Fail ("uv sync failed. If it says the lock file needs updating, pyproject.toml was edited " +
              "without re-locking; run 'uv lock' on the machine that made the change and commit uv.lock. " +
              "If it failed writing or loading a .dll, see the README's 'Security software' section.")
    }

    Write-Host "`n== Checking the environment (0_doctor.py)"
    & uv run --locked --no-sync python whisper-to-onnx\steps\0_doctor.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}

if (-not (Test-Path (Join-Path $repo "machine.yaml"))) {
    Write-Host "`nNext: copy machine.example.yaml to machine.yaml and set cnverc_path." -ForegroundColor Yellow
}
