# Windows entry point for the reviewer-requested Tier B GPU runs.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run-review-experiments.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run-review-experiments.ps1 -Lane smoke
#
# The runs execute inside WSL2 (scripts/run-review-experiments.sh), not in
# native Windows Python. With Smart App Control on, Windows blocks uv's venv
# launcher and the unsigned DLLs of uv-managed CPython (_ctypes.pyd), so torch
# cannot load natively; WSL2 sees the same GPU through the NVIDIA driver and is
# not subject to that policy.
#
# This wrapper translates arguments, keeps Windows from sleeping on AC power
# for the duration (restoring the previous timeouts afterwards), and streams
# the WSL run's output. Safe to re-run after an interruption.
param(
    [string]$Scans = "24 37 40 55 63",
    [string]$Experiments = "b2 b4 b6",
    [ValidateSet("all", "smoke", "full")]
    [string]$Lane = "all",
    [string]$Distro = "Ubuntu",
    [switch]$SkipSetup,
    [switch]$WithScale,
    [switch]$Force,
    [switch]$PreflightOnly
)
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$WslRepo = (wsl.exe -d $Distro -- wslpath -a ($RepoRoot -replace '\\', '/')).Trim()
if (-not $WslRepo) { throw "could not translate $RepoRoot into a WSL path (is $Distro installed?)" }

$BashArgs = @("--scans", $Scans, "--experiments", $Experiments, "--lane", $Lane)
if ($SkipSetup) { $BashArgs += "--skip-setup" }
if ($WithScale) { $BashArgs += "--with-scale" }
if ($Force) { $BashArgs += "--force" }
if ($PreflightOnly) { $BashArgs += "--preflight-only" }

# AC sleep/hibernate timeouts, in seconds, so they can be put back exactly.
function Get-AcTimeout([string]$Setting) {
    $line = powercfg /query SCHEME_CURRENT SUB_SLEEP $Setting |
        Select-String 'Current AC Power Setting Index:\s*0x([0-9a-fA-F]+)'
    if ($line) { return [Convert]::ToInt32($line.Matches[0].Groups[1].Value, 16) }
    return $null
}
function Set-AcTimeout([string]$Setting, [int]$Seconds) {
    powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP $Setting $Seconds | Out-Null
    powercfg /setactive SCHEME_CURRENT | Out-Null
}

$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
if ($battery -and $battery.BatteryStatus -ne 2) {
    Write-Warning "Running on battery. Plug in: a multi-hour GPU run will drain it and throttle."
}

$saved = @{}
foreach ($setting in "STANDBYIDLE", "HIBERNATEIDLE") {
    $value = Get-AcTimeout $setting
    if ($null -ne $value -and $value -ne 0) {
        $saved[$setting] = $value
        Set-AcTimeout $setting 0
    }
}
if ($saved.Count) { Write-Host "Sleep on AC disabled for the run; restored afterwards." }

try {
    # --exec, not --: without it WSL re-joins the arguments for a shell, which
    # splits a quoted "24 37 40" into separate words.
    wsl.exe -d $Distro --cd $WslRepo --exec bash scripts/run-review-experiments.sh @BashArgs
    $exit = $LASTEXITCODE
} finally {
    foreach ($setting in $saved.Keys) { Set-AcTimeout $setting $saved[$setting] }
}
if ($exit -ne 0) {
    Write-Host "WSL run exited with $exit. Logs: \\wsl$\$Distro\home\<you>\zshift\tier_b\logs" -ForegroundColor Red
    exit $exit
}
