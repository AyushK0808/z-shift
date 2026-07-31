# Windows (PowerShell) setup for the pinned MASt3R checkout.
# Mirrors scripts/setup-mast3r.sh.
param(
    [switch]$IncludeUpstreamRequirements
)
$ErrorActionPreference = "Stop"

$Repo = "https://github.com/naver/mast3r"
$Pinned = "f5209afc300cec36239a7ac992263f36847bbba0"
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Target = Join-Path $PSScriptRoot "..\third_party\mast3r"
$Target = [System.IO.Path]::GetFullPath($Target)

$Mast3rProjectContent = @'
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "mast3r"
version = "0.1.0"
description = "MASt3R: Matching And Stereo 3D Reconstruction"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["mast3r*"]
'@

$Dust3rProjectContent = $Mast3rProjectContent -replace 'name = "mast3r"', 'name = "dust3r"' `
  -replace 'description = "MASt3R: Matching And Stereo 3D Reconstruction"', `
  'description = "DUSt3R: Geometric 3D Vision Made Easy"' `
  -replace 'include = \["mast3r\*"\]', 'include = ["dust3r*"]'

function Write-NoBomUtf8 {
    param([string]$Path, [string]$Content)
    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    [System.IO.File]::WriteAllText($Path, $Content, (New-Object System.Text.UTF8Encoding($false)))
}

function Invoke-Checked {
    param([scriptblock]$Block)
    # Native stderr is turned into error records when captured under
    # ErrorActionPreference=Stop; rely on the exit code instead.
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Block
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $oldEap
    if ($exit -ne 0) {
        throw "Command failed with exit code ${exit}: $Block"
    }
}

function Invoke-BestEffort {
    param([scriptblock]$Block)
    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $Block } catch { }
    $ErrorActionPreference = $oldEap
}

if (Test-Path -LiteralPath $Target) {
    Write-Host "mast3r already cloned at $Target"
    Invoke-Checked { git -C $Target checkout $Pinned }
} else {
    Write-Host "Cloning mast3r into $Target ..."
    Invoke-Checked { git clone $Repo $Target }
    Invoke-Checked { git -C $Target checkout $Pinned }
}
Invoke-Checked { git -C $Target submodule update --init --recursive }

# Both repos are flat package dirs with no setup.py/pyproject.toml.
# Write minimal ones so pip install -e works without sys.path hacks.
# (third_party/ is gitignored so these won't pollute the repo.)

$Mast3rPyProjectPath = Join-Path $Target "pyproject.toml"
if (-not (Test-Path -LiteralPath $Mast3rPyProjectPath)) {
    Write-NoBomUtf8 -Path $Mast3rPyProjectPath -Content $Mast3rProjectContent
}

$Dust3rDir = Join-Path $Target "dust3r"
$Dust3rPyProjectPath = Join-Path $Dust3rDir "pyproject.toml"
if (-not (Test-Path -LiteralPath $Dust3rPyProjectPath)) {
    Write-NoBomUtf8 -Path $Dust3rPyProjectPath -Content $Dust3rProjectContent
}

Write-Host "Installing mast3r Python package (editable) ..."
Invoke-Checked { uv pip install -e $Target }

Write-Host "Installing dust3r Python package (editable) ..."
Invoke-Checked { uv pip install -e $Dust3rDir }

# The project's own dependency set (uv sync --dev) covers the MASt3R runtime
# requirements; installing the upstream requirements.txt here would override
# pinned versions (e.g. numpy<2). Only do it explicitly with
# -IncludeUpstreamRequirements.
if ($IncludeUpstreamRequirements) {
    Write-Host "Installing upstream requirements.txt files (opt-in) ..."
    Invoke-BestEffort { uv pip install -r (Join-Path $Target "requirements.txt") }
    Invoke-BestEffort { uv pip install -r (Join-Path $Dust3rDir "requirements.txt") }
}

Write-Host "Compiling RoPE CUDA kernels (optional, speeds up positional embeddings) ..."
Push-Location (Join-Path $Dust3rDir "croco\models\curope")
try {
    # Pin to the repo-root project: 'uv run' would otherwise discover the
    # mast3r/dust3r pyproject.toml and create a bare environment without
    # torch or setuptools.
    uv run --project $RepoRoot python setup.py build_ext --inplace
    if ($LASTEXITCODE -eq 0) {
        Write-Host "RoPE kernels compiled."
    } else {
        Write-Host "RoPE kernel compilation failed (non-CUDA env). PyTorch fallback will be used."
    }
} catch {
    Write-Host "RoPE kernel compilation failed (non-CUDA env). PyTorch fallback will be used."
}
Pop-Location

Write-Host ""
Write-Host "mast3r setup complete."
Write-Host 'Verify: uv run python -c "from mast3r.model import AsymmetricMASt3R; print(''OK'')"'
Write-Host ""
Write-Host "Note: 'uv sync' will remove the editable mast3r/dust3r installs."
Write-Host "Re-run this script after any 'uv sync'."
