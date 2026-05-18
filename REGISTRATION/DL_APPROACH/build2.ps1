param(
  [ValidateSet('onedir','onefile')]
  [string]$mode = 'onedir'
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

# Output paths
$out = '.\dist'
$build = '.\build_nuitka'

# Clean previous outputs
Remove-Item $out -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $build -Recurse -Force -ErrorAction SilentlyContinue

# Base args
$args = @(
  'matcher_gui.py',

  '--standalone',                 # bundle dependencies
  '--output-dir=.\dist',
  '--remove-output',

  '--assume-yes-for-downloads',   # auto-download dependencies

  '--enable-plugin=numpy',        # proper numpy handling
  '--windows-console-mode=disable', # like --windowed

  '--jobs=8',                     # parallel compile (adjust if needed)

  # optional cleanup / size reduction
  '--nofollow-import-to=pytest',
  '--nofollow-import-to=unittest',
  '--nofollow-import-to=numpy.f2py',
  '--nofollow-import-to=numpy.tests'
)

if ($mode -eq 'onefile') {
  $args += '--onefile'
} else {
  $args += '--output-filename=roi_matcher.exe'
}

Write-Host "Running: python -m nuitka $($args -join ' ')" -ForegroundColor Cyan
python -m nuitka @args

# Output check
if ($mode -eq 'onefile') {
  if (Test-Path "$out\matcher_gui.exe") {
    Write-Host "Built $out\matcher_gui.exe" -ForegroundColor Green
  } else {
    Write-Host "Build finished but EXE missing (check Defender or errors)" -ForegroundColor Yellow
  }
} else {
  if (Test-Path "$out\matcher_gui.dist\matcher_gui.exe") {
    Write-Host "Built $out\matcher_gui.dist\matcher_gui.exe" -ForegroundColor Green
  } else {
    Write-Host "Build finished but folder missing. Check log." -ForegroundColor Yellow
  }
}