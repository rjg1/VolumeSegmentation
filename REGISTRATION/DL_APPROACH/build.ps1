param(
  [ValidateSet('onedir','onefile')]
  [string]$mode = 'onedir'
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

# Paths all stay inside this folder
$dist = '.'
$work = '.\build'
$spec = '.\spec'
$rtmp = '.\_pyi_runtime'

# Clean previous outputs
taskkill /F /IM roi_matcher.exe 2>$null | Out-Null
Remove-Item .\dist -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\build -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\spec  -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\roi_matcher.exe -Force -ErrorAction SilentlyContinue

$common = @(
  '.\matcher_gui.py',
  '--windowed',
  '--name', 'roi_matcher',
  '--distpath', $dist,
  '--workpath', $work,
  '--specpath', $spec,
  '--noconfirm', '--clean', '--noupx',
  '--collect-submodules', 'numpy',
  '--collect-data', 'numpy',
  '--hidden-import=numpy.core',
  '--hidden-import=numpy._distributor_init'
)

if ($mode -eq 'onefile') {
  $args = @($common + @('--onefile', '--runtime-tmpdir', $rtmp))
} else {
  $args = @($common + @('--onedir'))
}

Write-Host "Running: pyinstaller $($args -join ' ')" -ForegroundColor Cyan
pyinstaller @args

if ($mode -eq 'onefile') {
  if (Test-Path .\roi_matcher.exe) {
    Write-Host "Built .\roi_matcher.exe" -ForegroundColor Green
  } else {
    Write-Host "Build finished but EXE missing (likely quarantined). Consider Defender exclusions above." -ForegroundColor Yellow
  }
} else {
  if (Test-Path .\roi_matcher\roi_matcher.exe) {
    Write-Host "Built .\roi_matcher\roi_matcher.exe" -ForegroundColor Green
  } else {
    Write-Host "Build finished but app folder missing. Check log for errors." -ForegroundColor Yellow
  }
}
