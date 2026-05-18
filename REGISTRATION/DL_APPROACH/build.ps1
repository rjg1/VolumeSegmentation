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
Remove-Item .\dist -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\build -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\spec  -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\roi_matcher.exe -Force -ErrorAction SilentlyContinue

# 🔑 Optimised common args
$common = @(
  '.\matcher_gui.py',
  '--windowed',
  '--name', 'roi_matcher',

  '--distpath', $dist,
  '--workpath', $work,
  '--specpath', $spec,

  '--noconfirm',
  '--clean',
  '--noupx',

  # ✅ Keep numpy but DON'T pull everything
  '--hidden-import=numpy',
  '--collect-data', 'numpy',

  # cellpose
  '--collect-all=cellpose',

  # ✅ Strip unused junk (big size reduction)
  '--exclude-module', 'numpy.f2py',
  '--exclude-module', 'numpy.f2py.tests',
  '--exclude-module', 'numpy.tests',
  '--exclude-module', 'pytest',
  '--exclude-module', 'unittest'
  '--exclude-module', 'torch.cuda',
  '--exclude-module', 'torch.backends.cuda',
  '--exclude-module', 'torch.distributed'
)

$torchLib = python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
$torchDlls = Get-ChildItem $torchLib -Filter *.dll
foreach ($dll in $torchDlls) {
    $common += @(
        '--add-binary',
        "$($dll.FullName);torch\lib"
    )
}

if ($mode -eq 'onefile') {
  $args = @(
    $common + @(
      '--onefile',
      '--runtime-tmpdir', $rtmp
    )
  )
} else {
  $args = @(
    $common + @(
      '--onedir'
    )
  )
}

Write-Host "Running: pyinstaller $($args -join ' ')" -ForegroundColor Cyan
pyinstaller @args

# Output check
if ($mode -eq 'onefile') {
  if (Test-Path .\roi_matcher.exe) {
    Write-Host "Built .\roi_matcher.exe" -ForegroundColor Green
  } else {
    Write-Host "Build finished but EXE missing (likely quarantined)." -ForegroundColor Yellow
  }
} else {
  if (Test-Path .\roi_matcher\roi_matcher.exe) {
    Write-Host "Built .\roi_matcher\roi_matcher.exe" -ForegroundColor Green
  } else {
    Write-Host "Build finished but app folder missing. Check log." -ForegroundColor Yellow
  }
}