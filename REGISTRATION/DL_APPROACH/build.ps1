@echo off
setlocal
REM Clean previous build artifacts (optional)
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
REM Build using the spec (Python 3.11 env must be active)
pyinstaller matcher_gui.spec --clean
echo.
echo Build finished. EXE is in .\dist\matcher_gui\matcher_gui.exe
pause
