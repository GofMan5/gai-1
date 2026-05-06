@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [GAI-1] .venv not found.
  echo Run first: scripts\setup_rtx3060_windows.ps1
  pause
  exit /b 1
)

".venv\Scripts\python.exe" "scripts\autotune_rtx3060.py" %*
if errorlevel 1 pause & exit /b 1

echo.
echo [GAI-1] Autotuned config written to configs\train_gpu_autotuned.json
pause
