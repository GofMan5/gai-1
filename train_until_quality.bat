@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="/?" goto :help
if "%~1"=="--help" goto :help
if "%~1"=="-h" goto :help

if not exist ".venv\Scripts\python.exe" (
  echo [GAI-1] .venv not found.
  echo Run first: scripts\setup_rtx3060_windows.ps1
  pause
  exit /b 1
)

".venv\Scripts\python.exe" "scripts\train_until_quality.py" %*
if errorlevel 1 pause & exit /b 1

echo.
echo [GAI-1] Training pipeline finished.
pause
exit /b 0

:help
echo GAI-1 staged training toward usable RU chat/reasoning.
echo.
echo Default first local target:
echo   train_until_quality.bat
echo.
echo Larger run:
echo   train_until_quality.bat --target-pretrain-step 12208 --chat-steps 1500 --reasoning-steps 3000
echo.
echo Fast data-only preparation:
echo   train_until_quality.bat --skip-pretrain --skip-chat --skip-reasoning --skip-eval
exit /b 0
