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

set STAGE=sft
set STEPS=100

if not "%~1"=="" set STAGE=%~1
if not "%~2"=="" set STEPS=%~2

".venv\Scripts\python.exe" "scripts\train_cycle.py" --stage %STAGE% --steps %STEPS%
if errorlevel 1 pause & exit /b 1

echo.
echo [GAI-1] Training cycle finished.
pause
exit /b 0

:help
echo GAI-1 incremental training cycle
echo.
echo Default chat SFT cycle, 100 optimizer steps:
echo   train_100.bat
echo.
echo Explicit modes:
echo   train_100.bat sft 100
echo   train_100.bat pretrain 100
echo.
echo Reports are written to reports\progress_*.jsonl
exit /b 0
