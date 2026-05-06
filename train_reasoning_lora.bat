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

if not exist "outputs\gai1_train_gpu\last.pt" (
  echo [GAI-1] base checkpoint not found.
  echo Run first: .venv\Scripts\python.exe scripts\train_pretrain.py
  pause
  exit /b 1
)

if not exist "data\raw\ru_turbo_alpaca_sample.jsonl" (
  ".venv\Scripts\python.exe" "scripts\download_ru_turbo_alpaca.py" --max-records 5000
  if errorlevel 1 pause & exit /b 1
)

set STEPS=1000
set STYLE=visible
set LEVEL=high

if not "%~1"=="" set STEPS=%~1
if not "%~2"=="" set STYLE=%~2
if not "%~3"=="" set LEVEL=%~3

".venv\Scripts\python.exe" "scripts\build_reasoning_sft.py" --input "data\raw\ru_turbo_alpaca_sample.jsonl" --output "data\sft\reasoning_ru.jsonl" --style %STYLE% --level %LEVEL%
if errorlevel 1 pause & exit /b 1

".venv\Scripts\python.exe" "scripts\train_sft.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --data "data\sft\reasoning_ru.jsonl" --out "outputs\gai1_reasoning_lora" --lora --max-steps %STEPS%
if errorlevel 1 pause & exit /b 1

echo.
echo [GAI-1] Reasoning LoRA saved to outputs\gai1_reasoning_lora\adapter.pt
echo Launch with:
echo   run_tui.bat --adapter outputs\gai1_reasoning_lora\adapter.pt --reasoning-view full
pause
exit /b 0

:help
echo GAI-1 reasoning LoRA trainer
echo.
echo Default:
echo   train_reasoning_lora.bat
echo.
echo Custom:
echo   train_reasoning_lora.bat 3000 visible high
echo   train_reasoning_lora.bat 3000 controller max
echo.
echo Modes:
echo   visible    trains public short reasoning + answer
echo   controller trains hidden-controller prompt + answer
exit /b 0
