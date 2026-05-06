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

set STEPS=500
if not "%~1"=="" set STEPS=%~1

".venv\Scripts\python.exe" "scripts\train_sft.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --data "data\raw\ru_turbo_alpaca_sample.jsonl" --lora --max-steps %STEPS%
if errorlevel 1 pause & exit /b 1

echo.
echo [GAI-1] Chat LoRA saved to outputs\gai1_sft_lora\adapter.pt
pause
exit /b 0

:help
echo GAI-1 chat LoRA trainer
echo.
echo Default:
echo   train_chat_lora.bat
echo.
echo Custom optimizer steps:
echo   train_chat_lora.bat 1500
exit /b 0
