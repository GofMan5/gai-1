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

if not exist "data\tokenizer\gai1_tokenizer.json" (
  echo [GAI-1] tokenizer not found.
  echo Run first: .venv\Scripts\python.exe scripts\train_tokenizer.py
  pause
  exit /b 1
)

if not exist "outputs\gai1_train_gpu\last.pt" (
  echo [GAI-1] checkpoint not found.
  echo Run first: .venv\Scripts\python.exe scripts\train_pretrain.py
  pause
  exit /b 1
)

if "%~1"=="" (
  if exist "outputs\gai1_reasoning_lora\adapter.pt" (
    ".venv\Scripts\python.exe" "scripts\tui.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --adapter "outputs\gai1_reasoning_lora\adapter.pt" --level high --reasoning-view full
  ) else if exist "outputs\gai1_sft_lora\adapter.pt" (
    ".venv\Scripts\python.exe" "scripts\tui.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --adapter "outputs\gai1_sft_lora\adapter.pt" --level high
  ) else (
    ".venv\Scripts\python.exe" "scripts\tui.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --level high
  )
) else (
  ".venv\Scripts\python.exe" "scripts\tui.py" --checkpoint "outputs\gai1_train_gpu\last.pt" %*
)

echo.
echo [GAI-1] TUI closed.
pause
exit /b 0

:help
echo GAI-1 TUI launcher
echo.
echo Default:
echo   run_tui.bat
echo.
echo With custom options:
echo   run_tui.bat --level max --max-new-tokens 200
echo   run_tui.bat --context-length 262144 --rope-scaling linear
echo   run_tui.bat --adapter outputs\gai1_reasoning_lora\adapter.pt --reasoning-view full
echo   run_tui.bat --adapter outputs\gai1_sft_lora\adapter.pt --level high
echo   run_tui.bat --checkpoint outputs\quantized\last_int8.pt --level high
echo.
echo In TUI:
echo   /effort low^|medium^|high^|max
echo   /stats
echo   /clear
echo   /quit
exit /b 0
