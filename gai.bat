@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"

if "%~1"=="" goto :menu
if /I "%~1"=="help" goto :help
if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help
if "%~1"=="/?" goto :help

set "CMD=%~1"
shift /1

if /I "%CMD%"=="setup" goto :setup
if /I "%CMD%"=="doctor" goto :doctor
if /I "%CMD%"=="prepare" goto :prepare
if /I "%CMD%"=="tokenizer" goto :tokenizer
if /I "%CMD%"=="pretrain" goto :pretrain
if /I "%CMD%"=="chat-lora" goto :chat_lora
if /I "%CMD%"=="reasoning-lora" goto :reasoning_lora
if /I "%CMD%"=="cycle" goto :cycle
if /I "%CMD%"=="quality" goto :quality
if /I "%CMD%"=="autotune" goto :autotune
if /I "%CMD%"=="chat" goto :chat
if /I "%CMD%"=="tui" goto :chat
if /I "%CMD%"=="serve" goto :serve
if /I "%CMD%"=="eval" goto :eval
if /I "%CMD%"=="quantize" goto :quantize
if /I "%CMD%"=="list" goto :list
if /I "%CMD%"=="artifacts" goto :list
if /I "%CMD%"=="estimate" goto :estimate
if /I "%CMD%"=="compare-quantized" goto :compare_quantized
if /I "%CMD%"=="test" goto :test

echo [GAI-1] Unknown command: %CMD%
echo.
goto :help

:menu
cls
echo ============================================================
echo  GAI-1 unified launcher
echo ============================================================
echo.
echo  1. Chat TUI
echo  2. Prepare RU data + tokenizer
echo  3. Pretrain base model
echo  4. Train chat LoRA
echo  5. Train reasoning LoRA
echo  6. 100-step training cycle
echo  7. Full quality pipeline
echo  8. Autotune RTX 3060 config
echo  9. Serve local API
echo 10. Eval gates
echo 11. Quantize checkpoint
echo 12. List artifacts
echo 13. Doctor
echo 14. Run tests
echo  0. Exit
echo.
set /p "MENU=Choose: "
if "%MENU%"=="1" goto :chat
if "%MENU%"=="2" goto :prepare
if "%MENU%"=="3" goto :pretrain
if "%MENU%"=="4" goto :chat_lora
if "%MENU%"=="5" goto :reasoning_lora
if "%MENU%"=="6" goto :cycle
if "%MENU%"=="7" goto :quality
if "%MENU%"=="8" goto :autotune
if "%MENU%"=="9" goto :serve
if "%MENU%"=="10" goto :eval
if "%MENU%"=="11" goto :quantize
if "%MENU%"=="12" goto :list
if "%MENU%"=="13" goto :doctor
if "%MENU%"=="14" goto :test
if "%MENU%"=="0" exit /b 0
echo [GAI-1] Unknown menu item: %MENU%
pause
goto :menu

:help
echo GAI-1 unified launcher
echo.
echo Usage:
echo   gai.bat
echo   gai.bat setup
echo   gai.bat prepare [fineweb_docs] [sft_records]
echo   gai.bat pretrain [extra train_pretrain.py args]
echo   gai.bat chat-lora [steps]
echo   gai.bat reasoning-lora [steps] [visible^|controller] [low^|medium^|high^|max]
echo   gai.bat cycle [sft^|pretrain] [steps]
echo   gai.bat quality [extra train_until_quality.py args]
echo   gai.bat autotune [extra autotune args]
echo   gai.bat chat [extra tui.py args]
echo   gai.bat serve [extra serve.py args]
echo   gai.bat eval [extra eval_gates.py args]
echo   gai.bat quantize [4^|8]
echo   gai.bat list
echo   gai.bat estimate [extra estimate args]
echo   gai.bat compare-quantized outputs\quantized\last_int4.pt
echo   gai.bat test
echo   gai.bat doctor
echo.
echo Recommended first path:
echo   gai.bat setup
echo   gai.bat prepare
echo   gai.bat pretrain
echo   gai.bat chat-lora 500
echo   gai.bat reasoning-lora 1000 visible high
echo   gai.bat chat
exit /b 0

:need_python
if exist "%PY%" exit /b 0
echo [GAI-1] .venv not found.
echo Run:
echo   gai.bat setup
exit /b 1

:need_tokenizer
if exist "data\tokenizer\gai1_tokenizer.json" exit /b 0
echo [GAI-1] tokenizer not found.
echo Run:
echo   gai.bat prepare
exit /b 1

:need_checkpoint
if exist "outputs\gai1_train_gpu\last.pt" exit /b 0
echo [GAI-1] base checkpoint not found.
echo Run:
echo   gai.bat pretrain
exit /b 1

:run
%*
if errorlevel 1 (
  echo.
  echo [GAI-1] Command failed.
  pause
  exit /b 1
)
echo.
echo [GAI-1] Done.
pause
exit /b 0

:setup
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_rtx3060_windows.ps1"
if errorlevel 1 (
  pause
  exit /b 1
)
echo.
echo [GAI-1] Setup finished.
pause
exit /b 0

:doctor
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :run "%PY%" "scripts\check_accelerator.py"
exit /b %ERRORLEVEL%

:prepare
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
set "FINEWEB_DOCS=1200"
set "SFT_RECORDS=5000"
if not "%~1"=="" set "FINEWEB_DOCS=%~1"
if not "%~2"=="" set "SFT_RECORDS=%~2"
"%PY%" "scripts\download_fineweb2_ru.py" --max-docs %FINEWEB_DOCS%
if errorlevel 1 (
  pause
  exit /b 1
)
"%PY%" "scripts\train_tokenizer.py"
if errorlevel 1 (
  pause
  exit /b 1
)
"%PY%" "scripts\download_ru_turbo_alpaca.py" --max-records %SFT_RECORDS%
if errorlevel 1 (
  pause
  exit /b 1
)
echo.
echo [GAI-1] Data and tokenizer prepared.
pause
exit /b 0

:tokenizer
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :run "%PY%" "scripts\train_tokenizer.py" %*
exit /b %ERRORLEVEL%

:pretrain
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :need_tokenizer
if errorlevel 1 (
  pause
  exit /b 1
)
if "%~1"=="" (
  call :run "%PY%" "scripts\train_pretrain.py"
) else (
  call :run "%PY%" "scripts\train_pretrain.py" %*
)
exit /b %ERRORLEVEL%

:chat_lora
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :need_checkpoint
if errorlevel 1 (
  pause
  exit /b 1
)
set "STEPS=500"
if not "%~1"=="" set "STEPS=%~1"
if not exist "data\raw\ru_turbo_alpaca_sample.jsonl" (
  "%PY%" "scripts\download_ru_turbo_alpaca.py" --max-records 5000
  if errorlevel 1 (
  pause
  exit /b 1
)
)
call :run "%PY%" "scripts\train_sft.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --data "data\raw\ru_turbo_alpaca_sample.jsonl" --lora --max-steps %STEPS%
exit /b %ERRORLEVEL%

:reasoning_lora
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :need_checkpoint
if errorlevel 1 (
  pause
  exit /b 1
)
set "STEPS=1000"
set "STYLE=visible"
set "LEVEL=high"
if not "%~1"=="" set "STEPS=%~1"
if not "%~2"=="" set "STYLE=%~2"
if not "%~3"=="" set "LEVEL=%~3"
if not exist "data\raw\ru_turbo_alpaca_sample.jsonl" (
  "%PY%" "scripts\download_ru_turbo_alpaca.py" --max-records 5000
  if errorlevel 1 (
  pause
  exit /b 1
)
)
"%PY%" "scripts\build_reasoning_sft.py" --input "data\raw\ru_turbo_alpaca_sample.jsonl" --output "data\sft\reasoning_ru.jsonl" --style %STYLE% --level %LEVEL%
if errorlevel 1 (
  pause
  exit /b 1
)
call :run "%PY%" "scripts\train_sft.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --data "data\sft\reasoning_ru.jsonl" --out "outputs\gai1_reasoning_lora" --lora --max-steps %STEPS%
exit /b %ERRORLEVEL%

:cycle
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
set "STAGE=sft"
set "STEPS=100"
if not "%~1"=="" set "STAGE=%~1"
if not "%~2"=="" set "STEPS=%~2"
call :run "%PY%" "scripts\train_cycle.py" --stage %STAGE% --steps %STEPS%
exit /b %ERRORLEVEL%

:quality
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :run "%PY%" "scripts\train_until_quality.py" %*
exit /b %ERRORLEVEL%

:autotune
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :run "%PY%" "scripts\autotune_rtx3060.py" %*
exit /b %ERRORLEVEL%

:chat
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :need_tokenizer
if errorlevel 1 (
  pause
  exit /b 1
)
call :need_checkpoint
if errorlevel 1 (
  pause
  exit /b 1
)
if "%~1"=="" (
  if exist "outputs\gai1_reasoning_lora\adapter.pt" (
    call :run "%PY%" "scripts\tui.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --adapter "outputs\gai1_reasoning_lora\adapter.pt" --level high --reasoning-view full
  ) else if exist "outputs\gai1_sft_lora\adapter.pt" (
    call :run "%PY%" "scripts\tui.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --adapter "outputs\gai1_sft_lora\adapter.pt" --level high
  ) else (
    call :run "%PY%" "scripts\tui.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --level high
  )
) else (
  call :run "%PY%" "scripts\tui.py" --checkpoint "outputs\gai1_train_gpu\last.pt" %*
)
exit /b %ERRORLEVEL%

:serve
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :need_tokenizer
if errorlevel 1 (
  pause
  exit /b 1
)
call :need_checkpoint
if errorlevel 1 (
  pause
  exit /b 1
)
if "%~1"=="" (
  if exist "outputs\gai1_reasoning_lora\adapter.pt" (
    call :run "%PY%" "scripts\serve.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --adapter "outputs\gai1_reasoning_lora\adapter.pt"
  ) else if exist "outputs\gai1_sft_lora\adapter.pt" (
    call :run "%PY%" "scripts\serve.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --adapter "outputs\gai1_sft_lora\adapter.pt"
  ) else (
    call :run "%PY%" "scripts\serve.py" --checkpoint "outputs\gai1_train_gpu\last.pt"
  )
) else (
  call :run "%PY%" "scripts\serve.py" %*
)
exit /b %ERRORLEVEL%

:eval
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :need_tokenizer
if errorlevel 1 (
  pause
  exit /b 1
)
call :need_checkpoint
if errorlevel 1 (
  pause
  exit /b 1
)
if "%~1"=="" (
  if exist "outputs\gai1_reasoning_lora\adapter.pt" (
    call :run "%PY%" "scripts\eval_gates.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --adapter "outputs\gai1_reasoning_lora\adapter.pt" --data "data\processed\fineweb2_ru_val.jsonl"
  ) else if exist "outputs\gai1_sft_lora\adapter.pt" (
    call :run "%PY%" "scripts\eval_gates.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --adapter "outputs\gai1_sft_lora\adapter.pt" --data "data\processed\fineweb2_ru_val.jsonl"
  ) else (
    call :run "%PY%" "scripts\eval_gates.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --data "data\processed\fineweb2_ru_val.jsonl"
  )
) else (
  call :run "%PY%" "scripts\eval_gates.py" %*
)
exit /b %ERRORLEVEL%

:quantize
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :need_checkpoint
if errorlevel 1 (
  pause
  exit /b 1
)
set "BITS=8"
if not "%~1"=="" set "BITS=%~1"
call :run "%PY%" "scripts\export_quantized.py" --checkpoint "outputs\gai1_train_gpu\last.pt" --bits %BITS%
exit /b %ERRORLEVEL%

:list
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :run "%PY%" "scripts\list_artifacts.py"
exit /b %ERRORLEVEL%

:estimate
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :run "%PY%" "scripts\estimate_training_steps.py" %*
exit /b %ERRORLEVEL%

:compare_quantized
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
if "%~1"=="" (
  echo [GAI-1] Usage: gai.bat compare-quantized outputs\quantized\last_int4.pt
  pause
  exit /b 1
)
call :run "%PY%" "scripts\compare_quantized.py" --quantized-checkpoint "%~1"
exit /b %ERRORLEVEL%

:test
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
call :run "%PY%" -m pytest -q
exit /b %ERRORLEVEL%
