@echo off
chcp 65001 >nul
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
if /I "%CMD%"=="advanced" goto :tools_menu

echo [GAI-1] Unknown command: %CMD%
echo.
goto :help

:menu
cls
echo ============================================================
echo  GAI-1 - локальная русскоязычная модель
echo ============================================================
echo.
echo Что хотите сделать?
echo.
echo  1. Поговорить с моделью
echo     Открывает чат. LoRA-адаптеры подключатся автоматически, если они уже есть.
echo.
echo  2. Подготовить проект
echo     Проверяет Python/CUDA/GPU, скачивает русские данные и обучает токенизатор.
echo.
echo  3. Обучать модель
echo     База, chat LoRA, reasoning LoRA, короткие циклы и полный pipeline.
echo.
echo  4. Проверить качество и файлы
echo     Eval gates, список чекпоинтов/адаптеров/отчетов, сравнение квантизации.
echo.
echo  5. Экспорт и запуск API
echo     Квантование модели и локальный OpenAI-compatible API.
echo.
echo  6. Диагностика и инструменты
echo     Doctor, автоподбор RTX 3060, тесты и полный список команд.
echo.
echo  0. Exit
echo.
set /p "MENU=Выберите 1-6 или 0: "
if "%MENU%"=="1" goto :chat
if "%MENU%"=="2" goto :first_setup
if "%MENU%"=="3" goto :train_menu
if "%MENU%"=="4" goto :quality_menu
if "%MENU%"=="5" goto :export_menu
if "%MENU%"=="6" goto :tools_menu
if "%MENU%"=="0" exit /b 0
echo [GAI-1] Не понял пункт: %MENU%
pause
goto :menu

:train_menu
cls
echo ============================================================
echo  GAI-1 - обучение модели
echo ============================================================
echo.
echo  1. Базовое обучение
echo     Обучает основную модель с нуля или продолжает с последнего чекпоинта.
echo.
echo  2. Дообучить для чата
echo     Обучает chat LoRA, чтобы модель лучше отвечала в диалоге.
echo.
echo  3. Дообучить reasoning
echo     Создает reasoning SFT-данные и обучает reasoning LoRA.
echo.
echo  4. Короткий цикл на 100 шагов
echo     Быстрая итерация обучения и пример ответа для контроля прогресса.
echo.
echo  5. Полный pipeline качества
echo     Готовит данные, обучает модель/LoRA и запускает проверки.
echo.
echo  B. Назад
echo  0. Exit
echo.
set /p "MENU=Выберите 1-5, B или 0: "
if "%MENU%"=="1" goto :pretrain
if "%MENU%"=="2" goto :chat_lora
if "%MENU%"=="3" goto :reasoning_lora
if "%MENU%"=="4" goto :cycle
if "%MENU%"=="5" goto :quality
if /I "%MENU%"=="B" goto :menu
if "%MENU%"=="0" exit /b 0
echo [GAI-1] Не понял пункт: %MENU%
pause
goto :train_menu

:quality_menu
cls
echo ============================================================
echo  GAI-1 - проверки и артефакты
echo ============================================================
echo.
echo  1. Проверить качество
echo     Запускает eval gates: loss, русский текст, повторы, prompt echo.
echo.
echo  2. Показать файлы модели
echo     Показывает чекпоинты, LoRA-адаптеры, отчеты и экспортированные модели.
echo.
echo  3. Оценить оставшееся обучение
echo     Примерно считает, сколько шагов еще нужно до связного общения.
echo.
echo  4. Сравнить квантованную модель
echo     Сравнивает FP16 и INT4/INT8 чекпоинт на одинаковых промптах.
echo.
echo  B. Назад
echo  0. Exit
echo.
set /p "MENU=Выберите 1-4, B или 0: "
if "%MENU%"=="1" goto :eval
if "%MENU%"=="2" goto :list
if "%MENU%"=="3" goto :estimate
if "%MENU%"=="4" goto :compare_quantized_menu
if /I "%MENU%"=="B" goto :menu
if "%MENU%"=="0" exit /b 0
echo [GAI-1] Не понял пункт: %MENU%
pause
goto :quality_menu

:export_menu
cls
echo ============================================================
echo  GAI-1 - экспорт и запуск
echo ============================================================
echo.
echo  1. Квантовать модель в INT8
echo     Меньше размер, обычно меньше потерь качества.
echo.
echo  2. Квантовать модель в INT4
echo     Еще меньше размер, но качество может просесть сильнее.
echo.
echo  3. Запустить локальный API
echo     Поднимает OpenAI-compatible сервер на localhost.
echo.
echo  B. Назад
echo  0. Exit
echo.
set /p "MENU=Выберите 1-3, B или 0: "
if "%MENU%"=="1" goto :quantize_int8
if "%MENU%"=="2" goto :quantize_int4
if "%MENU%"=="3" goto :serve
if /I "%MENU%"=="B" goto :menu
if "%MENU%"=="0" exit /b 0
echo [GAI-1] Не понял пункт: %MENU%
pause
goto :export_menu

:tools_menu
cls
echo ============================================================
echo  GAI-1 - диагностика и инструменты
echo ============================================================
echo.
echo  1. Проверить окружение
echo     Проверяет Python, PyTorch, CUDA и доступность GPU.
echo.
echo  2. Автоподбор RTX 3060
echo     Подбирает более быстрые настройки обучения для локальной видеокарты.
echo.
echo  3. Запустить тесты
echo     Запускает pytest для проверки проекта.
echo.
echo  4. Расширенные команды
echo     Показывает полный список CLI-команд и алиасов.
echo.
echo  B. Назад
echo  0. Exit
echo.
set /p "MENU=Выберите 1-4, B или 0: "
if "%MENU%"=="1" goto :doctor
if "%MENU%"=="2" goto :autotune
if "%MENU%"=="3" goto :test
if "%MENU%"=="4" goto :help
if /I "%MENU%"=="B" goto :menu
if "%MENU%"=="0" exit /b 0
echo [GAI-1] Не понял пункт: %MENU%
pause
goto :tools_menu

:help
echo GAI-1 - единый запускатель
echo.
echo Обычный запуск:
echo   gai.bat
echo.
echo Основной путь с нуля:
echo   gai.bat setup
echo   gai.bat doctor
echo   gai.bat prepare
echo   gai.bat pretrain
echo   gai.bat chat-lora 500
echo   gai.bat reasoning-lora 1000 visible high
echo   gai.bat chat
echo.
echo Частые команды:
echo   gai.bat chat                  открыть чат
echo   gai.bat prepare               скачать данные и обучить токенизатор
echo   gai.bat pretrain              обучать базовую модель
echo   gai.bat chat-lora 500         дообучить для чата
echo   gai.bat reasoning-lora 1000   дообучить reasoning
echo   gai.bat cycle sft 100         короткий цикл обучения
echo   gai.bat eval                  проверить качество
echo   gai.bat list                  показать артефакты
echo   gai.bat quantize 8            экспорт INT8
echo   gai.bat serve                 запустить локальный API
echo   gai.bat doctor                проверить окружение
echo.
echo Все команды и алиасы:
echo   setup, doctor, prepare, tokenizer, pretrain
echo   chat-lora, reasoning-lora, cycle, quality, autotune
echo   chat, tui, serve, eval, quantize, list, artifacts
echo   estimate, compare-quantized, test, advanced, help
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

:first_setup
if not exist "%PY%" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_rtx3060_windows.ps1"
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
call :need_python
if errorlevel 1 (
  pause
  exit /b 1
)
"%PY%" "scripts\check_accelerator.py"
if errorlevel 1 (
  pause
  exit /b 1
)
goto :prepare

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

:quantize_int8
call :quantize 8
exit /b %ERRORLEVEL%

:quantize_int4
call :quantize 4
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

:compare_quantized_menu
set "QCKPT=outputs\quantized\last_int8.pt"
set /p "QCKPT=Путь к INT4/INT8 чекпоинту [%QCKPT%]: "
if "%QCKPT%"=="" set "QCKPT=outputs\quantized\last_int8.pt"
call :compare_quantized "%QCKPT%"
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
