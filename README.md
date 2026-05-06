# GAI-1

[![tests](https://github.com/GofMan5/gai-1/actions/workflows/tests.yml/badge.svg)](https://github.com/GofMan5/gai-1/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

GAI-1 is a Russian-first trainable LLM scaffold. It includes a decoder-only Transformer, RU BPE tokenizer, CUDA pretraining, masked SFT, LoRA adapters, reasoning effort levels, terminal chat UI, quantized checkpoint export, eval gates, and a local API server.

This repository ships code, configs, tests, and docs. It does not ship production model weights. Short smoke runs prove the pipeline works; real chat quality requires much more data and training time.

## Features

- Russian-first tokenizer training with BPE.
- CUDA/fp16 training profile for RTX 3060 12GB.
- Decoder-only Transformer with RoPE, RMSNorm, SwiGLU, and optional MoE.
- Pretraining and supervised fine-tuning scripts.
- Prompt-masked SFT so loss is applied to assistant answers only.
- LoRA adapter training and loading.
- Reasoning effort modes: `low`, `medium`, `high`, `max`.
- Rich TUI with streaming output, token stats, VRAM stats, and reasoning trace.
- Quantized checkpoint export.
- OpenAI-style local chat API scaffold.

## Install

Windows PowerShell:

```powershell
git clone https://github.com/GofMan5/gai-1.git
cd gai-1
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .[dev]
```

For the local RTX 3060 setup helper:

```powershell
.\scripts\setup_rtx3060_windows.ps1
```

## Quickstart

Check CUDA:

```powershell
.\.venv\Scripts\python.exe .\scripts\check_accelerator.py
```

Prepare a small Russian pretraining sample and tokenizer:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_fineweb2_ru.py --max-docs 1200
.\.venv\Scripts\python.exe .\scripts\train_tokenizer.py
```

Train the base checkpoint:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_pretrain.py
```

Make it follow chat instructions with LoRA SFT:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_ru_turbo_alpaca.py --max-records 5000
.\train_chat_lora.bat 500
```

Run the TUI:

```powershell
.\run_tui.bat
```

`run_tui.bat` auto-loads `outputs\gai1_sft_lora\adapter.pt` when it exists.

## One Main Training Config

The primary GPU config is:

```text
configs/train_gpu.json
```

It is tuned for RTX 3060 12GB:

- CUDA with fp16 AMP.
- TF32 matmul where available.
- Micro-batch `2`.
- Gradient accumulation `16`.
- Effective batch `32`.

## TUI Commands

Inside the TUI:

```text
/effort low
/effort medium
/effort high
/effort max
/stats
/clear
/quit
```

Manual launch:

```powershell
.\.venv\Scripts\python.exe .\scripts\tui.py --checkpoint .\outputs\gai1_train_gpu\last.pt --adapter .\outputs\gai1_sft_lora\adapter.pt --level high
```

## Local API

```powershell
.\.venv\Scripts\python.exe .\scripts\serve.py --checkpoint .\outputs\gai1_train_gpu\last.pt
```

- Health: `http://127.0.0.1:8000/health`
- Chat: `POST http://127.0.0.1:8000/v1/chat/completions`

## Quantization

```powershell
.\.venv\Scripts\python.exe .\scripts\export_quantized.py --checkpoint .\outputs\gai1_train_gpu\last.pt --bits 8
.\.venv\Scripts\python.exe .\scripts\export_quantized.py --checkpoint .\outputs\gai1_train_gpu\last.pt --bits 4
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Artifact Policy

Do not commit:

- checkpoints or adapters;
- raw datasets;
- tokenizer artifacts generated from local data;
- cache folders;
- virtual environments;
- reports with private prompts or system details.

The `.gitignore` is configured for this. Publish trained weights only as deliberate release artifacts with a model card, dataset summary, eval results, and hashes.

## Data And License Notes

Project source code is Apache-2.0 licensed. Datasets downloaded by scripts keep their own licenses and terms.

Prototype SFT uses `IlyaGusev/ru_turbo_alpaca`, which is useful for local experiments but has dataset-card caveats around gpt-3.5-turbo-generated data and commercial competing products. Review `docs/DATASETS.md` before training or publishing derived artifacts.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Training](docs/TRAINING.md)
- [Datasets](docs/DATASETS.md)
- [TUI](docs/TUI.md)
- [Model card](MODEL_CARD.md)
- [Roadmap](ROADMAP.md)
- [Release checklist](docs/RELEASE.md)
- [Security policy](SECURITY.md)
- [Disclaimer](DISCLAIMER.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Keep changes scoped, tested, and clear about data/license implications.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
