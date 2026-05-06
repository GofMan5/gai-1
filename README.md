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
- Permanent experimental 256k context target with RoPE scaling and memory guards.

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

Use one launcher for normal work:

```powershell
.\gai.bat
```

## Quickstart

Recommended first path:

```powershell
.\gai.bat setup
.\gai.bat doctor
.\gai.bat prepare
.\gai.bat pretrain
.\gai.bat chat-lora 500
.\gai.bat reasoning-lora 1000 visible high
.\gai.bat chat
```

Useful commands:

```powershell
.\gai.bat
.\gai.bat list
.\gai.bat cycle sft 100
.\gai.bat quality
.\gai.bat autotune
.\gai.bat serve
.\gai.bat eval
.\gai.bat quantize 8
.\gai.bat test
```

Make it follow chat instructions with LoRA SFT:

```powershell
.\gai.bat chat-lora 500
```

Train a dedicated reasoning LoRA:

```powershell
.\gai.bat reasoning-lora 1000 visible high
```

For gradual quality checks, run 100-step cycles:

```powershell
.\gai.bat cycle sft 100
```

This resumes the existing LoRA adapter when present and writes fixed-prompt
progress reports to `reports\progress_*.jsonl`.

Run the TUI:

```powershell
.\gai.bat chat
```

`gai.bat chat` auto-loads `outputs\gai1_sft_lora\adapter.pt` when it exists.
If `outputs\gai1_reasoning_lora\adapter.pt` exists, it is preferred and launches
with the full reasoning view.

Estimate how many steps are still needed:

```powershell
.\gai.bat estimate
```

Prepare larger RU data and run the staged training pipeline:

```powershell
.\gai.bat quality
```

For a stronger local run:

```powershell
.\gai.bat quality --target-pretrain-step 12208 --chat-steps 1500 --reasoning-steps 3000
```

Autotune RTX 3060 throughput:

```powershell
.\gai.bat autotune
```

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
/reasoning full
/reasoning compact
/reasoning off
/stats
/config
/examples
/clear
/quit
```

Manual launch:

```powershell
.\.venv\Scripts\python.exe .\scripts\tui.py --checkpoint .\outputs\gai1_train_gpu\last.pt --adapter .\outputs\gai1_sft_lora\adapter.pt --level high
```

Runtime defaults to the permanent 256k target. You can pass it explicitly:

```powershell
.\gai.bat chat --context-length 262144 --rope-scaling linear
```

The current local checkpoint is not trained/evaluated at 256k. This only sets
the load-time context window with RoPE scaling. See
[Long context](docs/LONG_CONTEXT.md).

## Local API

```powershell
.\.venv\Scripts\python.exe .\scripts\serve.py --checkpoint .\outputs\gai1_train_gpu\last.pt --adapter .\outputs\gai1_sft_lora\adapter.pt
```

- Health: `http://127.0.0.1:8000/health`
- Chat: `POST http://127.0.0.1:8000/v1/chat/completions`

The API returns assistant-only text, OpenAI-style `choices` and `usage`, supports
`stop`, `top_k`, `top_p`, `repetition_penalty`, and rejects `stream=true` until
streaming is implemented.

Runtime scripts check checkpoint tokenizer metadata against the tokenizer loaded
from `--config` when metadata is available. `--allow-tokenizer-mismatch` exists
for debugging only.

## Quantization

```powershell
.\.venv\Scripts\python.exe .\scripts\export_quantized.py --checkpoint .\outputs\gai1_train_gpu\last.pt --bits 8
.\.venv\Scripts\python.exe .\scripts\export_quantized.py --checkpoint .\outputs\gai1_train_gpu\last.pt --bits 4
```

INT4 export keeps MoE router/gate tensors in fp16 by default and stores tied
`lm_head.weight` as an alias of `token_embedding.weight`. Quantized v2
artifacts are strict release files: they embed tokenizer metadata/payload,
source hashes, generation defaults, and quantization policy; the loader rejects
malformed records, unsupported bit widths, and v2 artifacts without tokenizer
metadata.

Compare FP16 vs quantized generations:

```powershell
.\.venv\Scripts\python.exe .\scripts\compare_quantized.py --quantized-checkpoint .\outputs\quantized\last_int4.pt
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

`checkpoints/` is a small public placeholder. Actual local training outputs are
kept in `outputs/` so generated weights stay out of git by default.

## Data And License Notes

Project source code is Apache-2.0 licensed. Datasets downloaded by scripts keep their own licenses and terms.

Prototype SFT uses `IlyaGusev/ru_turbo_alpaca`, which is useful for local experiments but has dataset-card caveats around gpt-3.5-turbo-generated data and commercial competing products. Review `docs/DATASETS.md` before training or publishing derived artifacts.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Training](docs/TRAINING.md)
- [Datasets](docs/DATASETS.md)
- [TUI](docs/TUI.md)
- [Long context](docs/LONG_CONTEXT.md)
- [Model card](MODEL_CARD.md)
- [Roadmap](ROADMAP.md)
- [Release checklist](docs/RELEASE.md)
- [Security policy](SECURITY.md)
- [Disclaimer](DISCLAIMER.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Keep changes scoped, tested, and clear about data/license implications.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
