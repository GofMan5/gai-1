# GAI-1 Training

The project keeps one primary GPU training path to avoid confusion.

## 1. Pretrain

```powershell
cd .\gai-1
.\.venv\Scripts\python.exe .\scripts\download_fineweb2_ru.py --max-docs 1200
.\.venv\Scripts\python.exe .\scripts\train_tokenizer.py
.\.venv\Scripts\python.exe .\scripts\train_pretrain.py
```

Config: `configs/train_gpu.json`.

Output: `outputs/gai1_train_gpu/last.pt`.

`checkpoints/` is not the active training output directory. It is only a small
public placeholder. Inspect local weights with:

```powershell
.\.venv\Scripts\python.exe .\scripts\list_artifacts.py
```

## 2. Chat SFT

```powershell
.\.venv\Scripts\python.exe .\scripts\download_ru_turbo_alpaca.py --max-records 5000
.\.venv\Scripts\python.exe .\scripts\train_sft.py --checkpoint .\outputs\gai1_train_gpu\last.pt --data .\data\raw\ru_turbo_alpaca_sample.jsonl --lora --max-steps 500
```

Helper:

```powershell
.\train_chat_lora.bat 500
```

SFT uses prompt masking: loss is applied to assistant tokens only, not to the user prompt. This is the correct path for turning the base model into a chat model.

Output: `outputs/gai1_sft_lora/adapter.pt`.

## 3. Run Chat

```powershell
.\run_tui.bat
```

`run_tui.bat` loads `outputs/gai1_sft_lora/adapter.pt` automatically when it exists.

## 4. Reasoning SFT

The visible TUI trace is not enough by itself. To make the model learn a
reasoning style, build teacher traces and train a separate LoRA:

```powershell
.\train_reasoning_lora.bat 1000 visible high
```

Outputs:

- `data/sft/reasoning_ru.jsonl` - generated reasoning-SFT records.
- `outputs/gai1_reasoning_lora/adapter.pt` - reasoning adapter.
- `outputs/gai1_reasoning_lora/manifest.json` - step/loss/dataset/base hashes.
- `outputs/gai1_reasoning_lora/train_log.jsonl` - per-step training log.

Modes:

- `visible` trains short public reasoning plus final answer.
- `controller` injects a hidden-controller plan into the prompt and trains only
  the final answer.

When `outputs/gai1_reasoning_lora/adapter.pt` exists, `run_tui.bat` prefers it
over the plain chat LoRA and starts with `--reasoning-view full`.

## How Many Steps

Current local checkpoint is only `step=100`. With the RTX 3060 config, one
optimizer step sees about `32 * 768 = 24,576` tokens, so the current base has
seen only about `2.46M` tokens.

Practical local targets:

- Barely coherent RU chat base: `100M-300M` pretrain tokens, about `4k-12k`
  total pretrain steps.
- Stronger small local model: `500M-1B` pretrain tokens, about `20k-41k` total
  pretrain steps.
- Chat LoRA on 5k records: about `500-2k` steps for first useful behavior.
- Reasoning LoRA on 5k teacher-trace records: about `1k-3k` steps for the first
  visible improvement, `3k-10k` for more stable behavior.

Use the estimator:

```powershell
.\.venv\Scripts\python.exe .\scripts\estimate_training_steps.py
```

Prepare a larger RU pack and run the staged pipeline:

```powershell
.\train_until_quality.bat
```

Default target is `4070` total pretrain steps, then chat LoRA and reasoning
LoRA. A stronger local target:

```powershell
.\train_until_quality.bat --target-pretrain-step 12208 --chat-steps 1500 --reasoning-steps 3000
```

The pipeline writes:

- `data/raw/fineweb2_ru_large.jsonl`
- `data/raw/ru_turbo_alpaca_large.jsonl`
- `data/sft/reasoning_ru.jsonl`
- `reports/train_until_quality.jsonl`

The generated large pretrain config enables streaming JSONL reads, so the web
corpus is not loaded fully into RAM. Eval gates now check perplexity plus basic
RU generation quality: minimum length, Cyrillic ratio, mojibake, repetition, and
prompt echo.

This still will not make the model Claude-level. Claude-like reasoning requires
massive pretraining, curated reasoning traces, preference tuning, verifier/eval
gates, and much larger compute. This repo now has the local training path for
that direction, not the final capability.

## Incremental 100-Step Cycles

Language-model training is controlled by optimizer steps, not classic epochs.
With the current sliding-token dataset, a real epoch is not a useful unit for
quick experiments.

Use this for gradual chat tuning:

```powershell
.\train_100.bat sft 100
```

It does three things:

- resumes `outputs/gai1_sft_lora/adapter.pt` if it exists;
- trains 100 more LoRA SFT optimizer steps;
- runs fixed prompts from `evals/progress_prompts_ru.txt` and writes a report to
  `reports/progress_*.jsonl`.

Use this for gradual base-model pretraining:

```powershell
.\train_100.bat pretrain 100
```

For pretraining, the helper reads the current `outputs/gai1_train_gpu/last.pt`
step and launches `train_pretrain.py` with the next target step.

## RTX 3060 Profile

- CUDA with fp16 AMP.
- TF32 enabled for matmul where supported.
- Fused AdamW when the installed PyTorch build supports it.
- Micro-batch `2`, gradient accumulation `16`.
- Effective batch `32`.
- Tokenizer: `data/tokenizer/gai1_tokenizer.json`.
- `train_log.jsonl` records tokens/sec, data wait time, VRAM, loss, and tokens seen.

Autotune micro-batch on your exact machine:

```powershell
.\autotune_rtx3060.bat
```

This writes `configs/train_gpu_autotuned.json`. Use it like:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_pretrain.py --config .\configs\train_gpu_autotuned.json
```

If VRAM is not enough, first lower `train.batch_size` from `2` to `1` in `configs/train_gpu.json`.
