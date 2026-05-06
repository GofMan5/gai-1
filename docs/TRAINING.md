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

The trainer also writes:

- `outputs/gai1_train_gpu/best.pt` - best checkpoint by validation loss when
  `data.val_path` is configured, otherwise by observed train loss.
- `outputs/gai1_train_gpu/training_state.pt` - optimizer, scaler, RNG state,
  best metric, and resume step for reliable continuation.
- `outputs/gai1_train_gpu/eval_log.jsonl` - validation loss/perplexity when a
  held-out `data.val_path` is configured.

Checkpoints include tokenizer metadata and dataset hashes so an incompatible
tokenizer/data swap is visible instead of silently producing garbage.
Runtime entrypoints (`chat.py`, `serve.py`, `tui.py`, `eval_gates.py`, and
`train_sft.py`) verify tokenizer kind, actual vocab size, and tokenizer file
sha256 when checkpoint metadata is available. Older checkpoints without
tokenizer metadata load as `unknown` for compatibility. Use
`--allow-tokenizer-mismatch` only for debugging, and `--strict-tokenizer-path`
when release reproducibility requires the same tokenizer path as well as the
same hash.

When `model.use_moe=true`, train and eval logs also include routing-health
metrics: router entropy/confidence, z-loss, load CV/min/max, dead experts, and
per-expert dispatch/primary-load arrays. These fields are diagnostics for
expert collapse; they do not by themselves prove the model is good.

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
.\gai.bat chat-lora 500
```

SFT uses prompt masking: loss is applied to assistant tokens only, not to the user prompt. This is the correct path for turning the base model into a chat model.
For multi-turn `messages` records, every assistant content span is trainable
while user/system text and role headers are ignored. Long SFT records are split
into supervised windows instead of being truncated; pass `--sft-stride` to
`scripts/train_sft.py` to use overlapping windows.

Output: `outputs/gai1_sft_lora/adapter.pt`.

## 3. Run Chat

```powershell
.\gai.bat chat
```

`gai.bat chat` loads `outputs/gai1_sft_lora/adapter.pt` automatically when it exists.

## 4. Reasoning SFT

The visible TUI trace is not enough by itself. To make the model learn a
reasoning style, build teacher traces and train a separate LoRA:

```powershell
.\gai.bat reasoning-lora 1000 visible high
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

When `outputs/gai1_reasoning_lora/adapter.pt` exists, `gai.bat chat` prefers it
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
.\gai.bat quality
```

Default target is `4070` total pretrain steps, then chat LoRA and reasoning
LoRA. A stronger local target:

```powershell
.\gai.bat quality --target-pretrain-step 12208 --chat-steps 1500 --reasoning-steps 3000
```

The pipeline writes:

- `data/raw/fineweb2_ru_large.jsonl`
- `data/processed/fineweb2_ru_train.jsonl`
- `data/processed/fineweb2_ru_val.jsonl`
- `data/processed/pretrain_split_manifest.json`
- `data/raw/ru_turbo_alpaca_large.jsonl`
- `data/sft/reasoning_ru.jsonl`
- `reports/train_until_quality.jsonl`

The generated large pretrain config enables streaming JSONL reads, so the web
corpus is not loaded fully into RAM. Pretrain data is deterministically deduped
and split into train/validation files; validation loss/perplexity is used for
`best.pt` when the validation file exists. The split manifest is intentionally
small and public-friendly: it records counts, source summary, algorithm version,
and sha256 hashes for input/train/validation files.

Eval gates now fail closed: they require `--data` or `data.val_path` and refuse
to evaluate on `data.train_path`. They check held-out perplexity plus basic RU
generation quality: minimum length, Cyrillic ratio, mojibake, repetition, and
prompt echo. `gai.bat quality` returns a non-zero exit code when eval
fails unless `--allow-failed-eval` is passed.

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
.\gai.bat cycle sft 100
```

It does three things:

- resumes `outputs/gai1_sft_lora/adapter.pt` if it exists;
- trains 100 more LoRA SFT optimizer steps;
- runs fixed prompts from `evals/progress_prompts_ru.txt` and writes a report to
  `reports/progress_*.jsonl`.

Use this for gradual base-model pretraining:

```powershell
.\gai.bat cycle pretrain 100
```

For pretraining, the helper reads the current `outputs/gai1_train_gpu/last.pt`
step and launches `train_pretrain.py` with the next target step.

## RTX 3060 Profile

- CUDA with fp16 AMP.
- TF32 enabled for matmul where supported.
- Fused AdamW when the installed PyTorch build supports it.
- Cosine learning-rate schedule with warmup and minimum LR.
- Micro-batch `2`, gradient accumulation `16`.
- Effective batch `32`.
- Tokenizer: `data/tokenizer/gai1_tokenizer.json`.
- `train_log.jsonl` records tokens/sec, data wait time, VRAM, loss, and tokens seen.
- MoE configs additionally log router/load health metrics so dead or collapsed
  experts are visible during training.
- `training_state.pt` records optimizer/scaler/RNG state even when public
  checkpoints keep optimizer payloads out of `last.pt`.

Autotune micro-batch on your exact machine:

```powershell
.\gai.bat autotune
```

This writes `configs/train_gpu_autotuned.json`. Use it like:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_pretrain.py --config .\configs\train_gpu_autotuned.json
```

If VRAM is not enough, first lower `train.batch_size` from `2` to `1` in `configs/train_gpu.json`.
