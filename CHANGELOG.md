# Changelog

## 0.1.0 - 2026-05-06

Initial public scaffold:

- Russian-first BPE tokenizer path.
- CUDA pretraining path for RTX 3060 12GB.
- Masked SFT dataset and LoRA fine-tuning.
- LoRA adapter loading in CLI/TUI.
- Reasoning effort levels: `low`, `medium`, `high`, `max`.
- Terminal chat UI with token and VRAM stats.
- Quantized checkpoint export.
- Local API server scaffold.
- Tests and open-source project metadata.
- Experimental 128k context loading path with RoPE scaling, GQA config support,
  TUI visible-context stats, and training memory guards.
- Incremental `train_100.bat` / `scripts/train_cycle.py` workflow for 100-step
  progress checks with fixed prompts.
- Quantized checkpoint v2 export with tied-weight aliases, tokenizer artifact
  metadata, relative source paths, source SHA256, generation config, and fp16
  MoE router/gate weights for INT4.
- TUI refresh cleanup, command suggestions, clearer turn separation, structured
  reasoning panel, grouped runtime stats, and repetition penalty.
- TUI startup loading screen with checkpoint, adapter, model, tokenizer,
  reasoning-profile, and warmup progress.
- TUI full reasoning view with plan, drafts, critic, verifier, and prompt
  injection so effort levels affect generation instead of only side-panel UI.
- Reasoning-SFT dataset builder, reasoning LoRA trainer, step estimator, and
  TUI auto-load path for `outputs/gai1_reasoning_lora/adapter.pt`.
- SFT/LoRA training metadata now records step, final loss, dataset hash,
  dataset record count, base checkpoint hash, and JSONL training logs.
- Model-backed TUI reasoning drafts, larger RU data preparation, and staged
  `train_until_quality.bat` pipeline for pretrain + chat LoRA + reasoning LoRA.
- Streaming pretrain dataset path for large JSONL corpora and stronger RU
  generation eval gates for mojibake, Cyrillic ratio, repetition, and prompt echo.
- RTX 3060 training optimization: fused AdamW support, JSONL throughput logs,
  dataloader wait metrics, VRAM tracking, and micro-batch autotune helper.
