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
- Permanent experimental 256k context loading path with RoPE scaling, GQA config support,
  TUI visible-context stats, and training memory guards.
- Incremental `gai.bat cycle` / `scripts/train_cycle.py` workflow for 100-step
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
  `gai.bat quality` pipeline for pretrain + chat LoRA + reasoning LoRA.
- Streaming pretrain dataset path for large JSONL corpora and stronger RU
  generation eval gates for mojibake, Cyrillic ratio, repetition, and prompt echo.
- RTX 3060 training optimization: fused AdamW support, JSONL throughput logs,
  dataloader wait metrics, VRAM tracking, and micro-batch autotune helper.
- Training-core hardening: cosine LR warmup/decay, reliable optimizer/scaler/RNG
  resume state, best checkpoint/adapters, tokenizer and dataset provenance in
  checkpoints, and optional validation loss/perplexity logging.
- Data/eval hardening: deterministic JSONL dedupe + train/validation split
  manifest, large-config validation path, held-out eval-gate default, fixed RU
  Cyrillic/mojibake heuristics, and repaired Russian seed/eval text.
- Release gates now fail closed instead of silently falling back to train data;
  split manifests include sha256 provenance and non-empty train/validation
  checks.
- Inference hardening: transformer KV-cache decode path, shared suffix-only
  generation helper with stop tokens/strings and usage stats, cleaner chat CLI,
  and OpenAI-style local API responses with adapter/device/context load args.
- SFT data quality hardening: all assistant turns in `messages` records now
  receive loss, long records are split into supervised windows with optional
  stride/overlap, SFT manifests log supervised/ignored token counts, and small
  datasets no longer hang behind `drop_last=True`.
- MoE telemetry hardening: router z-loss support, per-expert importance/load
  metrics, entropy/confidence/load-CV/dead-expert diagnostics, and pretrain/SFT
  JSONL logging for routing health.
- Tokenizer compatibility hardening: runtime scripts verify checkpoint tokenizer
  kind/vocab/hash against the actual loaded tokenizer, expose debug override
  flags, and SFT artifacts now carry tokenizer compatibility metadata.
- Quantized release hardening: export now fails closed on missing tokenizer
  artifacts, fixes UTF-8 generation metadata, records explicit router dtype
  policy, and validates quantized records/aliases before model load.
- Runtime text integrity hardening: TUI separators are ASCII-stable and tests
  now fail on common mojibake markers in source/config/eval text.
- Context target hardening: project target is fixed at 262144 tokens, runtime
  load defaults resolve to that target, oversized context requests are rejected,
  and long-context validation metadata no longer overclaims without eval proof.
- Unified Windows launcher: `gai.bat` is now the single root batch entrypoint
  for setup, data/tokenizer prep, training, chat, serving, eval, quantization,
  artifacts, tests, and diagnostics.
- Launcher UX cleanup: top-level `gai.bat` menu is now goal-based with Russian
  explanations, technical commands moved into submenus, and the batch file uses
  UTF-8 + CRLF for stable Windows `cmd.exe` labels.
