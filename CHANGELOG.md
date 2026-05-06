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
