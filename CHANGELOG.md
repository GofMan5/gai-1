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
