# Serious Model Checklist

Before long training:

- `scripts/check_accelerator.py` sees CUDA and RTX 3060.
- `scripts/download_fineweb2_ru.py` creates `data/raw/fineweb2_ru_sample.jsonl`.
- `scripts/train_tokenizer.py` creates `data/tokenizer/gai1_tokenizer.json`.
- `scripts/train_pretrain.py --max-steps 1` passes.
- `scripts/download_ru_turbo_alpaca.py` creates `data/raw/ru_turbo_alpaca_sample.jsonl`.
- `scripts/train_sft.py --lora --max-steps 1` saves an adapter.
- `scripts/tui.py --adapter ...` loads without errors.
- `scripts/export_quantized.py --bits 8` creates a release checkpoint.
- `scripts/eval_gates.py` passes on the target checkpoint before release.

Still quality work, not framework work:

- Much more pretraining data and steps.
- Legally clean production SFT data.
- Held-out Russian chat/reasoning/code evals.
- Real verifier/tool checks for agent behavior.
- Longer LoRA or full SFT training after the base model becomes coherent.
