# Checkpoints Directory

This directory is intentionally kept as a release/registry placeholder.

Local training artifacts are written to `outputs/` by default:

- Base model: `outputs/gai1_train_gpu/last.pt`
- Chat LoRA: `outputs/gai1_sft_lora/adapter.pt`
- Reasoning LoRA: `outputs/gai1_reasoning_lora/adapter.pt`
- Quantized exports: `outputs/quantized/*.pt`

The repository does not commit trained weights. `checkpoints/` stays small in
git so public clones do not accidentally include huge private artifacts.

To inspect local artifacts, run:

```powershell
.\.venv\Scripts\python.exe .\scripts\list_artifacts.py
```
