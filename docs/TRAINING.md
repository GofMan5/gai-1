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

## RTX 3060 Profile

- CUDA with fp16 AMP.
- TF32 enabled for matmul where supported.
- Micro-batch `2`, gradient accumulation `16`.
- Effective batch `32`.
- Tokenizer: `data/tokenizer/gai1_tokenizer.json`.

If VRAM is not enough, first lower `train.batch_size` from `2` to `1` in `configs/train_gpu.json`.
