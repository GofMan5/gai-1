# GAI-1 Architecture

GAI-1 строится как модельная фабрика, а не один скрипт.

## Контуры

1. `model core` - decoder-only Transformer/MoE.
2. `data engine` - ingestion, cleaning, dedupe, tokenizer, shards, manifests.
3. `training` - pretrain, continued pretrain, SFT, DPO/RL later.
4. `reasoning runtime` - planner, drafts, critic, verifier, rollback, final renderer.
5. `serving` - fp16/int8/int4 checkpoints, later vLLM/TensorRT-LLM.

## Текущее Ядро

- RMSNorm.
- RoPE with cache.
- PyTorch SDPA causal attention.
- SwiGLU.
- Optional simple top-k MoE.
- AMP fp16 training.
- Gradient accumulation.
- fp16 checkpoint storage.
- int8/int4 quantized release export.

## RTX 3060 12GB Training

Есть один канонический GPU-тренинг:

- `configs/train_gpu.json`

Он использует CUDA, fp16 AMP, TF32, pinned memory и gradient accumulation. 32GB RAM используется для dataloader/cache без бессмысленного раздувания. Другие локальные training-конфиги убраны, чтобы не было путаницы.

## Reasoning Levels

Режимы `low`, `medium`, `high`, `max` управляют не магией модели, а runtime-циклом:

- глубина плана;
- число черновиков;
- число critic/verifier passes;
- rollback budget;
- tool budget;
- private token budget;
- self-consistency.

Профили лежат в `configs/reasoning_modes.json`. Добавить новый уровень можно через JSON без правки Python-кода, если набор полей тот же.

## Quantization

Квантование не уменьшает число параметров. Оно уменьшает байты на параметр:

- fp32: 4 bytes;
- fp16/bf16: 2 bytes;
- int8: 1 byte;
- int4: 0.5 bytes.

Текущий `export_quantized.py` делает storage quantization. Для настоящей экономии VRAM на inference нужен следующий слой: quantized kernels через vLLM/TensorRT-LLM или отдельные int4/int8 linear layers.

## Что Улучшать Дальше

1. Нормальный RU-first BPE/SentencePiece tokenizer вместо byte tokenizer.
2. Packed binary shards вместо чтения JSONL в память.
3. LoRA/QLoRA adapters отдельно от base checkpoint.
4. Eval gates для русского чата, reasoning, code и safety.
5. vLLM-compatible export.
6. MoE training через Megatron/NeMo, когда локальный dense pipeline стабилен.
