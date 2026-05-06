# GAI-1 Architecture

GAI-1 is a model factory scaffold, not a single script.

## Layers

1. `model core` - decoder-only Transformer/MoE.
2. `data engine` - ingestion, cleaning, tokenizer, shards, manifests.
3. `training` - pretrain, continued pretrain, SFT, preference tuning later.
4. `reasoning runtime` - planner, drafts, critic, verifier, rollback, final renderer.
5. `serving` - fp16/int8/int4 checkpoints, later vLLM or TensorRT-LLM.

## Current Core

- RMSNorm.
- RoPE with cache and optional scaling.
- PyTorch SDPA causal attention.
- Optional GQA through `n_kv_head`.
- SwiGLU.
- Optional simple top-k MoE.
- AMP fp16 training.
- Optional gradient checkpointing.
- Gradient accumulation.
- fp16 checkpoint storage.
- int8/int4 quantized release export.

## Long Context

The target path includes 8k, 32k, and 256k context stages. Current code can
load a checkpoint with a larger context window using RoPE scaling, but a
checkpoint is not considered 256k-ready until it has been context-tuned and
evaluated at that length.

Full-attention 256k training is not realistic on RTX 3060 12GB. See
`docs/LONG_CONTEXT.md`.

## RTX 3060 12GB Training

Canonical local GPU training config:

```text
configs/train_gpu.json
```

It uses CUDA, fp16 AMP, TF32, pinned memory, and gradient accumulation.

## Reasoning Levels

Reasoning modes `low`, `medium`, `high`, `max` control a runtime loop:

- planning depth;
- draft count;
- critic/verifier passes;
- rollback budget;
- tool budget;
- private token budget;
- self-consistency.

Profiles live in `configs/reasoning_modes.json`.

## Quantization

Quantization reduces bytes per parameter, not parameter count:

- fp32: 4 bytes;
- fp16/bf16: 2 bytes;
- int8: 1 byte;
- int4: 0.5 bytes.

Current `export_quantized.py` is storage quantization. Real inference VRAM
savings need quantized kernels through an inference engine or dedicated int4/int8
linear layers.

## Next Improvements

1. Larger RU-first tokenizer.
2. Packed binary data shards.
3. QLoRA path.
4. Long-context eval gates.
5. vLLM-compatible export.
6. MoE training through Megatron/NeMo when the local dense path is stable.
