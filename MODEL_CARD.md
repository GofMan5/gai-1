# Model Card: GAI-1 Scaffold

## Status

GAI-1 is currently a model architecture and training pipeline. The repository
does not ship production weights.

## Intended Use

- Local research and experimentation.
- Russian-first tokenizer and language-model training.
- Testing pretraining, SFT, LoRA, quantization, and TUI/API serving paths.

## Out of Scope

- Production assistant deployment without additional training and evaluation.
- High-stakes advice.
- Safety-critical automation.
- Claims of parity with commercial frontier models.

## Architecture

- Decoder-only Transformer.
- RoPE positional encoding.
- RMSNorm.
- SwiGLU feed-forward layers.
- Optional small MoE path.
- LoRA adapter support.

## Training Data

Training scripts download datasets from their original public sources. Dataset
licenses are separate from this repository license. See `docs/DATASETS.md`.

## Evaluation

The repository includes lightweight eval gates, but they are not enough for a
production release. A serious release needs held-out evaluations for:

- Russian chat quality;
- instruction following;
- reasoning;
- code;
- hallucinations;
- safety;
- regression behavior.

## Limitations

- Quality depends on data volume, data quality, and training time.
- Short smoke runs will produce incoherent generations.
- 128k context is an experimental target. A checkpoint must be tuned and
  evaluated at 128k before being described as 128k-ready.
- The reasoning trace shown in TUI is a runtime scaffold, not hidden model
chain-of-thought.
- Quantized exports are storage-oriented and not a replacement for optimized
inference engines such as vLLM or TensorRT-LLM.
