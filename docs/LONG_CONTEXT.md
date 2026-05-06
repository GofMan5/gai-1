# Long Context Plan

GAI-1 now has an experimental 128k context target, but the current local RTX
3060 training path is not a validated 128k training stack.

## Current State

- Current local checkpoint was trained at 768 tokens.
- `scripts/tui.py` can load a checkpoint with `--context-length 131072`.
- RoPE can be extended with `--rope-scaling linear` or `dynamic_ntk`.
- TUI shows both total accumulated tokens and visible context tokens.

This means the model can be loaded with a larger context window, but quality at
128k requires context-extension training and evaluation.

## Why 128k Is Hard

Full causal attention is O(n^2). A 128k sequence is not just 166x longer than
768 tokens; attention score work and memory grow quadratically. On RTX 3060
12GB, full-attention 128k training is not realistic.

## Experimental Config

```text
configs/train_128k_experimental.json
```

This config is a blueprint. `scripts/train_pretrain.py` includes a memory guard
that refuses unsafe full-attention launches by default.

## Real Path To 128k

1. Train a coherent base model at short context.
2. Extend to 8k.
3. Tune/evaluate at 32k.
4. Tune/evaluate at 128k with long-document data.
5. Use a distributed stack with sequence parallelism, efficient attention, and
   long-context eval gates.

Release gates for calling a checkpoint 128k-ready:

- needle retrieval at 128k;
- long QA at 128k;
- memory benchmark;
- inference latency benchmark;
- documented training data and context-extension method.

## Local TUI Launch

```powershell
.\run_tui.bat --context-length 131072 --rope-scaling linear
```

This is useful for experimentation, not proof that the model understands 128k
context.
