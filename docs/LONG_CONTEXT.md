# Long Context Plan

GAI-1 now has a permanent experimental 256k context target, but the current
local RTX 3060 training path is not a validated 256k training stack.

## Current State

- Current local checkpoint was trained at 768 tokens.
- Runtime loading defaults to the project target of `--context-length 262144`.
- RoPE can be extended with `--rope-scaling linear` or `dynamic_ntk`.
- TUI shows both total accumulated tokens and visible context tokens.

This means the model can be loaded with a larger context window, but quality at
256k requires context-extension training and evaluation.

## Why 256k Is Hard

Full causal attention is O(n^2). A 256k sequence is not just 341x longer than
768 tokens; attention score work and memory grow quadratically. On RTX 3060
12GB, full-attention 256k training is not realistic.

## Experimental Config

```text
configs/train_256k_experimental.json
```

This config is a blueprint. `scripts/train_pretrain.py` includes a memory guard
that refuses unsafe full-attention launches by default.

## Real Path To 256k

1. Train a coherent base model at short context.
2. Extend to 8k.
3. Tune/evaluate at 32k.
4. Tune/evaluate at 256k with long-document data.
5. Use a distributed stack with sequence parallelism, efficient attention, and
   long-context eval gates.

Release gates for calling a checkpoint 256k-ready:

- needle retrieval at 256k;
- long QA at 256k;
- memory benchmark;
- inference latency benchmark;
- documented training data and context-extension method.

## Local TUI Launch

```powershell
.\run_tui.bat --context-length 262144 --rope-scaling linear
```

This is useful for experimentation, not proof that the model understands 256k
context.
