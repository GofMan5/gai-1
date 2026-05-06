# GAI-1 Scaling Plan

## Stage 1: Tiny

Цель: проверить код, checkpoints, формат данных, генерацию.

- byte tokenizer;
- tiny Transformer/MoE;
- локальный JSONL dataset;
- smoke tests.

## Stage 2: Small Dense

Цель: первая реальная русская base/chat модель.

- SentencePiece/BPE tokenizer;
- 100M-1B dense decoder;
- clean RU/EN/code mixture;
- perplexity eval;
- SFT chat.

## Stage 3: Useful Dense

Цель: модель, которую можно использовать в задачах.

- 7B-14B dense;
- continued pretraining на RU/code/math;
- SFT + DPO;
- vLLM serving;
- regression eval gates.

## Stage 4: Reasoning Agent

Цель: поведение уровня "план -> проверка -> ответ".

- tool-use traces;
- verifier для кода/математики;
- self-refine runtime;
- RL только на задачах с проверяемым reward.

## Stage 5: MoE

Цель: sparse scaling.

- сначала 16 experts;
- затем 32-64 experts;
- expert utilization monitoring;
- expert parallel training;
- distributed MoE serving.

## Stage 6: 5T Total Parameters

Цель: долгосрочный frontier-style target.

Это возможно только как MoE на кластере с быстрым interconnect, sharded checkpoints, стабильным data engine и continuous evals. Dense 5T не является практичной целью.

