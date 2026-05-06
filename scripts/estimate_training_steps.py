from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate GAI-1 training steps from config, checkpoint, and dataset size.")
    parser.add_argument("--config", default="configs/train_gpu.json")
    parser.add_argument("--checkpoint", default="outputs/gai1_train_gpu/last.pt")
    parser.add_argument("--sft-records", type=int, default=5000)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    if not path.exists():
        return 0
    payload = torch.load(path, map_location="cpu")
    return int(payload.get("step", 0))


def jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def estimate_pretrain_tokens(cfg: Any, steps: int) -> int:
    return int(cfg.train.batch_size) * int(cfg.train.gradient_accumulation_steps) * int(cfg.data.block_size) * steps


def steps_for_tokens(cfg: Any, target_tokens: int) -> int:
    tokens_per_step = int(cfg.train.batch_size) * int(cfg.train.gradient_accumulation_steps) * int(cfg.data.block_size)
    return math.ceil(target_tokens / max(1, tokens_per_step))


def sft_steps_for_epochs(cfg: Any, records: int, epochs: int) -> int:
    examples_per_step = int(cfg.train.batch_size) * int(cfg.train.gradient_accumulation_steps)
    return math.ceil(records * epochs / max(1, examples_per_step))


def main() -> int:
    args = parse_args()
    cfg = load_config(ROOT / args.config)
    step = checkpoint_step(ROOT / args.checkpoint)
    seen_tokens = estimate_pretrain_tokens(cfg, step)
    records = args.sft_records
    default_sft_path = ROOT / "data/raw/ru_turbo_alpaca_sample.jsonl"
    if default_sft_path.exists():
        records = jsonl_records(default_sft_path) or records

    report = {
        "checkpoint": args.checkpoint,
        "current_pretrain_step": step,
        "tokens_per_optimizer_step": int(cfg.train.batch_size) * int(cfg.train.gradient_accumulation_steps) * int(cfg.data.block_size),
        "approx_seen_pretrain_tokens": seen_tokens,
        "local_quality_targets": {
            "minimum_coherent_ru_chat": {
                "pretrain_tokens": "100M-300M",
                "pretrain_steps_total": f"{steps_for_tokens(cfg, 100_000_000)}-{steps_for_tokens(cfg, 300_000_000)}",
                "more_steps_from_now": f"{max(0, steps_for_tokens(cfg, 100_000_000) - step)}-{max(0, steps_for_tokens(cfg, 300_000_000) - step)}",
            },
            "stronger_small_model": {
                "pretrain_tokens": "500M-1B",
                "pretrain_steps_total": f"{steps_for_tokens(cfg, 500_000_000)}-{steps_for_tokens(cfg, 1_000_000_000)}",
                "more_steps_from_now": f"{max(0, steps_for_tokens(cfg, 500_000_000) - step)}-{max(0, steps_for_tokens(cfg, 1_000_000_000) - step)}",
            },
            "chat_sft_lora": {
                "records": records,
                "recommended_epochs": "3-10",
                "optimizer_steps": f"{sft_steps_for_epochs(cfg, records, 3)}-{sft_steps_for_epochs(cfg, records, 10)}",
            },
            "reasoning_sft_lora": {
                "records": records,
                "recommended_epochs": "5-20",
                "optimizer_steps": f"{sft_steps_for_epochs(cfg, records, 5)}-{sft_steps_for_epochs(cfg, records, 20)}",
            },
        },
        "hard_truth": "This is not Claude-level reasoning. It is a small local model plus a reasoning controller; real reasoning quality needs reasoning-SFT data, eval gates, and far more tokens.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
