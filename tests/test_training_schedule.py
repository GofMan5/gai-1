from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from gai1.config import GAIConfig
import train_pretrain
import train_sft


def test_pretrain_cosine_schedule_warms_up_and_decays() -> None:
    cfg = GAIConfig()
    cfg.train.learning_rate = 1.0
    cfg.train.min_learning_rate = 0.1
    cfg.train.warmup_steps = 2
    cfg.train.max_steps = 10
    cfg.train.lr_scheduler = "cosine"

    assert train_pretrain.learning_rate_at_step(cfg, 0) == 0.5
    assert train_pretrain.learning_rate_at_step(cfg, 1) == 1.0
    assert train_pretrain.learning_rate_at_step(cfg, 10) == 0.1


def test_sft_schedule_uses_total_target_steps() -> None:
    cfg = GAIConfig()
    cfg.train.learning_rate = 1.0
    cfg.train.min_learning_rate = 0.0
    cfg.train.warmup_steps = 0
    cfg.train.lr_scheduler = "cosine"

    early = train_sft.learning_rate_at_step(cfg, 5, max_steps=20)
    later = train_sft.learning_rate_at_step(cfg, 15, max_steps=20)

    assert early > later


def test_set_optimizer_lr_updates_all_param_groups() -> None:
    p1 = torch.nn.Parameter(torch.tensor([1.0]))
    p2 = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW(
        [
            {"params": [p1], "lr": 1.0},
            {"params": [p2], "lr": 0.5},
        ]
    )

    train_pretrain.set_optimizer_lr(optimizer, 0.123)

    assert [group["lr"] for group in optimizer.param_groups] == [0.123, 0.123]
