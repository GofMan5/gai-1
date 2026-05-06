from __future__ import annotations

import torch

from gai1.config import ModelConfig
from gai1.model import GAIModel
from gai1.training import flatten_moe_metrics


def test_moe_forward_reports_routing_health() -> None:
    cfg = ModelConfig(block_size=16, n_layer=2, n_head=2, n_embd=32, use_moe=True, n_experts=4, n_experts_per_token=2)
    model = GAIModel(cfg)
    x = torch.randint(4, cfg.vocab_size, (2, 16))
    _logits, loss, info = model(x, x)

    assert loss is not None
    assert torch.isfinite(info["moe_aux_loss"])
    assert torch.isfinite(info["moe_router_z_loss"])
    assert torch.isfinite(info["moe_router_entropy"])
    assert torch.isfinite(info["moe_load_cv"])
    assert info["moe_dispatch_load"].numel() == cfg.n_experts
    assert info["moe_primary_load"].numel() == cfg.n_experts
    assert info["moe_importance"].numel() == cfg.n_experts
    torch.testing.assert_close(info["moe_dispatch_load"].sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(info["moe_primary_load"].sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)


def test_dense_forward_keeps_moe_metrics_empty_except_zero_losses() -> None:
    cfg = ModelConfig(block_size=8, n_layer=1, n_head=2, n_embd=32, use_moe=False)
    model = GAIModel(cfg)
    x = torch.randint(4, cfg.vocab_size, (1, 8))
    _logits, loss, info = model(x, x)

    assert loss is not None
    assert float(info["moe_aux_loss"]) == 0.0
    assert float(info["moe_router_z_loss"]) == 0.0
    assert "moe_dispatch_load" not in info


def test_moe_z_loss_weight_changes_loss() -> None:
    cfg = ModelConfig(
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=32,
        use_moe=True,
        n_experts=2,
        moe_aux_loss_weight=0.0,
        moe_z_loss_weight=0.1,
    )
    model = GAIModel(cfg)
    x = torch.randint(4, cfg.vocab_size, (1, 8))
    logits, loss, info = model(x, x)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1), ignore_index=-100)

    assert loss is not None
    torch.testing.assert_close(loss, ce + cfg.moe_z_loss_weight * info["moe_router_z_loss"], atol=1e-5, rtol=1e-5)


def test_flatten_moe_metrics_serializes_scalars_and_vectors() -> None:
    info = {
        "moe_router_entropy": torch.tensor(0.5),
        "moe_load_cv": torch.tensor(0.1),
        "moe_dispatch_load": torch.tensor([0.25, 0.75]),
    }

    row = flatten_moe_metrics(info, prefix="val_")

    assert row["val_moe_router_entropy"] == 0.5
    assert row["val_moe_load_cv"] == 0.10000000149011612
    assert row["val_moe_dispatch_load"] == [0.25, 0.75]
