from __future__ import annotations

import torch

from gai1.config import ModelConfig
from gai1.model import GAIModel


def assert_cached_next_logits_match_full(cfg: ModelConfig) -> None:
    torch.manual_seed(123)
    model = GAIModel(cfg)
    model.eval()
    idx = torch.randint(4, cfg.vocab_size, (1, min(5, cfg.block_size - 1)))
    next_id = torch.randint(4, cfg.vocab_size, (1, 1))
    with torch.no_grad():
        full_logits, _loss, _info = model(torch.cat((idx, next_id), dim=1))
        _prefill_logits, _loss, info = model(idx, use_cache=True)
        cached_logits, _loss, _info = model(next_id, past_key_values=info["past_key_values"], use_cache=True)

    torch.testing.assert_close(cached_logits[:, -1, :], full_logits[:, -1, :], atol=1e-5, rtol=1e-5)


def test_kv_cache_matches_full_forward_tiny() -> None:
    cfg = ModelConfig(block_size=16, n_layer=2, n_head=2, n_embd=32, use_moe=False, dropout=0.0)
    assert_cached_next_logits_match_full(cfg)


def test_kv_cache_matches_full_forward_gqa_rope_scaling() -> None:
    cfg = ModelConfig(
        block_size=24,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        use_moe=False,
        dropout=0.0,
        rope_scaling="linear",
        rope_scaling_factor=2.0,
        rope_original_context=8,
    )
    assert_cached_next_logits_match_full(cfg)
