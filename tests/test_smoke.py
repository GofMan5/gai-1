from __future__ import annotations

import torch

from gai1.config import ModelConfig
from gai1.model import GAIModel
from gai1.tokenizer import ByteTokenizer


def test_byte_tokenizer_roundtrip_ru() -> None:
    tokenizer = ByteTokenizer()
    text = "Привет, GAI-1"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_model_forward_tiny() -> None:
    cfg = ModelConfig(block_size=16, n_layer=1, n_head=2, n_embd=32, use_moe=True, n_experts=2)
    model = GAIModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, info = model(x, x)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None
    assert "moe_aux_loss" in info


def test_model_forward_with_gqa_and_rope_scaling() -> None:
    cfg = ModelConfig(
        block_size=32,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        use_moe=False,
        rope_scaling="linear",
        rope_scaling_factor=4.0,
        rope_original_context=8,
    )
    model = GAIModel(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    logits, loss, _info = model(x, x)
    assert logits.shape == (1, 16, cfg.vocab_size)
    assert loss is not None
