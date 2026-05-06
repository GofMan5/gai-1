from __future__ import annotations

from gai1.config import ModelConfig
from gai1.lora import LoRAConfig, inject_lora, trainable_parameter_count
from gai1.model import GAIModel


def test_lora_injection_freezes_base() -> None:
    model = GAIModel(ModelConfig(vocab_size=64, block_size=8, n_layer=1, n_head=2, n_embd=32, use_moe=False))
    replaced = inject_lora(model, LoRAConfig(rank=2))
    assert replaced
    assert trainable_parameter_count(model) > 0
    assert all(("lora_" in name) == param.requires_grad for name, param in model.named_parameters())

